import numpy as np
import math
import scipy
from scipy import linalg
from scipy.optimize import minimize
from interpolation_models.core import nearestPD as NPD
from interpolation_models.core import constrNMPy as cNM
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import MinMaxScaler as MS
from sklearn.metrics.pairwise import check_pairwise_arrays
import cma

'''
Improvements:
- in-house preprocessing

'''
class Kriging:

    def __init__(self,x,y,kernels,theta0="",weight0="",optimizer="SLSQP",optimizer_noise=1.0,eps=1.48e-08,restarts=1,preprocessing="standardize"):
        if preprocessing == "normalize":
            self.x_scaler = MS()
            self.y_scaler = MS()
            self.y_std = 1.0
        elif preprocessing == "standardize":
            self.x_scaler = SS()
            self.y_scaler = SS()
            self.y_std = np.std(y.copy())

        try:
            self.x_scaler.fit(x)

        except:
            x = x[:,np.newaxis] #hack for 1D
            self.x_scaler.fit(x)

        try:
            self.y_scaler.fit(y)

        except:
            y = y[:,np.newaxis] #hack for 1D
            self.y_scaler.fit(y)

        self.x = self.x_scaler.transform(x)
        self.y = self.y_scaler.transform(y)

        self.kernels = kernels
        self.optimizer = optimizer
        self.optimizer_noise = optimizer_noise
        self.eps = eps
        self.restarts = restarts
        self.Ns = self.x.shape[0]
        self.Nk = self.x.shape[1]
        self.likelihood_threshold = 30
        self.likelihood_w = -1
        self.likelihood = -1
        if (weight0==""):
            weight0 = [1/len(self.kernels)] * len(self.kernels)
        weights = weight0
        for i in range(0,self.Nk-1):
            weights = np.vstack((weights,weight0))
        self.weights0 = weights

        if theta0 == "": # to ensure varying initial guesses across board
            theta0 = []
            for i in range(self.Nk):
                theta0.append(np.random.uniform(1e-2,5))
        else:
            theta0 = theta0
        self.theta0 = theta0

        if self.optimizer == "CMA-ES" and self.Nk == 1: # hack to avoid certain errors
            self.optimizer = "SLSQP"



    def compute_K(self,D,kernel,theta,weights):  
        kernels = self.kernels
        K = np.ones((D.shape[0],1))
        #weights input must have been flattened
        weights = np.array_split(weights,len(weights))
        for i in range(self.Nk):
            d_comp = (D[:,i]).reshape(D.shape[0],1) #componentwise distance
            K_w = np.zeros((D.shape[0],1))
            for w in range(len(kernels)):
                kernel = kernels[w]
                weight = weights.pop(0)

                if kernel == "exponential":
                    K_k = np.exp(-1.0 * np.abs(d_comp))
                elif kernel == "matern3_2":
                    K_k = (1 + np.sqrt(3)*np.abs(d_comp)) * np.exp(-np.sqrt(3)*np.abs(d_comp))
                elif kernel == "gaussian":
                    K_k = np.exp(-0.5 * (d_comp**2))
                elif kernel == "matern5_2":
                    K_k = (1 + np.sqrt(5)*np.abs(d_comp) + (5/3*(d_comp**2))) * np.exp((-np.sqrt(5))*np.abs(d_comp))
                else:
                    print("Unknown kernel")
                K_w += weight * K_k
            K *= K_w
        return K

    def compute_componentwise_distance(self,D,theta):
        if isinstance(theta, list):
            theta = np.asarray(theta)
        else:
            theta = theta
        D_corr = np.einsum("j,ij->ij", (1/theta).T, D)
        return D_corr    

    def compute_rr(self,D,theta,weights):
        self.D = self.compute_componentwise_distance(D,theta)
        r = self.compute_K(self.D,self.kernels,theta,weights)                                                                                                                                                             
        return r

    def NLL(self, hyperparameter):
        nugget = 2.22e-11 # a very small value
        theta = np.zeros(len(self.theta0))
        hyperparameter = np.array_split(hyperparameter,len(hyperparameter))
        for i in range(len(self.theta0)):
            theta[i] = hyperparameter.pop(0)
        weights = np.concatenate(hyperparameter)

        y = self.y
        n = len(y)

        # Calculate matrix of distances D between samples
        D, self.ij = cross_distances(self.x)
        # compute the correlation matrix
        r = self.compute_rr(D,theta,weights)
        R = np.eye(self.Ns) * (1.0 + nugget)
        R[self.ij[:, 0], self.ij[:, 1]] = r[:, 0]
        R[self.ij[:, 1], self.ij[:, 0]] = r[:, 0]

        y = self.y
        n = len(y)
        self.F = np.ones(self.Ns)[:,np.newaxis]
        self.R = R # so I can reuse this R at any point in the code
        # Cholesky decomposition of R
        try:
            C = linalg.cholesky(self.R, lower=True)
        except (linalg.LinAlgError, ValueError) as e:
            print("exception : ", e)
            self.R = NPD.nearestPD(self.R)
            C = np.linalg.cholesky(self.R) #using cholesky decomposition from the numpy library helps sometimes
            # raise e        
        # Get generalized least squared solution
        Ft = linalg.solve_triangular(C, self.F, lower=True)
        Q, G = linalg.qr(Ft, mode="economic")
        Yt = linalg.solve_triangular(C, self.y, lower=True)
        self.beta = linalg.solve_triangular(G, np.dot(Q.T, Yt))
        rho = Yt - np.dot(Ft, self.beta)
        sigma2 = (rho ** 2.0).sum(axis=0) / (n)
        self.gamma = linalg.solve_triangular(C.T, rho)
        self.sigma2 = sigma2 * self.y_std ** 2.0 #wrong
        self.G = G
        self.C = C 
        self.Ft = Ft
        # The determinant of R is equal to the squared product of the diagonal
        # elements of its Cholesky decomposition C
        detR = (np.diag(C) ** (2.0 / n)).prod()
        nll = n * np.log10(sigma2.sum()) + n * np.log10(detR)
        return nll

    def constraint_func(self,hyperparameter):
        theta = np.zeros(len(self.theta0))
        hyperparameter = np.array_split(hyperparameter,len(hyperparameter))
        for i in range(len(self.theta0)):
            theta[i] = hyperparameter.pop(0)
        weights = np.concatenate(hyperparameter)
        weights = weights.tolist()
        weights = np.array_split(weights,len(weights))
        w = []
        w_d = []
        for o in range(self.Nk):
            w.append([])
        for o in range(self.Nk):
            for j in range(len(self.kernels)):
                x = weights.pop(0)
                w[o].append(float(x))    
        w = np.array(w)   
        i = np.linspace(0,self.Nk-1,self.Nk)
        i = i.tolist()
        for t in range(len(i)):
            i[t] = int(i[t])
        w_d = w.sum(1)[i] - 1  
        # return (np.dot(w,w.T) - np.eye(self.Nk))
        return w_d

    def get_theta(self,hyperparameter):
        xk = hyperparameter
        # print(xk)
        # cons = ({'type':'eq','fun':self.constraint_func})

        bounds = []
        theta_bound = (1e-4,1e3)
        weight_bound = (1e-6,1)

        for i in range(len(self.theta0)): 
            bounds.append(theta_bound)
        weight_f = (np.array(self.weights0)).flatten()
        for j in range(len(weight_f)):
            bounds.append(weight_bound)

        while (self.likelihood < self.likelihood_threshold):
            if (self.optimizer=="CMA-ES"):
                # constraint = lambda n: self.constraint_func(n)
                # constraint_list = sum([constraint(i) for i in self.Nk], ())
                LB = []
                UB = []
                for i in range(len(bounds)):
                    LB.append(bounds[i][0])
                    UB.append(bounds[i][1])
                new_bounds = [LB,UB]
                xopts, es = cma.evolution_strategy.fmin_con(self.NLL, xk, 0.1, h=lambda xk:np.array(self.constraint_func(xk)), \
                    options={'bounds':new_bounds, 'verbose':-9},restarts=self.restarts)
                # ,'verbose':-9,'ftarget':1e-5'popsize':40,
                # es.logger.plot()
                # es.objective_function_complements[0].logger.plot()
                # plt.show()
                # print(es.stop())
                # lambda xk: [(self.constraint_func(xk)).sum()]
                if xopts is None:
                    optimal_hyperparameter = es.best
                else:
                    optimal_hyperparameter = xopts
            
            elif self.optimizer == "nelder-mead-c":
                LB = []
                UB = []
                for i in range(len(bounds)):
                    LB.append(bounds[i][0])
                    UB.append(bounds[i][1])
                res = cNM.constrNM(self.NLL,xk,LB,UB,full_output=True)
                optimal_hyperparameter = res['xopt']

            elif self.optimizer == "nelder-mead" or "SLSQP" or "COBYLA" or "TNC":
                cons = ({'type':'eq','fun':self.constraint_func})
                res = minimize(self.NLL,xk,method=self.optimizer,bounds=bounds,constraints=cons,options={'ftol':1e-20,'disp':False})
                optimal_hyperparameter = res.x
            xk = self.get_new_initial_points_w() #reset starting point
            self.likelihood = -float(self.NLL(optimal_hyperparameter)) # compute the maximum likelihood
        optimal_hyperparameter = optimal_hyperparameter.tolist()
        self.theta = self.theta0
        for i in range(len(self.theta0)):   
            self.theta[i] = optimal_hyperparameter.pop(0)


        weights_vector = optimal_hyperparameter
        self.weights_vector = np.copy(weights_vector)
        weights_vector = np.array_split(weights_vector,len(weights_vector))
        
        weights = []
        for p in range(self.Nk):
            weights.append([])
        for l in range(self.Nk):
            for m in range(len(self.kernels)):
                w = weights_vector.pop(0)
                weights[l].append(float(w))
        weights = np.array(weights)
        self.weights = weights
        # self.weights = weights / np.sum(weights,axis=1,keepdims=True)
        print(self.theta)
        print(self.weights)
        self.info = {'kernel weights':self.weights,
                        'Theta':self.theta,
                        'Likelihood':self.likelihood}

    def train(self):
        weights0 = (np.array(self.weights0)).flatten()
        weights0 = weights0.tolist()
        hyperparameter = self.theta0 + weights0
        self.get_theta(hyperparameter)


    def predict(self,testdata):
        if(self.Nk == 1):
            testdata = testdata[:,np.newaxis]
        self.testdata = self.x_scaler.transform(testdata) # scale the test points
        
        self.x_test = self.testdata
        test_size = self.x_test.shape[0]
        # Get pairwise componentwise L1-distances to the input training set
        dx = differences(self.x_test, Y=self.x.copy())
        # d = self.compute_componentwise_distance(dx)
        # Compute the cross correlation matrix   
        r_x = (self.compute_rr(dx,self.theta,self.weights_vector)).reshape(test_size,self.Ns) 
        f = np.ones(test_size)[:,np.newaxis]
        y_predict = np.dot(f,self.beta) + np.dot(r_x,self.gamma)
        self.y_predict = self.y_scaler.inverse_transform(y_predict)
        return self.y_predict   

    def computeRMSE(self,y_exact):
        m = len(self.y_predict) 
        sum = 0.0
        for i in range(m):
            sum += np.power((y_exact[i] - self.y_predict[i]),2)
        self.RMSE = np.sqrt(sum / m)
        return self.RMSE    

    def computeNRMSE(self,y_exact):
        m = len(self.y_predict) 
        sum = 0.0
        for i in range(m):
            try:
                sum += np.power((y_exact[i] - self.y_predict[i]),2)
            except:
                # self.y_predict = self.y_predict.reshape(m,)
                y_exact = np.asarray(y_exact)
                sum += np.power((y_exact[i] - self.y_predict[i]),2)
        self.RMSE = np.sqrt(sum / m)
        self.RMSE /= (np.max(y_exact)-np.min(y_exact))
        return self.RMSE
        
    def get_new_initial_points_w(self):
        new_start_point = []
        for i in range(self.Nk): # theta part
            new_start_point.append(np.random.uniform(1e-2,5))
        weight_f = (np.array(self.weights0)).flatten()
        for j in range(len(weight_f)):
            new_start_point.append(np.random.uniform(1e-6,0.25))
        return new_start_point


def cross_distances(X):
    """
    Computes the nonzero componentwise cross-distances between the vectors
    in X.

    Parameters
    ----------

    X: np.ndarray [n_obs, dim]
            - The input variables.

    Returns
    -------

    D: np.ndarray [n_obs * (n_obs - 1) / 2, dim]
            - The cross-distances between the vectors in X.

    ij: np.ndarray [n_obs * (n_obs - 1) / 2, 2]
            - The indices i and j of the vectors in X associated to the cross-
            distances in D.
    """

    n_samples, n_features = X.shape
    n_nonzero_cross_dist = n_samples * (n_samples - 1) // 2
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int)
    D = np.zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0

    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)
        D[ll_0:ll_1] = X[k] - X[(k + 1) : n_samples]

    return D, ij.astype(np.int)


def differences(X, Y):
    X, Y = check_pairwise_arrays(X, Y)
    D = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    return D.reshape((-1, X.shape[1]))