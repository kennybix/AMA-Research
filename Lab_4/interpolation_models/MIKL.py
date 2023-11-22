import numpy as np
import math
import scipy
from scipy.optimize import minimize
from interpolation_models.core import nearestPD as NPD
from interpolation_models.core import constrNMPy as cNM
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import MinMaxScaler as MS
import cma
from sklearn.metrics.pairwise import check_pairwise_arrays
from scipy import linalg


'''
Improvements:
- in-house preprocessing

'''
class Kriging:

    def __init__(self,x,y,kernels,theta0="",weight0="",weight_optimizer="SLSQP",optimizer="nelder-mead-c",optimizer_noise=1.0,eps=1.48e-08,restarts=1,preprocessing="normalize"):


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
        # self.optimizer = optimizer
        self.optimizer = 'nelder-mead-c' #hardcode #to be changed. Take note!!!
        self.weight_optimizer = 'SLSQP'
        # self.weight_optimizer =  self.optimizer #use same optimizers for both processes
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


    def compute_R(self,D,theta):
        R_total = np.ones((D.shape[0],1))
        for i in range(self.Nk):
            index = self.kernels_indexes[i]
            kernel = self.kernels[index] # select best kernel for each dimension
            thetai = theta[i]
            d_comp = (D[:,i]).reshape(D.shape[0],1) #componentwise distance
            d_comp = np.abs(d_comp)
            # d_comp = np.abs(d_comp)
            # make fluid to accomodate cross-correlation matrix later
            R = np.zeros((D.shape[0],1)) #initialize correlation for each variable
            if kernel == "exponential":
                R = (np.exp(-1.0 * np.abs(d_comp)/thetai))
            elif kernel == "matern3_2":
                R = (1 + (np.sqrt(3)*d_comp)/thetai)*np.exp(((-np.sqrt(3))*d_comp)/thetai) 
            elif kernel == "gaussian":
                R = (np.exp(-0.5 * ((d_comp/thetai) ** 2))) #on the suspicion that d is already squared
            elif kernel == "matern5_2":
                R = (1 + (np.sqrt(5)*d_comp)/thetai + (5/3*(d_comp/thetai)**2))*np.exp(((-np.sqrt(5))*d_comp)/thetai)
            else:
                        print("Unknown kernel")
            R_total *= R
        return R_total

    def compute_R_w(self,D,theta,weights):
        kernels = self.kernels
        #weights input must be flattened
        weights = np.array_split(weights,len(weights))
        R_total = np.ones((D.shape[0],1))
        for i in range(self.Nk):
            thetai = theta[i]
            d_comp = (D[:,i]).reshape(D.shape[0],1) #componentwise distance
            d_comp = np.abs(d_comp)
            # make fluid to accomodate cross-correlation matrix later
            R_w = np.zeros((D.shape[0],1))
            for w in range(len(kernels)):#
                kernel = kernels[w] #select the appropriate kernel
                weight = weights.pop(0)
                R = np.zeros((D.shape[0],1)) #initialize correlation for each variable           
                if kernel == "exponential":
                    R = (np.exp(-1.0 * np.abs(d_comp)/thetai))
                elif kernel == "matern3_2":
                    R = (1 + (np.sqrt(3)*d_comp)/thetai)*np.exp(((-np.sqrt(3))*d_comp)/thetai) 
                elif kernel == "gaussian":
                    R = (np.exp(-0.5 * ((d_comp/thetai) ** 2))) #on the suspicion that d is already squared
                elif kernel == "matern5_2":
                    R = (1 + (np.sqrt(5)*d_comp)/thetai + (5/3*(d_comp/thetai)**2))*np.exp(((-np.sqrt(5))*d_comp)/thetai)
                else:
                    print("Unknown kernel")
                R_w += weight * R
            R_total *= R_w
        return R_total


    def NLL_w(self, hyperparameter):
        theta = np.zeros(len(self.theta0))
        hyperparameter = np.array_split(hyperparameter,len(hyperparameter))
        for i in range(len(self.theta0)):
            theta[i] = hyperparameter.pop(0)
        weights = np.concatenate(hyperparameter)

        nugget = 2.22e-11 # a very small value

        # Calculate matrix of distances D between samples
        D, self.ij = cross_distances(self.x)
        # compute the correlation matrix
        r = self.compute_R_w(D,theta,weights)
        R = np.eye(self.Ns) * (1.0 + nugget)
        R[self.ij[:, 0], self.ij[:, 1]] = r[:, 0]
        R[self.ij[:, 1], self.ij[:, 0]] = r[:, 0]

        y = self.y
        n = len(y)

        F = np.ones(self.Ns)[:,np.newaxis]
        FT = (F).T
        Ri = np.linalg.inv(R) 
        Beta = np.dot(np.linalg.inv(np.dot(FT,np.dot(Ri,F))),np.dot(FT,np.dot(Ri,y)))
        Y = (y - np.dot(F,Beta))
        sigma2 = 1.0/self.Ns * np.dot(Y.T,(np.dot(Ri,Y)))

        try:
            nll = 1.0/2.0 * ((self.Ns * np.log(sigma2)) + np.log(np.linalg.det(R)))

            if (nll == -np.inf or math.isnan(nll)):
                nll = np.inf
        except np.linalg.LinAlgError: 
            print("Error in Linear Algebraic operation")
            nll = np.inf
        return float(nll)

    def NLL(self, theta):
        nugget = 2.22e-11 # a very small value

        # Calculate matrix of distances D between samples
        D, self.ij = cross_distances(self.x)
        # compute the correlation matrix
        r = self.compute_R(D,theta)
        R = np.eye(self.Ns) * (1.0 + nugget)
        R[self.ij[:, 0], self.ij[:, 1]] = r[:, 0]
        R[self.ij[:, 1], self.ij[:, 0]] = r[:, 0]

        y = self.y
        n = len(y)
        self.R = R # so I can reuse this R at any point in the code
        self.F = np.ones(self.Ns)[:,np.newaxis]
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


    # def NLL(self, theta):
    #     nugget = 2.22e-11 # a very small value

    #     # Calculate matrix of distances D between samples
    #     D, self.ij = cross_distances(self.x)
    #     # compute the correlation matrix
    #     r = self.compute_R(D,theta)
    #     R = np.eye(self.Ns) * (1.0 + nugget)
    #     R[self.ij[:, 0], self.ij[:, 1]] = r[:, 0]
    #     R[self.ij[:, 1], self.ij[:, 0]] = r[:, 0]

    #     y = self.y
    #     n = len(y)
    #     self.R = R # so I can reuse this R at any point in the code

    #     self.F = np.ones(self.Ns)[:,np.newaxis]
    #     FT = (self.F).T
    #     if (NPD.isPD(R) == False): 
    #         R = NPD.nearestPD(R)
    #     Ri = np.linalg.inv(R)
    #     self.Ri = Ri
    #     self.Beta = np.dot(np.linalg.inv(np.dot(FT,np.dot(Ri,self.F))),np.dot(FT,np.dot(Ri,y)))
    #     self.Y = (y - np.dot(self.F,self.Beta))
    #     self.sigma2 = 1.0/self.Ns * np.dot(self.Y.T,(np.dot(Ri,self.Y)))

    #     try:
    #         nll = 1.0/2.0 * ((self.Ns * np.log(self.sigma2)) + np.log(np.linalg.det(R)))

    #         if (nll == -np.inf or math.isnan(nll)):
    #             nll = np.inf
    #     except np.linalg.LinAlgError: 
    #         print("Error in Linear Algebraic operation")
    #         nll = np.inf
    #     return float(nll)

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

    def get_theta_w(self,hyperparameter):
        xk = hyperparameter
        bounds = []
        theta_bound = (1e-4,1e3)
        weight_bound = (1e-6,1)

        for i in range(len(self.theta0)): 
            bounds.append(theta_bound)
        weight_f = (np.array(self.weights0)).flatten()
        for j in range(len(weight_f)):
            bounds.append(weight_bound)

        
        while (self.likelihood_w < self.likelihood_threshold):
            if (self.weight_optimizer=="CMA-ES"):
                LB = []
                UB = []
                for i in range(len(bounds)):
                    LB.append(bounds[i][0])
                    UB.append(bounds[i][1])
                new_bounds = [LB,UB]
                xopts, es = cma.evolution_strategy.fmin_con(self.NLL_w, xk, 1.0, h=lambda xk:np.array(self.constraint_func(xk)), \
                    options={'bounds':new_bounds, 'verbose':-9,'CMA_stds':xk},restarts=self.restarts)
                if xopts is None:
                    optimal_hyperparameter = es.best
                else:
                    optimal_hyperparameter = xopts
            
            elif self.weight_optimizer == "nelder-mead-c":
                LB = []
                UB = []
                for i in range(len(bounds)):
                    LB.append(bounds[i][0])
                    UB.append(bounds[i][1])
                res = cNM.constrNM(self.NLL_w,xk,LB,UB,full_output=True)
                optimal_hyperparameter = res['xopt']

            elif self.weight_optimizer == "nelder-mead" or "SLSQP" or "COBYLA" or "TNC":
                cons = ({'type':'eq','fun':self.constraint_func})
                res = minimize(self.NLL_w,xk,method=self.weight_optimizer,bounds=bounds,constraints=cons,options={'disp':False})
                optimal_hyperparameter = res.x
            xk = self.get_new_initial_points_w() #reset starting point
            self.likelihood_w = -float(self.NLL_w(optimal_hyperparameter))
        optimal_hyperparameter = np.array_split(optimal_hyperparameter,len(optimal_hyperparameter))

        # self.theta_w = self.theta0
        # for i in range(len(self.theta0)):   
        #     self.theta_w[i] = optimal_hyperparameter.pop(0)

        self.theta_w = []
        for i in range(len(self.theta0)):   
            self.theta_w.append(optimal_hyperparameter.pop(0)[0])

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
        self.kernels_indexes = []
        for k in range(self.Nk):
            x = np.argmax(self.weights[k,:])
            self.kernels_indexes.append(x)
        # self.weights = weights / np.sum(weights,axis=1,keepdims=True)
        # print(self.theta)
        # print(self.weights)
        return self

    def get_weights(self):
        weights0 = (np.array(self.weights0)).flatten()
        weights0 = weights0.tolist()
        hyperparameter = self.theta0 + weights0
        self.get_theta_w(hyperparameter) #change to get weights 
        return self

    def get_theta(self,theta):
        xk = theta
        bounds = []
        theta_bound = (1e-4,1e3)

        for i in range(len(self.theta0)): 
            bounds.append(theta_bound)

        if (self.optimizer=="CMA-ES"):
            LB = []
            UB = []
            for i in range(len(bounds)):
                LB.append(bounds[i][0])
                UB.append(bounds[i][1])
            new_bounds = [LB,UB]
            xopts, es = cma.fmin2(self.NLL,xk,0.1,{'bounds':new_bounds,'verbose':-9},restarts=self.restarts)
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
            res = minimize(self.NLL,xk,method=self.optimizer,bounds=bounds,options={'ftol':1e-20,'disp':False})
            optimal_hyperparameter = res.x
        self.theta = optimal_hyperparameter
        self.likelihood = -float(self.NLL(self.theta)) # make the likelihood value available
        return self

    def train(self):
        self.get_weights()
        hyperparameter = self.get_new_initial_points()
        # hyperparameter = self.theta_w
        self.get_theta(hyperparameter)
        # get the information here
        self.mixed_kernels = get_kernel_names(self.kernels,self.kernels_indexes)
        self.info = {'weight':self.weights,
                        'Theta_with_weight':self.theta_w,
                        'Likelihood_with_theta':self.likelihood_w,
                        'chosen_kernels': self.mixed_kernels,
                        'Theta':self.theta,
                        'Likelihood':self.likelihood}
        return self

    def predict(self,testdata):
        if(self.Nk == 1):
            testdata = testdata[:,np.newaxis]
        self.x_test = self.x_scaler.transform(testdata) # scale the test points
        del testdata
        test_size = self.x_test.shape[0]
        # Get pairwise componentwise L1-distances to the input training set
        dx = differences(self.x_test, Y=self.x.copy())
        # d = self.compute_componentwise_distance(dx)
        # Compute the cross correlation matrix
        r_x = (self.compute_R(dx,self.theta)).reshape(test_size,self.Ns) 
        f = np.ones(test_size)[:,np.newaxis]
        y_predict = np.dot(f,self.beta) + np.dot(r_x,self.gamma)
        self.y_predict = self.y_scaler.inverse_transform(y_predict)
        return self.y_predict

    def predict_variance(self,testdata):
        if(self.Nk == 1):
            testdata = testdata[:,np.newaxis]
        self.x_test = self.x_scaler.transform(testdata) # scale the test points
        del testdata
        test_size = self.x_test.shape[0]
        dx = differences(self.x_test, Y=self.x.copy())
        variance = np.zeros(test_size) # variance calculation should not be part
        r_x = (self.compute_R(dx,self.theta)).reshape(test_size,self.Ns)
        rt = linalg.solve_triangular(self.C, r_x.T, lower=True)

        u = linalg.solve_triangular((self.G).T, np.dot((self.Ft).T, rt))

        A = self.sigma2
        B = 1.0 - (rt ** 2.0).sum(axis=0) + (u ** 2.0).sum(axis=0)
        variance = np.einsum("i,j -> ji", A, B)

        # Mean Squared Error might be slightly negative depending on
        # machine precision: force to zero!
        variance[variance < 0.0] = 0.0      
        self.variance = variance
        return self.variance

    # def predict(self,testdata):
    #     if(self.Nk == 1):
    #         testdata = testdata[:,np.newaxis]
    #     self.testdata = self.x_scaler.transform(testdata) # scale the test points
    #     self.x_test = self.testdata
    #     test_size = self.x_test.shape[0]
    #     # compute r(x)
    #     dx = differences(self.x_test, Y=self.x.copy())
    #     r_x = (self.compute_R(dx,self.theta)).reshape(test_size,self.Ns) 
    #     if not(NPD.isPD(self.R)): #sane check
    #         R = NPD.nearestPD(self.R)
    #     else:
    #         R = self.R  
    #     R = np.linalg.cholesky(R) # lower cholesky factor
    #     f = np.ones(test_size)[:,np.newaxis]
    #     y_predict = np.dot(f,self.Beta) + np.dot((np.dot(r_x,self.Ri)), self.Y)
    #     variance = np.zeros(test_size)
    #     # for i in range(test_size):
    #     #     xtest = self.x_test[i,:][np.newaxis,:]
    #     #     r_xi = self.compute_R(self.x,xtest,self.theta)
    #     #     variance[i] = self.sigma2*(1 - np.dot(np.dot(r_xi.T,self.Ri), r_xi) + \
    #     #     np.dot((1 - \
    #     #     np.dot(np.dot((self.F).T,self.Ri),r_xi))**2,np.linalg.inv(np.dot(np.dot((self.F).T,self.Ri),self.F))))
    #     self.y_predict = self.y_scaler.inverse_transform(y_predict)
    #     self.variance = variance
    #     return self.y_predict

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


    def get_new_initial_points(self):
        new_start_point = []
        for i in range(self.Nk): # theta part
            new_start_point.append(np.random.uniform(1e-2,5))
        return new_start_point

def get_kernel_names(kernel_list,kernel_index):
    mixed_kernel = []
    for i in range(len(kernel_index)):
        mixed_kernel.append(kernel_list[kernel_index[i]])
    return mixed_kernel

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


