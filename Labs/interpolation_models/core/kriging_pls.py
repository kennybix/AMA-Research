import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy.optimize import minimize
from interpolation_models.core import nearestPD as NPD
from interpolation_models.core import constrNMPy as cNM
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import MinMaxScaler as MS
from scipy import linalg
import cma
# import gurobipy as gp 
# import pyopt as po


from sklearn.cross_decomposition import PLSRegression  
from sklearn.metrics import mean_squared_error, r2_score  
from sklearn.model_selection import cross_val_predict    
from sys import *

from sklearn.metrics.pairwise import check_pairwise_arrays
from scipy.special import kv,kn,gamma


'''
Improvements:
- in-house preprocessing

'''
def optimise_pls_cv(X, y, n_comp, plot_components=False): #code to optimize the pls
    '''Run PLS including a variable number of components,
    up to n_comp,
    and calculate MSE
    '''   
    #now we want to use LOOCV 
    cv_n = X.shape[0] #number of sample points hack
    mse = []      
    component = np.arange(1, n_comp)        
    for i in component:         
        pls = PLSRegression(n_components=i)            
        # Cross-validation          
        y_cv = cross_val_predict(pls, X, y, cv=cv_n)
        mse.append(mean_squared_error(y, y_cv))            
        comp = 100*(i+1)/n_comp #marks the progress 
        # Trick to update status on the same line 
        stdout.write("\r%d%% completed" % comp)          
        stdout.flush()      
        stdout.write("\n")# Calculate and print the position of minimum in MSE      
    msemin = np.argmin(mse)  
    opt_comp = msemin+1    
    print("Suggested number of components: ",opt_comp)      
    stdout.write("\n") 
    return opt_comp


class Kriging:

    def __init__(self,x,y,kernel,theta0="",optimizer="nelder-mead-c",optimizer_noise=1.0,eps=1.48e-08,restarts=1,preprocessing="standardize",pls_n_comp="", pls_cv=5):

        # self.pls_n_comp = optimise_pls_cv(x.copy(),y.copy(),x.shape[1])
        if pls_n_comp == "":
            self.pls_n_comp = 1
        else:
            self.pls_n_comp = pls_n_comp
        
        self.pls = PLSRegression(n_components=self.pls_n_comp)
        self.coeff_pls = self.pls.fit(x.copy(),y.copy()).x_rotations_

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


        self.kernel = kernel
        self.optimizer = optimizer
        self.optimizer_noise = optimizer_noise
        self.eps = eps
        self.restarts = restarts
        self.Ns = self.x.shape[0]
        self.Nk = self.x.shape[1]

        self.theta0 = [0.01]*self.pls_n_comp
        
        if self.optimizer == "CMA-ES" and self.Nk == 1: # hack to avoid certain errors
            self.optimizer = "nelder-mead-c"
    
    
    def compute_K(self,D,kernel,theta):  
        K = np.ones((D.shape[0],1))
        for i in range(self.pls_n_comp):
            d_comp = (D[:,i]).reshape(D.shape[0],1)
            if kernel == "exponential":
                K_p = np.exp(-1.0 * (d_comp))
            elif kernel == "matern3_2":
                K_p = (1 + (np.sqrt(3)*d_comp)) * np.exp((-np.sqrt(3))*d_comp)
            elif kernel == "gaussian":
                K_p = np.exp(-0.5 * (d_comp))
            elif kernel == "matern5_2":
                K_p = (1 + np.sqrt(5)*np.abs(d_comp) + (5/3*(d_comp**2))) * np.exp((-np.sqrt(5))*np.abs(d_comp))
            
            # elif kernel == "matern":
            #     K_p = np.multiply(float(np.power(2,(1-self.nu)) / gamma(self.nu)) , np.power((np.sqrt(2*self.nu) * d_comp),self.nu))
            #     K_p = np.multiply(K_p,kv(self.nu,np.sqrt(2*self.nu)*d_comp) )                   
            # else:
            #     print("Unknown kernel")
            K *= K_p
            del d_comp
        return K

    def compute_componentwise_distance(self,D,theta):
        kernel = self.kernel
        coeff_pls = self.coeff_pls
        D_corr = np.zeros((D.shape[0], coeff_pls.shape[1]))
        if kernel == "gaussian":
            D_corr = np.dot(D**2,coeff_pls**2)
        elif kernel == "matern5_2":
            D_corr = np.dot(D,coeff_pls)
        else:
            # kernel = "exponential" or "matern3_2" or "free form matern"
            D_corr = np.dot(np.abs(D),coeff_pls)
        D_corr = np.einsum("j,ij->ij", (theta).T, D_corr)
        return D_corr
    

    def compute_rr(self,D,theta):
        self.D = self.compute_componentwise_distance(D,theta) #I don't need to compute this everytime
        # during training
        r = self.compute_K(self.D,self.kernel,theta)                                                                                                                                                             
        return r

    def NLL(self, theta):
        nugget = 2.22e-14 # a very small value

        if self.kernel == "matern": #making room for the free matern kernel
            theta = theta.tolist() #temporarily convert to list
            self.nu = theta.pop(0) #remove the first element
            theta = np.array(theta) #convert theta back to an array
        else:
            self.nu = 0

        # Calculate matrix of distances D between samples
        D, self.ij = cross_distances(self.x)
        # compute the correlation matrix
        r = self.compute_rr(D,theta)
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
            # C = np.linalg.cholesky(self.R)
        except (linalg.LinAlgError, ValueError) as e:
            print("exception : ", e)
            self.R = NPD.nearestPD(self.R)
            C = np.linalg.cholesky(self.R)
            # C = linalg.cholesky(self.R, lower=True)
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

    def get_theta(self,theta0):
        xk  = theta0
        bounds = []
        theta_bound = (1e-4,1e3)

        for i in range(self.pls_n_comp):
            bounds.append(theta_bound)  

        if (self.optimizer=="CMA-ES"):
            LB = []
            UB = []
            for i in range(len(bounds)):
                LB.append(bounds[i][0])
                UB.append(bounds[i][1])
            new_bounds = [LB,UB]
            xopts, es = cma.fmin2(self.NLL,xk,1.0,{'bounds':new_bounds,'verbose':-9,'CMA_stds':xk},restarts=self.restarts)
            if xopts is None:
                theta = es.best
            else:
                theta = xopts
        
        elif self.optimizer == "nelder-mead-c":
            LB = []
            UB = []
            for i in range(len(bounds)):
                LB.append(bounds[i][0])
                UB.append(bounds[i][1])
            res = cNM.constrNM(self.NLL,xk,LB,UB,full_output=True)
            theta = res['xopt']

        elif self.optimizer == "nelder-mead" or "SLSQP" or "COBYLA" or "TNC":
                res1 = minimize(self.NLL,xk,method=self.optimizer,options={'disp':False})
                theta = res1.x        
        self.theta = theta
        self.likelihood = -self.NLL(theta) #could incur extra computational time
        self.info = {'Theta':self.theta,
                        'Likelihood':self.likelihood}

    def train(self):
        self.get_theta(self.theta0)


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
        r_x = (self.compute_rr(dx,self.theta)).reshape(test_size,self.Ns)
        # r_x[0][0] = 1.0 # need to hack nan values
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
        r_x = (self.compute_rr(dx,self.theta)).reshape(test_size,self.Ns)
        # r_x[0][0] = 1.0 # need to hack nan values
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