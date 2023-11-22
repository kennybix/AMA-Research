import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy.optimize import minimize
from interpolation_models.core import nearestPD as NPD
from interpolation_models.core import constrNMPy as cNM
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import MinMaxScaler as MS

import cma

from scipy.special import kv,kn,gamma

'''
Improvements:
- free form Matern kernels
- in-house preprocessing

'''
class Kriging:

    def __init__(self,x,y,theta0,nu0,optimizer="CMA-ES",optimizer_noise=0.01,eps=1.48e-08,restarts=3,preprocessing="normalize"):
        if preprocessing == "normalize":
            self.x_scaler = MS()
            self.y_scaler = MS()
        elif preprocessing == "standardize":
            self.x_scaler = SS()
            self.y_scaler = SS()
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

        self.theta0 = theta0
        self.nu0 = nu0
        self.optimizer = optimizer
        self.optimizer_noise = optimizer_noise
        self.eps = eps
        self.restarts = restarts
        self.Ns = self.x.shape[0]
        self.Nk = self.x.shape[1]
        if self.optimizer == "CMA-ES" and self.Nk == 1: # hack to avoid certain errors
            self.optimizer = "nelder-mead-c"

    def compute_R(self,X,Y,theta,nu):
        # Notations:
        #  distance = the Ns x Ns matrix
        #  K = the Ns x Ns spatial correlation matrix following the Matern class
        #  nu = the variable of the free-form matern; 
        #  theta = the correlation range
        #  kv = Modified Bessel function of second kind of real order v(scipy.special function)
        #  gamma = Gamma function (scipy.special function)
        #  This form of the Matern is found in Rasmussen, Carl Edward and Williams,Christopher K. I. (2006) Gaussian Processes for Machine Learning.    
        n = X.shape[0] # size of training sample
        m = Y.shape[0] # size of test sample
        R_total = 1
        for i in range(self.Nk):
            # make fluid to accomodate cross-correlation matrix later
            R = np.zeros((n,m)) #initialize correlation for each variable]
            X_d = X[:,i]
            Y_d = Y[:,i]
            theta_d = theta[i]
            for j in range(n):
                for k in range(m):
                    h = abs(X_d[j]-Y_d[k]) # compute the 
                    h = h/theta_d
                    # h = h/theta
                    R[j,k] = (np.power(2,(1-nu)) / gamma(nu)) * np.power((np.sqrt(2*nu) * h),nu) * kn(nu,np.sqrt(2*nu)*h) # Matern free-form function
                    if(math.isnan(R[j,k])):
                        R[j,k] = 1.0
            R_total *=R #change for multidimension
        return R_total
        
    def NLL(self, hyperparameter):
        
        hyperparameter = np.array_split(hyperparameter,len(hyperparameter))
        nu = hyperparameter.pop(0)
        theta = np.concatenate(hyperparameter)
        y = self.y
        n = len(y)

        R = self.compute_R(self.x,self.x,theta,nu)
        np.fill_diagonal(R,1.0) #this should only be for the training samples

        if not(NPD.isPD(R)): #sane check
            R = NPD.nearestPD(R)
            
        self.R = R
        self.Ns = self.x.shape[0]#hack for EGO
        self.F = np.ones(self.Ns)[:,np.newaxis]
        FT = (self.F).T
        Ri = np.linalg.inv(R)
        self.Ri = Ri
        self.Beta = np.dot(np.linalg.inv(np.dot(FT,np.dot(Ri,self.F))),np.dot(FT,np.dot(Ri,y)))
        self.Y = (y - np.dot(self.F,self.Beta))
        self.sigma2 = 1.0/self.Ns * np.dot(self.Y.T,(np.dot(Ri,self.Y)))
        try:    
            nll = 1.0/2.0 * ((self.Ns * np.log(self.sigma2)) + np.log(np.linalg.det(R)))
            if (nll == -np.inf or math.isnan(nll)):
                nll = np.inf
        except np.linalg.LinAlgError: 
            print("Error in Linear Algebraic operation")
            nll = np.inf
        self.likelihood = float(nll)
        return float(nll)

    def get_theta(self, theta0, nu0):
        xk  =  nu0 + theta0
        bounds = []
        nu_bound = (0.5,10)
        theta_bound = (1e-5,1e6)

        bounds.append(nu_bound)

        for i in range(len(self.theta0)):
            bounds.append(theta_bound)  

        if (self.optimizer=="CMA-ES"):
            xopts, es = cma.fmin2(self.NLL,xk,self.optimizer_noise,{'bounds':[0.00001,1.0],'verbose':-9},restarts=self.restarts)
            if xopts is None:
                hyperparameter = es.best
            else:
                hyperparameter = xopts
      
        elif self.optimizer == "nelder-mead-c":
            LB = [0.01] +  [0.00001]*len(self.theta0)
            UB = [10] +  [1.0]*len(self.theta0)
            res = cNM.constrNM(self.NLL,xk,LB,UB,full_output=True)
            hyperparameter = res['xopt']

        elif self.optimizer == "nelder-mead" or "SLSQP" or "COBYLA" or "TNC":
                res1 = minimize(self.NLL,xk,method=self.optimizer,bounds=bounds,options={'ftol':1e-20,'disp':False})
                hyperparameter = res1.x

        hyperparameter = np.array_split(hyperparameter,len(hyperparameter))
        self.nu = hyperparameter.pop(0)
        self.theta = np.concatenate(hyperparameter)


    def train(self):
        self.get_theta(self.theta0,self.nu0)


    def predict(self,testdata):
        if(self.Nk == 1):
            testdata = testdata[:,np.newaxis]
        self.testdata = self.x_scaler.transform(testdata) # scale the test points
        self.x_test = np.array(self.testdata)
        test_size = self.x_test.shape[0]
        # compute r(x)
        r_x = self.compute_R(self.x,self.x_test,self.theta,self.nu)
        # print(r_x)
        R = self.R
        # R = np.linalg.cholesky(R) # lower cholesky factor
        f = np.ones(test_size)[:,np.newaxis]
        y_predict = np.dot(f,self.Beta) + np.dot((np.dot(r_x.T,self.Ri)), self.Y)
        variance = np.zeros(test_size)
        # for i in range(test_size):
        #     r_xi = self.compute_R(self.x,self.x_test[i,:],self.theta,self.nu)
        #     variance[i] = self.sigma2*(1 - np.dot(np.dot(r_xi.T,self.Ri), r_xi) + \
        #     np.dot((1 - \
        #     np.dot(np.dot((self.F).T,self.Ri),r_xi))**2,np.linalg.inv(np.dot(np.dot((self.F).T,self.Ri),self.F))))
        self.y_predict = self.y_scaler.inverse_transform(y_predict)
        self.variance = variance
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
            sum += np.power((y_exact[i] - self.y_predict[i]),2)
        self.RMSE = np.sqrt(sum / m)
        self.RMSE /= (np.max(y_exact)-np.min(y_exact))
        return self.RMSE
