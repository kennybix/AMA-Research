import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy.optimize import minimize
from interpolation_models.core import nearestPD as NPD
from interpolation_models.core import constrNMPy as cNM
import cma

from scipy.special import kv,kn,gamma


class Kriging:

    def __init__(self,x,y,theta0,optimizer="CMA-ES",optimizer_noise=1.0,eps=1.48e-08,restarts=1):
        self.x = x
        self.y = y
        self.theta0 = theta0
        self.optimizer = optimizer
        self.optimizer_noise = optimizer_noise
        self.eps = eps
        self.restarts = restarts
        self.Ns = self.x.shape[0]
        self.Nk = self.x.shape[1]
        if self.optimizer == "CMA-ES" and self.Nk == 1: # hack to avoid certain errors
            self.optimizer = "nelder-mead-c"

    def compute_R(self,X,Y,theta):
        R_total = 1
        n = X.shape[0] # size of training sample
        m = Y.shape[0] # size of test sample

        for i in range(self.Nk):
            # make fluid to accomodate cross-correlation matrix later
            R = np.zeros((n,m)) #initialize correlation for each variable
            for j in range(n):
                for k in range(m):
                    h = abs(X[j]-Y[k]) # compute the 
                    h = abs(h/theta)
                    if (h > 0 or h==0) and h < 0.5:
                        R[j,k] = 1 - 6*(h**2) + 6*(h**3)
                    elif h > 0.5 and (h < 1.0 or h==1.0):
                        R[j,k] = 2*(1-h)**3
                    elif h > 1.0:
                        R[j,k] = 0
            R_total *= R
        # R_total = R
        return R_total

    def compute_Beta(self,R,y):
        self.F = np.ones(self.Ns)[:,np.newaxis]
        if (NPD.isPD(R)):
            R = R
        else:
            R = NPD.nearestPD(R)
        Ri = np.linalg.inv(R)
        self.Ri = Ri
        FT = (self.F).T
        temp = (np.dot(FT,np.dot(Ri,self.F)))
        temp2 = (np.dot(FT,np.dot(Ri,y)))
        invtemp = np.linalg.inv(temp)
        Beta = np.dot(invtemp,temp2)
        return Beta

    def NLL(self, theta):
        y = self.y
        n = len(y)
        R = self.compute_R(self.x,self.x,theta)
        self.R = R # so I can reuse this R at any point in the code
        try:
            self.Beta = self.compute_Beta(R,y)
            self.Y = (y - np.dot(self.F,self.Beta))
            c = np.linalg.inv(np.linalg.cholesky(R))      
            Ri = np.dot(c.T,c)
            self.sigma2 = 1.0/self.Ns * np.dot(self.Y.T,(np.dot(Ri,self.Y)))
            nll = 1.0/2.0 * ((self.Ns * np.log(self.sigma2)) + np.log(np.linalg.det(R)))

            if (nll == -np.inf or math.isnan(nll)):
                nll = np.inf
        except np.linalg.LinAlgError: 
            print("Error in Linear Algebraic operation")
            nll = np.inf
        return float(nll)

    def get_theta(self,theta0):
        xk  = theta0
        bounds = []
        theta_bound = (0.0001,1.0)

        for i in range(len(self.theta0)):
            bounds.append(theta_bound)  

        if (self.optimizer=="CMA-ES"):
            xopts, es = cma.fmin2(self.NLL,xk,self.optimizer_noise,{'bounds':[0.00001,1.0],'verbose':-9},restarts=self.restarts)
            if xopts is None:
                theta = es.best
            else:
                theta = xopts
            self.theta = theta
        
        elif self.optimizer == "nelder-mead-c":
            LB = [0.000001]*len(self.theta0)
            UB =  [1.0]*len(self.theta0)
            res = cNM.constrNM(self.NLL,xk,LB,UB,full_output=True)
            theta = res['xopt']

        elif self.optimizer == "nelder-mead" or "SLSQP" or "COBYLA" or "TNC":
                res1 = minimize(self.NLL,xk,method=self.optimizer,bounds=bounds,options={'ftol':1e-20,'disp':False})
                theta = res1.x
        self.theta = theta
        self.likelihood = self.NLL(theta)

    def train(self):
        self.get_theta(self.theta0)


    def predict(self,testdata):
        self.testdata = testdata
        self.x_test = np.array(self.testdata)
        test_size = self.x_test.shape[0]
        # compute r(x)
        r_x = self.compute_R(self.x,self.x_test,self.theta)
        if not(NPD.isPD(self.R)): #sane check
            R = NPD.nearestPD(self.R)
        else:
            R = self.R
        R = np.linalg.cholesky(R) # lower cholesky factor
        f = np.ones(test_size)[:,np.newaxis]
        self.y_predict = np.dot(f,self.Beta) + np.dot((np.dot(r_x.T,self.Ri)), self.Y)
        self.variance = np.zeros(test_size)
        for i in range(test_size):
            r_xi = self.compute_R(self.x,self.x_test[i,:],self.theta)
            self.variance[i] = self.sigma2*(1 - np.dot(np.dot(r_xi.T,self.Ri), r_xi) + \
            np.dot((1 - \
            np.dot(np.dot((self.F).T,self.Ri),r_xi))**2,np.linalg.inv(np.dot(np.dot((self.F).T,self.Ri),self.F))))
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
