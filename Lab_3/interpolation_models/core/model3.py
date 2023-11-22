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


'''
Improvements:
- in-house preprocessing

'''
class Kriging:

    def __init__(self,x,y,kernel,theta0,optimizer="CMA-ES",optimizer_noise=0.1,eps=1.48e-08,restarts=1,preprocessing="normalize"):
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

        self.kernel = kernel
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
        kernel = self.kernel
        R_total = 1
        n = X.shape[0] # size of training sample
        m = Y.shape[0] # size of test sample

        for i in range(self.Nk):
            # make fluid to accomodate cross-correlation matrix later
            R = np.zeros((n,m)) #initialize correlation for each variable
            thetai = theta[i]
            for j in range(n):
                for k in range(m):
                    h = abs(X[j][i]-Y[k][i]) # compute the distance, h
                    if kernel == "exponential":
                        R[j,k] = (np.exp(-1.0 * np.abs(h)/thetai))
                    elif kernel == "matern3_2":
                        R[j,k] = (1 + (np.sqrt(3)*h)/thetai)*np.exp(((-np.sqrt(3))*h)/thetai) 
                    elif kernel == "gaussian":
                        R[j,k] = (np.exp(-0.5 * ((h/thetai) ** 2))) #on the suspicion that d is already squared
                    elif kernel == "matern5_2":
                        R[j,k] = (1 + (np.sqrt(5)*h)/thetai + (5/3*(h/thetai)**2))*np.exp(((-np.sqrt(5))*h)/thetai) 
                    elif kernel == "cubic":
                        h = abs(h/thetai)
                        if (h > 0 or h==0) and h < 0.5:
                            R[j,k] = 1 - 6*(h**2) + 6*(h**3)
                        elif h > 0.5 and (h < 1.0 or h==1.0):
                            R[j,k] = 2*(1-h)**3
                        elif h > 1.0:
                            R[j,k] = 0
                    else:
                        print("Unknown kernel")

            R_total *= R
        # R_total = R
        return R_total

    def NLL(self, theta):
        y = self.y
        n = len(y)
        R = self.compute_R(self.x,self.x,theta)
        self.R = R # so I can reuse this R at any point in the code

        self.F = np.ones(self.Ns)[:,np.newaxis]
        FT = (self.F).T
        if(NPD.isPD(R)==False): # if R is not positive definite / non-singular
            R = NPD.nearestPD(R)
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
        return float(nll)

    def get_theta(self,theta0):
        xk  = theta0
        bounds = []
        theta_bound = (0.0001,100000.0)

        for i in range(len(self.theta0)):
            bounds.append(theta_bound)  

        if (self.optimizer=="CMA-ES"):
            LB = []
            UB = []
            for i in range(len(bounds)):
                LB.append(bounds[i][0])
                UB.append(bounds[i][1])
            new_bounds = [LB,UB]
            xopts, es = cma.fmin2(self.NLL,xk,self.optimizer_noise,{'bounds':new_bounds,'verbose':-9},restarts=self.restarts)
            if xopts is None:
                theta = es.best
            else:
                theta = xopts
            self.theta = theta
        
        elif self.optimizer == "nelder-mead-c":
            LB = []
            UB = []
            for i in range(len(bounds)):
                LB.append(bounds[i][0])
                UB.append(bounds[i][1])
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
        # if(self.Nk == 1):
        #     testdata = testdata[:,np.newaxis]
        self.testdata = self.x_scaler.transform(testdata) # scale the test points
        self.x_test = self.testdata
        test_size = self.x_test.shape[0]

        NLL = self.NLL(self.theta) #an hack to get all parameters and make the obj func evaluation available
        # compute r(x)
        r_x = self.compute_R(self.x,self.x_test,self.theta)
        # self.R = self.compute_R(self.x,self.x,self.theta) #hack for ensemble method
        if not(NPD.isPD(self.R)): #sane check
            R = NPD.nearestPD(self.R)
        else:
            R = self.R
        R = np.linalg.cholesky(R) # lower cholesky factor
        f = np.ones(test_size)[:,np.newaxis]
        y_predict = np.dot(f,self.Beta) + np.dot((np.dot(r_x.T,self.Ri)), self.Y)
        variance = np.zeros(test_size)
        for i in range(test_size):
            xtest = self.x_test[i,:][np.newaxis,:]
            r_xi = self.compute_R(self.x,xtest,self.theta)
            variance[i] = self.sigma2*(1 - np.dot(np.dot(r_xi.T,self.Ri), r_xi) + \
            np.dot((1 - \
            np.dot(np.dot((self.F).T,self.Ri),r_xi))**2,np.linalg.inv(np.dot(np.dot((self.F).T,self.Ri),self.F))))
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
            try:
                sum += np.power((y_exact[i] - self.y_predict[i]),2)
            except:
                # self.y_predict = self.y_predict.reshape(m,)
                y_exact = np.asarray(y_exact)
                sum += np.power((y_exact[i] - self.y_predict[i]),2)
        self.RMSE = np.sqrt(sum / m)
        self.RMSE /= (np.max(y_exact)-np.min(y_exact))
        return self.RMSE
