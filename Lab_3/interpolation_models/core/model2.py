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

from sklearn.cross_decomposition import PLSRegression  
from sampling_algorithm import adaptive_sampling as adsp
'''
The model trains with a pseudo data and adj
Improvements:
- in-house preprocessing

'''
def get_weight(coeff):
    coeff_scale = MS()
    norm_coeff = coeff_scale.fit_transform(coeff)
    weight_coeff = coeff # initialization
    sum_coeff = norm_coeff.sum(0)
    for i in range(len(coeff)):
        weight_coeff[i] = norm_coeff[i] / sum_coeff
    return weight_coeff

def adjust_theta(pseudo_theta,pseudo_pls,whole_pls):
    # pseudo_weight = get_weight(pseudo_pls)
    # whole_weight = get_weight(whole_pls)
    pseudo_weight = pseudo_pls
    whole_weight = whole_pls

    whole_theta = pseudo_theta #initialization
    for i in range(len(pseudo_theta)):
        whole_theta[i] = (whole_weight[i] * pseudo_theta[i])/pseudo_weight[i]
    return whole_theta


class Kriging:

    def __init__(self,x,y,kernel,theta0="",optimizer="CMA-ES",optimizer_noise=0.1,eps=1.48e-08,restarts=1,preprocessing="normalize",pseudo_size=30):

        self.whole_x = x
        self.whole_y = y
        # get whole data pls coefficient
        pls = PLSRegression(n_components=1)
        self.whole_coeff_pls = pls.fit(x.copy(),y.copy()).x_rotations_     

        # reduce data by some method and get the coefficients
        initial_size = int(pseudo_size*0.75)
        plan = adsp.Adaptive_Sampling(x,y,initial_ns=initial_size, final_ns=pseudo_size)
        x_sample,y_sample = plan.adapt_samples()


        self.pseudo_coeff_pls = pls.fit(x_sample.copy(),y_sample.copy()).x_rotations_ 

        self.preprocessing = preprocessing
        if self.preprocessing == "normalize":
            self.x_scaler = MS()
            self.y_scaler = MS()
        elif self.preprocessing == "standardize":
            self.x_scaler = SS()
            self.y_scaler = SS()

        try:
            self.x_scaler.fit(x_sample)

        except:
            x_sample = x_sample[:,np.newaxis] #hack for 1D
            self.x_scaler.fit(x_sample)

        try:
            self.y_scaler.fit(y_sample)

        except:
            y_sample = y_sample[:,np.newaxis] #hack for 1D
            self.y_scaler.fit(y_sample)

        self.x = self.x_scaler.transform(x_sample)
        self.y = self.y_scaler.transform(y_sample)

        self.kernel = kernel
        self.Ns = self.x.shape[0]
        self.Nk = self.x.shape[1]
        if theta0 == "": #choose initial theta randomly
            theta0 = [] #empty list
            for i in range(self.Nk):
                theta0.append(np.random.uniform(1e-2,5))
        else: 
            theta0 = theta0

        self.theta0 = theta0
        self.optimizer = optimizer
        self.optimizer_noise = optimizer_noise
        self.eps = eps
        self.restarts = restarts

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

        self.F = np.ones(len(y))[:,np.newaxis]
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
        theta_bound = (1e-4,1e5)

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

        pseudo_theta = theta 
        self.theta = adjust_theta(pseudo_theta,self.pseudo_coeff_pls,self.whole_coeff_pls) #correct theta
        # self.likelihood = self.NLL(self.theta)

    def train(self):
        self.get_theta(self.theta0)


    def predict(self,testdata):

        #adjust scales again
        if self.preprocessing == "normalize":
            self.x_scaler = MS()
            self.y_scaler = MS()
        elif self.preprocessing == "standardize":
            self.x_scaler = SS()
            self.y_scaler = SS()

        try:
            self.x_scaler.fit(self.whole_x)

        except:
            self.whole_x = self.whole_x[:,np.newaxis] #hack for 1D
            self.x_scaler.fit(self.whole_x)

        try:
            self.y_scaler.fit(self.whole_y)

        except:
            self.whole_y = self.whole_y[:,np.newaxis] #hack for 1D
            self.y_scaler.fit(self.whole_y)

        self.x = self.x_scaler.transform(self.whole_x)
        self.y = self.y_scaler.transform(self.whole_y)




        self.testdata = self.x_scaler.transform(testdata) # scale the test points
        self.x_test = self.testdata
        test_size = self.x_test.shape[0]

        # NLL = self.NLL(self.theta) #an hack to get all parameters and make the obj func evaluation available
        R = self.compute_R(self.x,self.x,self.theta)
        # compute r(x)
        r_x = self.compute_R(self.x,self.x_test,self.theta)
        # self.R = self.compute_R(self.x,self.x,self.theta) #hack for ensemble method
        if not(NPD.isPD(R)): #sane check
            R = NPD.nearestPD(R)
        else:
            R = R
        # R = np.linalg.cholesky(R) # lower cholesky factor
        Ri = np.linalg.inv(R)
        f = np.ones(test_size)[:,np.newaxis]
        self.F = np.ones(len(self.y))[:,np.newaxis]
        self.Y = (self.y - np.dot(self.F,self.Beta))
        y_predict = np.dot(f,self.Beta) + np.dot((np.dot(r_x.T,Ri)), self.Y)
        # variance = np.zeros(test_size)
        # for i in range(test_size):
        #     xtest = self.x_test[i,:][np.newaxis,:]
        #     r_xi = self.compute_R(self.x,xtest,self.theta)
        #     variance[i] = self.sigma2*(1 - np.dot(np.dot(r_xi.T,self.Ri), r_xi) + \
        #     np.dot((1 - \
        #     np.dot(np.dot((self.F).T,self.Ri),r_xi))**2,np.linalg.inv(np.dot(np.dot((self.F).T,self.Ri),self.F))))
        self.y_predict = self.y_scaler.inverse_transform(y_predict)
        # self.variance = variance
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
