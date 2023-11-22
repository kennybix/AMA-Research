import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy.optimize import minimize
from interpolation_models.core import nearestPD as NPD
from interpolation_models.core import constrNMPy as cNM
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import MinMaxScaler as MS
from sklearn.metrics.pairwise import check_pairwise_arrays
import cma

from scipy.special import kv,kn,gamma

'''
Improvements:
- free form Matern kernels
- in-house preprocessing

'''
class Kriging:

    def __init__(self,x,y,kernels="",theta0="",nu0="",optimizer="CMA-ES",optimizer_noise=0.01,eps=1.48e-08,restarts=3,preprocessing="normalize"):
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

        # self.optimizer = optimizer
        self.optimizer = 'COBYLA' #hardcoded
        self.optimizer_noise = optimizer_noise
        self.eps = eps
        self.restarts = restarts
        self.Ns = self.x.shape[0]
        self.Nk = self.x.shape[1]
        theta0 = [] #empty list
        for i in range(self.Nk):
            theta0.append(np.random.uniform(1e-2,5))
        self.theta0 = theta0
        print("Initial theta value: {0}".format(self.theta0))

        # self.nu0 = [0.6] * self.Nk
        nu0 = []
        for j in range(self.Nk):
            nu0.append(np.random.uniform(0.5,2.5))
        self.nu0 = nu0
        print("Initial nu value: {0}".format(self.nu0))
        self.likelihood = -1 #initial location
        self.likelihood_threshold = 30 #failsafe system
        if self.optimizer == "CMA-ES" and self.Nk == 1: # hack to avoid certain errors
            self.optimizer = "nelder-mead-c"

    def compute_R(self,D,theta,nu):
        # Notations:
        #  distance = the Ns x Ns matrix
        #  K = the Ns x Ns spatial correlation matrix following the Matern class
        #  nu = the variable of the free-form matern; 
        #  theta = the correlation range
        #  kv = Modified Bessel function of second kind of real order v(scipy.special function)
        #  gamma = Gamma function (scipy.special function)
        #  This form of the Matern is found in Rasmussen, Carl Edward and Williams,Christopher K. I. (2006) Gaussian Processes for Machine Learning.    
  
        d = np.einsum("j,ij->ij", (1/theta).T, D) #I don't need to compute this everytime
        # R = 1
        R = np.ones(d.shape[0])
        # nu = nu[0]
        for i in  range(self.Nk):
            nu_d = nu[i]
            d_comp = (d[:,i]).reshape(d.shape[0],1)
            d_comp = np.abs(d_comp)
            K = (np.power(2,(1-nu_d)) / gamma(nu_d)) * np.power((np.sqrt(2*nu_d) * d_comp),nu_d) * kv(nu_d,np.sqrt(2*nu_d)*d_comp) # Matern free-form function
            K = [x[0] for x in K]
            K = [1.0 if math.isnan(x) else x for x in K]
            # if(math.isnan(R[j,k])):
            #     R[j,k] = 1.0
            R =[a*b for a,b in zip(R,K)]
            # R *= K                                                                                                                                                             
        return R


    def NLL(self, hyperparameter):
        nugget = 2.22e-11
        # nugget = 0
        hyperparameter = np.array_split(hyperparameter,len(hyperparameter))
        nu = []
        for i in range(self.Nk):
            nu.append(hyperparameter.pop(0))
        theta = np.concatenate(hyperparameter)
        y = self.y
        n = len(y)

        # Calculate matrix of distances D between samples
        D, self.ij = cross_distances(self.x)
        # compute the correlation matrix
        r = self.compute_R(D,theta,nu)
        if isinstance(r,list):
            r = (np.array(r)).reshape(len(r),1)
        else: r = r
        
        R = np.eye(self.Ns) * (1.0 + nugget)
        R[self.ij[:, 0], self.ij[:, 1]] = r[:, 0]
        R[self.ij[:, 1], self.ij[:, 0]] = r[:, 0]

        # R = self.compute_R(self.x,self.x,theta,nu)
        # np.fill_diagonal(R,1.0) #this should only be for the training samples

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
        return float(nll)



    def get_theta(self, theta0, nu0):
        xk  =  nu0 + theta0
        bounds = []
        nu_bound = (0.5,3.0)
        theta_bound = (1e-2,6e1)

        for j in range(len(self.nu0)):
            bounds.append(nu_bound)
        for i in range(len(self.theta0)):
            bounds.append(theta_bound)  
        
        while(self.likelihood < self.likelihood_threshold):
            if (self.optimizer=="CMA-ES"):
                LB = []
                UB = []
                for i in range(len(bounds)):
                    LB.append(bounds[i][0])
                    UB.append(bounds[i][1])
                new_bounds = [LB,UB]
                xopts, es = cma.fmin2(self.NLL, xk, 0.1,options={'bounds':new_bounds,'verbose':-9, 'popsize':40},restarts=self.restarts)
                if xopts is None:
                    hyperparameter = es.best
                else:
                    hyperparameter = xopts
        
            elif self.optimizer == "nelder-mead-c":
                LB = [0.5]*self.Nk  +  [1e-4]*len(self.theta0)
                UB = [3.0]*self.Nk  +  [1e3]*len(self.theta0)
                res = cNM.constrNM(self.NLL,xk,LB,UB,full_output=True)
                hyperparameter = res['xopt']

            elif self.optimizer == "COBYLA" :
                nu_bound = [0.5,3]
                theta_bounds = [1e-4,1e3]
                constraints = []
                limit, _rhobeg = 10*(self.Nk+2), 0.3

                for ii in range(2*self.Nk):
                    if (ii-self.Nk) < 0: #first half goes to nu
                        constraints.append(lambda hyperparameter, i=ii: hyperparameter[i] - nu_bound[0])
                        constraints.append(lambda hyperparameter, i=ii: nu_bound[1] - hyperparameter[i])
                    else:
                        constraints.append(lambda hyperparameter, i=ii: hyperparameter[i] - theta_bounds[0])
                        constraints.append(lambda hyperparameter, i=ii: theta_bounds[1] - hyperparameter[i])

                res = minimize(self.NLL,xk,constraints=[{"fun": con, "type": "ineq"} for con in constraints],
                            method=self.optimizer)
                # ,options={"rhobeg": _rhobeg, "tol": 1e-10, "maxiter": limit}
                # ,options={"rhobeg": _rhobeg, "tol": 1e-4, "maxiter": limit}
                hyperparameter = np.copy(res.x) #initialization

            elif self.optimizer == "nelder-mead" or "SLSQP" or "TNC":
                    res1 = minimize(self.NLL,xk,method=self.optimizer,bounds=bounds,options={'ftol':1e-20,'disp':False})
                    hyperparameter = res1.x
            xk = self.get_new_initial_points()
            self.likelihood = -float(self.NLL(hyperparameter))
        hyperparameter = np.array_split(hyperparameter,len(hyperparameter))
        self.nu = []
        for i in range(self.Nk):
            self.nu.append(hyperparameter.pop(0))
        # self.nu = hyperparameter.pop(0)
        self.theta = np.concatenate(hyperparameter)
        self.info = {'nu':self.nu,
                        'Theta':self.theta,
                        'Likelihood':self.likelihood}
        print(self.nu)
        print(self.theta)

    def train(self):
        self.get_theta(self.theta0,self.nu0)


    def predict(self,testdata):
        if(self.Nk == 1):
            testdata = testdata[:,np.newaxis]
        self.testdata = self.x_scaler.transform(testdata) # scale the test points
        self.x_test = np.array(self.testdata)
        test_size = self.x_test.shape[0]
        # Get pairwise componentwise L1-distances to the input training set
        dx = differences(self.x_test, Y=self.x.copy())

        # compute r(x)
        r_x = self.compute_R(dx,self.theta,self.nu)
        if isinstance(r_x,list):
            r_x = np.array(r_x)
        else:
            r_x = r_x 
        r_x = r_x.reshape(test_size,self.Ns)
        # r_x = self.compute_R(self.x,self.x_test,self.theta,self.nu)
        # print(r_x)
        R = self.R
        # R = np.linalg.cholesky(R) # lower cholesky factor
        f = np.ones(test_size)[:,np.newaxis]
        y_predict = np.dot(f,self.Beta) + np.dot((np.dot(r_x,self.Ri)), self.Y)
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
            try:
                sum += np.power((y_exact[i] - self.y_predict[i]),2)
            except:
                # self.y_predict = self.y_predict.reshape(m,)
                y_exact = np.asarray(y_exact)
                sum += np.power((y_exact[i] - self.y_predict[i]),2)
        self.RMSE = np.sqrt(sum / m)
        self.RMSE /= (np.max(y_exact)-np.min(y_exact))
        return self.RMSE
    def get_new_initial_points(self):
        new_start_point = []
        for j in range(self.Nk):
            new_start_point.append(np.random.uniform(0.5,3.0)) # nu part
        for i in range(self.Nk): # theta part
            new_start_point.append(np.random.uniform(1e-2,5))
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

