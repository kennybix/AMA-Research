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
# import gurobipy as gp 
# import pyopt as po
from sklearn.metrics.pairwise import check_pairwise_arrays

'''
Improvements:
- in-house preprocessing

'''
class Kriging:

    def __init__(self,x,y,kernel,theta0="",optimizer="nelder-mead-c",optimizer_noise=1.0,eps=1.48e-08,restarts=1,preprocessing="normalize"):
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

        if theta0 == "":
            self.theta0 = [0.5] * self.Nk
        else:
            self.theta0 = theta0

        if self.optimizer == "CMA-ES" and self.Nk == 1: # hack to avoid certain errors
            self.optimizer = "nelder-mead-c"

    def compute_R(self,D,theta):
        d = np.einsum("j,ij->ij", (1/theta).T, D) #I don't need to compute this everytime
        kernel = self.kernel
        R = 1
        for i in  range(self.Nk):
            d_comp = (d[:,i]).reshape(d.shape[0],1)
            if kernel == "exponential":
                K = (np.exp(-1.0 * np.abs(d_comp)))
            elif kernel == "matern3_2":
                K = (1 + (np.sqrt(3)*np.abs(d_comp)))*np.exp(((-np.sqrt(3))*np.abs(d_comp))) 
            elif kernel == "gaussian":
                K = (np.exp(-0.5 * (d_comp ** 2))) #on the suspicion that d is already squared
            elif kernel == "matern5_2":
                K = (1 + (np.sqrt(5)*np.abs(d_comp)) + ((5/3)*d_comp**2))*np.exp(((-np.sqrt(5))*np.abs(d_comp))) 
            else:
                print("Unknown kernel")
            R *= K                                                                                                                                                             
        return R

    def NLL(self, theta):
        nugget = 2.22e-11 # a very small value
        nugget = 0 # strict interpolation

        # Calculate matrix of distances D between samples
        D, self.ij = cross_distances(self.x)
        # compute the correlation matrix
        r = self.compute_R(D,theta)
        R = np.eye(self.Ns) * (1.0 + nugget)
        R[self.ij[:, 0], self.ij[:, 1]] = r[:, 0]
        R[self.ij[:, 1], self.ij[:, 0]] = r[:, 0]

        y = self.y
        n = len(y)
        self.F = np.ones(self.Ns)[:,np.newaxis]
        # self.R = R # so I can reuse this R at any point in the code
        if not(NPD.isPD(R)): #sane check
            R = NPD.nearestPD(R)
        else:
            R = R
        # R = np.linalg.cholesky(R) # lower cholesky factor
        self.R = R
        # y = self.y
        # n = len(y)
        # R = self.compute_R(self.x,self.x,theta)
        # self.R = R # so I can reuse this R at any point in the code
        # self.F = np.ones(self.Ns)[:,np.newaxis]


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
        # return nll 

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
            xopts, es = cma.fmin2(self.NLL,xk,1.0,{'bounds':new_bounds,'verbose':-9,'CMA_stds':xk},restarts=self.restarts)
            if xopts is None:
                theta = es.best
            else:
                theta = xopts
            self.theta = theta
        
        # elif self.optimizer == "pyopt":
            
        #     opt_prob = po.Optimization('Optimization problem',self.NLL)
        #     opt_prob.addVar(xk,'c',lower=0.0,upper=1.0,value= 0.5)
        #     [fstr, theta, inform] = slsqp(opt_prob,sens_type='FD')

        # elif self.optimizer == "gurobi":
        #     gp_model = gp.Model()
        #     xk = gp_model.addVar(lb=0.0001,ub=1.0)
        #     gp_model.setObjective(self.NLL)
        #     gp_model.optimize()
        #     theta = xk.X
        # def add_constr(model, expression, name=""):
        #     if expression is True:
        #         return model.addConstr(0, GRB.EQUAL, 0, name)
        #     elif expression is False:
        #         raise Exception('`False` as constraint for {}'.format(name))
        #     else:
        #         return model.addConstr(expression, name)
        elif self.optimizer == "nelder-mead-c":
            LB = []
            UB = []
            for i in range(len(bounds)):
                LB.append(bounds[i][0])
                UB.append(bounds[i][1])
            res = cNM.constrNM(self.NLL,xk,LB,UB,full_output=True)
            theta = res['xopt']

        elif self.optimizer == "COBYLA" :
            theta_bounds = [1e-6,2e6]
            constraints = []
            limit, _rhobeg = 10 *self.Nk, 0.5
            for ii in range(self.Nk):
                constraints.append(lambda theta, i=ii: theta[i] - theta_bounds[0])
                constraints.append(lambda theta, i=ii: theta_bounds[1] - theta[i])
            res = minimize(self.NLL,xk,constraints=[{"fun": con, "type": "ineq"} for con in constraints],
                        method=self.optimizer)
            # ,options={"rhobeg": _rhobeg, "tol": 1e-4, "maxiter": limit}
            theta = np.copy(res.x) #initialization

        elif self.optimizer == "nelder-mead" or "SLSQP" or "TNC":
                res1 = minimize(self.NLL,xk,method=self.optimizer,bounds=bounds,options={'ftol':1e-20,'disp':False})
                theta = res1.x


            
        self.theta = theta
        # self.likelihood = self.NLL(theta)

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

        NLL = self.NLL(self.theta) #an hack to get all parameters and make the obj func evaluation available
        # compute r(x)
        r_x = (self.compute_R(dx,self.theta)).reshape(test_size,self.Ns)
        # self.R = self.compute_R(self.x,self.x,self.theta) #hack for ensemble method
        f = np.ones(test_size)[:,np.newaxis]
        y_predict = np.dot(f,self.Beta) + np.dot((np.dot(r_x,self.Ri)), self.Y)
        variance = np.zeros(test_size)
        for i in range(test_size):
            xtest = self.x_test[i,:][np.newaxis,:]
            dxi = differences(xtest, Y=self.x.copy())
            r_xi = (self.compute_R(dxi,self.theta)).reshape(1,self.Ns)

            variance[i] = self.sigma2*(1 - np.dot(np.dot(r_xi,self.Ri), r_xi.T) + \
            np.dot((1 - \
            np.dot(np.dot((self.F).T,self.Ri),r_xi.T))**2,np.linalg.inv(np.dot(np.dot((self.F).T,self.Ri),self.F))))
        self.y_predict = self.y_scaler.inverse_transform(y_predict)
        self.variance = variance
        return self.y_predict,self.variance

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