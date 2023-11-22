import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.random import random
import scipy
from datetime import datetime
from scipy.optimize import minimize
from interpolation_models.core import nearestPD as NPD
from interpolation_models.core import constrNMPy as cNM
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import MinMaxScaler as MS
from sklearn.metrics.pairwise import check_pairwise_arrays
from scipy import linalg
import optuna
import psycopg2

from scipy.special import kv,kn,gamma

'''
Improvements:
- free form Matern kernels
- in-house preprocessing
- caterogical selection of kernels

'''
class Kriging:

    def __init__(self,x,y,kernels,theta0="",nu0="",optimizer="CMA-ES",optimizer_noise=0.01,eps=1.48e-08,restarts=3,preprocessing="normalize"):
 

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

        self.kernel = "" #just a dummy to conform
        
        # self.optimizer = optimizer
        self.optimizer = 'COBYLA' #hardcoded---change later
        # self.optimizer = 'optuna'
        self.optimizer_noise = optimizer_noise
        self.eps = eps
        self.restarts = restarts
        self.Ns = self.x.shape[0]
        self.Nk = self.x.shape[1]
        self.likelihood = -1 # initial point
        self.likelihood_threshold = 60
        if theta0 == "":
            theta0 = [] #empty list
            for i in range(self.Nk):
                theta0.append(np.random.uniform(1e-2,5))
        else: 
            theta0 = theta0
        self.theta0 = theta0
        print("Initial theta value: {0}".format(self.theta0))
        if nu0 == "":
            # self.nu0 = [np.random.uniform(0.5,2.5)]
            self.nu0 = [np.random.choice([0.5,1.5,2.5,np.inf])]
        else:
            self.nu0 = nu0
        print("Initial nu value: {0}".format(self.nu0))
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
        
        # ensure that theta is an array
        if isinstance(theta,list):
            theta = convert_array_to_float(theta)
            theta = np.array(theta)
        else:
            # convert to list, adjust then back to array
            theta = theta.tolist()
            theta = convert_array_to_float(theta)
            theta = np.array(theta) 
        d = np.einsum("j,ij->ij", (1/theta).T, D) #I don't need to compute this everytime
        # R = 1
        R = np.ones(d.shape[0])
        nu = nu[0]
        for i in  range(self.Nk):
            d_comp = (d[:,i]).reshape(d.shape[0],1)
            d_comp = np.abs(d_comp)
            
            if nu == 'inf': #use Gaussian kernel
                K = np.exp(-0.5 * (d_comp**2))
            else: #use the general matern kernel
                nu = float(nu)
                K = (np.power(2,(1-nu)) / gamma(nu)) * np.power((np.sqrt(2*nu) * d_comp),nu) * kv(nu,np.sqrt(2*nu)*d_comp) # Matern free-form function
            K = [x[0] for x in K]
            K = [1.0 if math.isnan(x) else x for x in K]
            # if(math.isnan(R[j,k])):
            #     R[j,k] = 1.0
            R =[a*b for a,b in zip(R,K)]
            # R *= K                                                                                                                                                             
        return R


    # def NLL(self, hyperparameter):
    #     nugget = 2.22e-11
    #     # nugget = 0
    #     hyperparameter = np.array_split(hyperparameter,len(hyperparameter))
    #     nu = hyperparameter.pop(0)
    #     theta = np.concatenate(hyperparameter)
    #     theta = convert_array_to_float(theta) # to ensure that theta is always a value and not a string
    #     y = self.y
    #     n = len(y)

    #     # Calculate matrix of distances D between samples
    #     D, self.ij = cross_distances(self.x)
    #     # compute the correlation matrix
    #     r = self.compute_R(D,theta,nu)
    #     if isinstance(r,list):
    #         r = (np.array(r)).reshape(len(r),1)
    #     else: r = r
        
    #     R = np.eye(self.Ns) * (1.0 + nugget)
    #     R[self.ij[:, 0], self.ij[:, 1]] = r[:, 0]
    #     R[self.ij[:, 1], self.ij[:, 0]] = r[:, 0]

    #     # R = self.compute_R(self.x,self.x,theta,nu)
    #     # np.fill_diagonal(R,1.0) #this should only be for the training samples

    #     if not(NPD.isPD(R)): #sane check
    #         R = NPD.nearestPD(R)
            
    #     self.R = R
    #     self.Ns = self.x.shape[0]#hack for EGO
    #     self.F = np.ones(self.Ns)[:,np.newaxis]
    #     FT = (self.F).T
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



    def NLL(self, hyperparameter):
        nugget = 2.22e-11
        # nugget = 0
        hyperparameter = np.array_split(hyperparameter,len(hyperparameter))
        nu = hyperparameter.pop(0)
        theta = np.concatenate(hyperparameter)
        theta = convert_array_to_float(theta) # to ensure that theta is always a value and not a string
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


    def objective(self,trial):
        # define theta bound
        # theta_bound = (1e-4,1e3)
        theta_bound = (1e-1,3e1)
        # create the variable
        theta = np.zeros(self.Nk) #create empty slots

        nu  = trial.suggest_categorical('nu_v', ['0.5', '1.5', '2.5','inf'])
        for i in range(self.Nk): # loop through the dimension
            variable = 'theta' + str(i)
            theta[i] = trial.suggest_float(variable, theta_bound[0], theta_bound[1])
        
        hyperparameter = [nu] + theta.tolist()
        return self.NLL(hyperparameter)

    def get_theta(self):
        # optuna as the only optimizer
        SEED = 42
        np.random.seed(SEED)
        conn = psycopg2.connect("dbname=mydb user=postgres password=Ayodeji@1994")

        name = get_study_name()
        study = optuna.create_study(direction="minimize",
                                    sampler = optuna.samplers.TPESampler(seed=SEED),
                                    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5),
                                    study_name=name,
                                storage="postgresql://postgres:Ayodeji@1994@localhost/mydb",
                                load_if_exists=True)
        while (self.likelihood < self.likelihood_threshold):
            study.optimize(self.objective, n_trials=50,n_jobs=12)
            hyperparameter = list((study.best_trial.params).values())       
            self.likelihood = -float(self.NLL(hyperparameter)) # sane coding

        hyperparameter = np.array_split(hyperparameter,len(hyperparameter))
        self.nu = hyperparameter.pop(0)
        self.theta = np.concatenate(hyperparameter)
        self.info = {'selected kernel':self.nu,
                        'Theta':self.theta,
                        'Likelihood':self.likelihood}
        print(self.nu)
        print(self.theta)
        print(self.likelihood)

    def train(self):
        self.get_theta()

    # def predict(self,testdata):
    #     if(self.Nk == 1):
    #         testdata = testdata[:,np.newaxis]
    #     self.testdata = self.x_scaler.transform(testdata) # scale the test points
    #     self.x_test = np.array(self.testdata)
    #     test_size = self.x_test.shape[0]
    #     # Get pairwise componentwise L1-distances to the input training set
    #     dx = differences(self.x_test, Y=self.x.copy())

    #     # compute r(x)
    #     r_x = self.compute_R(dx,self.theta,self.nu)
    #     if isinstance(r_x,list):
    #         r_x = np.array(r_x)
    #     else:
    #         r_x = r_x 
    #     r_x = r_x.reshape(test_size,self.Ns)
    #     # r_x = self.compute_R(self.x,self.x_test,self.theta,self.nu)
    #     # print(r_x)
    #     R = self.R
    #     # R = np.linalg.cholesky(R) # lower cholesky factor
    #     f = np.ones(test_size)[:,np.newaxis]
    #     y_predict = np.dot(f,self.Beta) + np.dot((np.dot(r_x,self.Ri)), self.Y)
    #     variance = np.zeros(test_size)
    #     # for i in range(test_size):
    #     #     r_xi = self.compute_R(self.x,self.x_test[i,:],self.theta,self.nu)
    #     #     variance[i] = self.sigma2*(1 - np.dot(np.dot(r_xi.T,self.Ri), r_xi) + \
    #     #     np.dot((1 - \
    #     #     np.dot(np.dot((self.F).T,self.Ri),r_xi))**2,np.linalg.inv(np.dot(np.dot((self.F).T,self.Ri),self.F))))
    #     self.y_predict = self.y_scaler.inverse_transform(y_predict)
    #     self.variance = variance
    #     return self.y_predict

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
        r_x = (self.compute_R(dx,self.theta,self.nu))
        if isinstance(r_x,list):
            r_x = np.array(r_x)
        else:
            r_x = r_x
        r_x = r_x.reshape(test_size,self.Ns) 
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
        r_x = (self.compute_R(dx,self.theta,self.nu)).reshape(test_size,self.Ns)
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

def convert_array_to_float(string_array):
    num_array = []
    for i in range(len(string_array)):
        num_array.append(float(string_array[i]))
    return num_array


def get_study_name():
    stamp = str(datetime.now())
    stamp = stamp.replace(" ", "_") # replace space
    stamp = stamp.replace("-", "_") # replace -
    stamp = stamp.replace(":", "_") # replace :
    name = 'study'+'_'+stamp
    return name