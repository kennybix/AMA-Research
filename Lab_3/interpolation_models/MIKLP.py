import numpy as np
import math
import scipy
from scipy.optimize import minimize
from interpolation_models.core import nearestPD as NPD
from interpolation_models.core import constrNMPy as cNM
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import MinMaxScaler as MS
import cma
from sklearn.cross_decomposition import PLSRegression  

'''
Improvements:
- in-house preprocessing

'''
class Kriging:

    def __init__(self,x,y,kernels,theta0="",weight0="",weight_optimizer="SLSQP",optimizer="SLSQP",optimizer_noise=1.0,eps=1.48e-08,restarts=1,preprocessing="standardize",pls_n_comp=""):

        if pls_n_comp == "":
            self.pls_n_comp = 1
        else:
            self.pls_n_comp = pls_n_comp
        
        self.pls = PLSRegression(n_components=self.pls_n_comp)
        self.coeff_pls = self.pls.fit(x.copy(),y.copy()).x_rotations_


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

        self.kernels = kernels
        self.Ns = self.x.shape[0]
        self.Nk = self.x.shape[1]
        
        if theta0 == "": # to ensure varying initial guesses across board
            theta0 = []
            for i in range(self.pls_n_comp):
                theta0.append(np.random.uniform(1e-2,5))
        else:
            theta0 = theta0        
        self.theta0 = theta0

        # self.optimizer = optimizer
        self.optimizer = 'nelder-mead-c' #hardcoded--to be changed later!
        self.weight_optimizer = weight_optimizer
        self.weight_optimizer =  self.optimizer #use same optimizers for both processes
        self.optimizer_noise = optimizer_noise
        self.eps = eps
        self.restarts = restarts
        if (weight0==""):
            weight0 = [1/len(self.kernels)] * len(self.kernels)
        weights = weight0
        for i in range(0,self.Nk-1):
            weights = np.vstack((weights,weight0))
        self.weights0 = weights


        if self.optimizer == "CMA-ES" and self.Nk == 1: # hack to avoid certain errors
            self.optimizer = "SLSQP"

    def compute_R(self,X,Y,theta):
        R_total = 1
        n = X.shape[0] # size of training sample
        m = Y.shape[0] # size of test sample

        for i in range(self.Nk):
            index = self.kernels_indexes[i]
            kernel = self.kernels[index] # select best kernel for each dimension
            # make fluid to accomodate cross-correlation matrix later
            R_p = 1
            for p in range(self.pls_n_comp):
                thetai = theta[p]  # get theta for each pls component
                R = np.zeros((n,m)) #initialize correlation for each variable#
                for j in range(n):
                    for k in range(m):
                        h = self.coeff_pls[i][p] * abs(X[j][i]-Y[k][i]) # compute wh
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
                R_p *=R
            R_total *= R_p
        return R_total

    def compute_R_w(self,X,Y,theta,weights):
        kernels = self.kernels
        #weights input must be flattened
        weights = np.array_split(weights,len(weights))
        R_total = 1
        n = X.shape[0] # size of training sample
        m = Y.shape[0] # size of test sample

        for i in range(self.Nk):
            R_p = 1
            for p in range(self.pls_n_comp):
                thetai = theta[p] #theta is a subject of the pls component
                # make fluid to accomodate cross-correlation matrix later
                R_w = np.zeros((n,m))
                for w in range(len(kernels)):#
                    kernel = kernels[w] #select the appropriate kernel
                    weight = weights.pop(0)
                    R = np.zeros((n,m)) #initialize correlation for each variable
                    
                    for j in range(n):
                        for k in range(m):
                            h = self.coeff_pls[i][p] * abs(X[j][i]-Y[k][i]) # compute the distance, h
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
                    R_w += weight * R
                R_p *= R_w
            R_total *= R_p
        # R_total = R
        return R_total

    def NLL_w(self, hyperparameter):
        theta = np.zeros(len(self.theta0))
        hyperparameter = np.array_split(hyperparameter,len(hyperparameter))
        for i in range(len(self.theta0)):
            theta[i] = hyperparameter.pop(0)
        weights = np.concatenate(hyperparameter)

        y = self.y
        n = len(y)
        R = self.compute_R_w(self.x,self.x,theta,weights)
        self.R = R # so I can reuse this R at any point in the code

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
        # self.likelihood = -float(nll) # make the likelihood value available
        return float(nll)


    def NLL(self, theta):
        y = self.y
        n = len(y)
        R = self.compute_R(self.x,self.x,theta)
        self.R = R # so I can reuse this R at any point in the code

        self.F = np.ones(self.Ns)[:,np.newaxis]
        FT = (self.F).T
        if (NPD.isPD(R) == False): 
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
        self.likelihood = -float(nll) # make the likelihood value available
        return float(nll)

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
        theta_bound = (0.00001,100000.0)
        weight_bound = (0,1)

        for i in range(len(self.theta0)): 
            bounds.append(theta_bound)
        weight_f = (np.array(self.weights0)).flatten()
        for j in range(len(weight_f)):
            bounds.append(weight_bound)

        if (self.optimizer=="CMA-ES"):
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
        
        elif self.optimizer == "nelder-mead-c":
            LB = []
            UB = []
            for i in range(len(bounds)):
                LB.append(bounds[i][0])
                UB.append(bounds[i][1])
            res = cNM.constrNM(self.NLL_w,xk,LB,UB,full_output=True)
            optimal_hyperparameter = res['xopt']

        elif self.optimizer == "nelder-mead" or "SLSQP" or "COBYLA" or "TNC":
            cons = ({'type':'eq','fun':self.constraint_func})
            res = minimize(self.NLL_w,xk,method=self.optimizer,bounds=bounds,constraints=cons,options={'disp':False})
            optimal_hyperparameter = res.x
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
        theta_bound = (0.00001,10000000)

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
        return self

    def train(self):
        self.get_weights()
        hyperparameter = self.theta_w
        self.get_theta(hyperparameter)
        self.mixed_kernels = get_kernel_names(self.kernels,self.kernels_indexes)
        self.info = {'weight':self.weights,
                        'Theta_with_weight':self.theta_w,
                        'chosen_kernels': self.mixed_kernels,
                        'Theta':self.theta,
                        'Likelihood':self.likelihood}
        return self


    def predict(self,testdata):
        if(self.Nk == 1):
            testdata = testdata[:,np.newaxis]
        self.testdata = self.x_scaler.transform(testdata) # scale the test points
        self.x_test = self.testdata
        test_size = self.x_test.shape[0]
        # compute r(x)
        r_x = self.compute_R(self.x,self.x_test,self.theta)
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

def get_kernel_names(kernel_list,kernel_index):
    mixed_kernel = []
    for i in range(len(kernel_index)):
        mixed_kernel.append(kernel_list[kernel_index[i]])
    return mixed_kernel