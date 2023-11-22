# import necessary libraries

import numpy as np 
from Adaptive_sampling import kriging_ego_2 as KCORE2
from Adaptive_sampling import lhs 
from Adaptive_sampling import scaler 
from Adaptive_sampling import constrNMPy as cNM
from scipy.optimize import minimize
from Adaptive_sampling import halton_sequence as hs
from scipy.special import erf
import pyDOE2 as pyd

'''
Input:
        1. Objective function
        2. Starting point counts
        3. Stopping criteria or final counts
        4. Number of validation points

Output:
        1. Improved Kriging model
        2. Improved sample points
'''

'''
write function to get nearest x


'''
def get_data(index,data_array):
    m = len(index)
    data_array = np.array(data_array)
    data = []
    for i in range(m):
        data.append(data_array[int(index[i])])
    return data

def closest(points, x):
    if isinstance(points,list):
        points = np.asarray(points)
    else:
        points = points
    
    if isinstance(x,list):
        x = np.asarray(x)
    else:
        x = x

    return sorted(points, key=lambda p: np.linalg.norm(p-x))[:1] # 1 for top one

class adaptive_sample():
    def __init__(self,obj_func,initial_guess="",bounds="",initial_ns="",max_steps="",DOE="LHS",kernel="gaussian",tolerance=1e-4):

        self.obj_func = obj_func
        if initial_guess == "":
            initial_guess = [] #empty list
            #pick starting point within the bounds
            for bound in range(len(bounds)):
                initial_guess.append(np.random.uniform(bounds[bound][0],bounds[bound][1]))
        else:
            initial_guess = initial_guess
        self.initial_guess = initial_guess
        self.bounds = bounds
        self.LB = []
        self.UB = []
        for i in range(len(self.bounds)):
            self.LB.append(self.bounds[i][0])
            self.UB.append(self.bounds[i][1])
        self.dim = len(self.bounds) #get this from the bounds
        self.tolerance = tolerance
        # if (initial_ns > (self.x_data).shape[0]): #hack to ensure the initial sample size is lesser than the data size
        #     self.initial_ns = initial_ns / 1.5
        # else:
        #     self.initial_ns = initial_ns

        self.initial_ns = initial_ns
        self.max_steps = max_steps
        self.DOE = DOE
        self.kernel = kernel

    def loss_func(self,x):
        #hybrid optimization because we just search but an efficient search
        # we have to always update the model and refresh
        # m, s = self.model.predict(x)
        # fmin = self.model.get_fmin()
        x = x.reshape(1,self.dim)
        fhat,s2 = self.model.predict(x) #modify predict to return two values just for this process
        fhat = fhat.reshape(len(s2),)
        s2 = s2.reshape(len(s2),)
        y = self.obj_func(x)
        if isinstance(y,list):
            y = np.array(y)
        else:
            y = y
        #have different formulations to compute the loss function, so I can select 
        # EI inclusive
        with np.errstate(divide='warn'):
            EI = (self.y_min - fhat) * (0.5 + 0.5*erf((self.y_min - fhat)/np.sqrt(2 * s2))) + \
                    np.sqrt(0.5*s2/np.pi)*np.exp(-0.5*(self.y_min - fhat)**2/s2)
            EI[s2 == 0.0] = 0.0 #no improvement is expected at sample points
        # L = 0.5*(y-fhat)**2 + 0.5*s2 #modified for maximization
        L = EI
        return -L
    def get_optimal_samples(self):
        
        # generate initial samples, we can even do that using our previous code
        if (self.DOE=="LHS"):
            x_initial_sample = (lhs.sample(self.dim,self.initial_ns)).T
        elif (self.DOE=="HS"):
            x_initial_sample = hs.halton_sequence(self.initial_ns,self.dim)

        # now scale
        x_initial_sample = scaler.scale(x_initial_sample,self.bounds)  

        # get the corresponding f values       
        y_initial_sample = self.obj_func(x_initial_sample)

        # y_sample = y_sample.reshape(len(y_sample),1) #single output 

        x_sample = x_initial_sample
        y_sample = y_initial_sample

        #initialize the list of f_min
        f_min_list = (np.linspace(1,4,4)).tolist() #f_min_list = [1.0,2.0,3.0,4.0]

        theta0 = [0.01]*self.dim
        count = 0 #initializing count
        starting_point = self.initial_guess
        self.y_min = min(y_sample) #initialization
        self.bounds = np.array(self.bounds)
        while(count<self.max_steps): #stopping criteria
            # main task: get new points
            # build surrogate model
            self.model = KCORE2.Kriging(x_sample,y_sample,self.kernel,theta0=theta0,optimizer="nelder-mead-c") #initialiize model
            self.model.train() # train model
            #using the value of theta in last training as the initial value to help with faster convergence
            theta0 = self.model.theta #update the value of theta0 

            # optimize using manual search to get new point
            #choose the starting point randomly within the bound domain
            # starting_point = (np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(1, self.dim)))[0]
            res = cNM.constrNM(self.loss_func,starting_point,self.LB,self.UB,full_output=True)
            x_new = res['xopt']
            # res = minimize(self.loss_func, x0=starting_point, bounds=self.bounds, method='L-BFGS-B')
            # x_new = res.x
            starting_point = x_new 
            x_new = x_new.reshape(1,self.dim)
            y_new = self.obj_func(x_new)
            f_min_list = push_and_pop(y_new[0],f_min_list) #add the new minima to the list
            y_new = y_new.reshape(1,1)
            y_sample = y_sample.reshape(len(y_sample),1)
            #add new point to existing sample
            x_sample = np.vstack((x_sample,x_new))
            y_sample = np.vstack((y_sample,y_new))  
            self.y_min = min(y_sample) #update
            count +=1
            
            #get optimal  results
            self.fmin = min(y_sample[:,0]) #choose the minimum
            #get the corresponding x point at f_min
            min_f_point = closest(y_sample,self.fmin)
            min_pos = ((y_sample[:,0]).tolist()).index(min_f_point[0])
            self.x = x_sample[min_pos]
            # self.fmin = y_new

            if(np.var(f_min_list)<=self.tolerance):break #convergence criteria

            # print(self.x)
            # print(self.fmin)
        return x_sample,y_sample
    

def push_and_pop(value,current_list):
    #delete the first member of the list
    del(current_list[0])
    #add a new member to the list
    current_list.append(value)
    return current_list

# import numpy as np
# from scipy.stats import norm
# from scipy.optimize import minimize

# def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
#     ''' 
#     Computes the EI at points X based on existing samples X_sample 
#     and Y_sample using a Gaussian process surrogate model. 
#     Args: X: Points at which EI shall be computed (m x d). X_sample: 
#     Sample locations (n x d). Y_sample: Sample values (n x 1). 
#     gpr: A GaussianProcessRegressor fitted to samples. 
#     xi: Exploitation-exploration trade-off parameter. 
#     Returns: Expected improvements at points X. 
#     '''
#     mu, sigma = gpr.predict(X, return_std=True)
#     mu_sample = gpr.predict(X_sample)

#     sigma = sigma.reshape(-1, X_sample.shape[1])
    
#     # Needed for noise-based model,
#     # otherwise use np.max(Y_sample).
#     # See also section 2.4 in [...]
#     mu_sample_opt = np.max(mu_sample)

#     with np.errstate(divide='warn'):
#         imp = mu - mu_sample_opt - xi
#         Z = imp / sigma
#         ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
#         ei[sigma == 0.0] = 0.0

#     return ei



# def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
#     ''' 
#     Proposes the next sampling point by optimizing the acquisition function. 
#     Args: acquisition: Acquisition function. X_sample: Sample locations (n x d). 
#     Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor fitted to samples. 
#     Returns: Location of the acquisition function maximum. 
#     '''
#     dim = X_sample.shape[1]
#     min_val = 1
#     min_x = None
    
#     def min_obj(X):
#         # Minimization objective is the negative acquisition function
#         return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)
    
#     # Find the best optimum by starting from n_restart different random points.
#     for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
#         res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')        
#         if res.fun < min_val:
#             min_val = res.fun[0]
#             min_x = res.x           
            
#     return min_x.reshape(-1, 1)