# import necessary libraries

import numpy as np
from numpy.lib.function_base import diff 
from scipy.optimize import minimize
from active_learning import kriging_ego_2 as KCORE2
from active_learning import lhs 
from active_learning import scaler 
from active_learning import constrNMPy as cNM
from active_learning import PSO
from scipy.optimize import minimize
from active_learning import halton_sequence as hs
from scipy.special import erf
from sklearn.preprocessing import MinMaxScaler as MS
import pyDOE2 as pyd

from active_learning import map_sampling_plan as mps 
from active_learning import sampling_plan_for_real_data as sprd

# Modified 30-May-2022
# Author: Kehinde Oyetunde


'''
Input:
        1. Input and output data
        2. Initial sample points
        3. Stopping criteria or final counts
        4. Preferred DOE method
        5. Kernel to be used

Output:
        1. Improved Kriging model
        2. Improved sample points
'''


'''
The stopping criteria here is the variance of the maximum improvement


'''
def get_data(index,data_array):
    m = len(index)
    data_array = np.array(data_array)
    data = []
    for i in range(m):
        data.append(data_array[int(index[i])])
    return data

# def closest(points, x):
#     if isinstance(points,list):
#         points = np.asarray(points)
#     else:
#         points = points
    
#     if isinstance(x,list):
#         x = np.asarray(x)
#     else:
#         x = x

#     return sorted(points, key=lambda p: np.linalg.norm(p-x))[:1] # 1 for top one

class active_learn():
    def __init__(self,whole_data,initial_ns,stopping_criteria=0.3,max_steps=1000,DOE="LHS",kernel="gaussian",tolerance=1e-4):
        # whole_data = {"x_data":x_data, "y_data":y_data} #nature of the whole data
        self.whole_data = whole_data
        self.x_data = np.copy(self.whole_data["x_data"])
        self.y_data = np.copy(self.whole_data["y_data"])        

        self.x_scaling_model = MS()
        self.x_scaling_model.fit(self.x_data ) #the fitting dataset
        self.x_data_norm = self.x_scaling_model.transform(self.x_data)
        self.x_data_norm = (self.x_data_norm).tolist() 
        #get the bounds from the data
        self.bounds = get_bounds(self.x_data) # getting data bounds
        self.LB = []
        self.UB = []
        for i in range(len(self.bounds)):
            self.LB.append(self.bounds[i][0])
            self.UB.append(self.bounds[i][1])    
        self.initial_guess = []
        for j in range(len(self.bounds)):
            self.initial_guess.append(np.random.uniform(self.bounds[j][0],self.bounds[j][1]))   
        self.stopping_criteria = stopping_criteria
        self.dim = (self.x_data).shape[1] #get this from the x data
        self.tolerance = tolerance

        self.initial_ns = initial_ns
        self.max_steps = max_steps
        self.DOE = DOE
        self.kernel = kernel

    def closest_final(self,points, x):
        # if isinstance(points,list):
        #     points = np.asarray(points)
        # else:
        #     points = points
        
        # if isinstance(x,list):
        #     x = np.asarray(x)
        # else:
        #     x = x

        closest_point = sorted(points, key=lambda p: np.linalg.norm(p-x))[:1] # 1 for top one
        return closest_point

    def closest(self,points, x):
        if isinstance(points,list):
            points = np.asarray(points)
        else:
            points = points
        
        if isinstance(x,list):
            x = np.asarray(x)
        else:
            x = x

        normalized_closest_point = sorted(points, key=lambda p: np.linalg.norm(p-x))[:1] # 1 for top one
        denormalized_closest_point = (self.x_scaling_model).inverse_transform(normalized_closest_point)
        denormalized_closest_point = self.closest_final(self.whole_data["x_data"],denormalized_closest_point)
        return denormalized_closest_point

    def compute_EI(self,x):
        x = x.reshape(1,self.dim)
        fhat,s2 = self.gp_old.predict(x) #modify predict to return two values just for this process
        fhat = fhat.reshape(len(s2),)
        s2 = s2.reshape(len(s2),)
        # have different formulations to compute the loss function, so I can select 
        # EI inclusive
        with np.errstate(divide='warn'):
            EI = (self.y_min - fhat) * (0.5 + 0.5*erf((self.y_min - fhat)/np.sqrt(2 * s2))) + \
                    np.sqrt(0.5*s2/np.pi)*np.exp(-0.5*(self.y_min - fhat)**2/s2)
            EI[s2 == 0.0] = 0.0 #no improvement is expected at sample points
        # L = 0.5*(y-fhat)**2 + 0.5*s2 #modified for maximization
        return EI

    def loss_func(self,x):
        #hybrid optimization because we just search but an efficient search
        # we have to always update the model and refresh
        # m, s = self.gp_old.predict(x)
        # fmin = self.gp_old.get_fmin()
        x = x.reshape(1,self.dim)
        x = self.closest(self.x_scaling_model.transform(self.x_data),self.x_scaling_model.transform(x)) #get the similar value of x in the real dataset
        
        fhat,s2 = self.gp_old.predict(x) #modify predict to return two values just for this process
        fhat = fhat.reshape(len(s2),)
        s2 = s2.reshape(len(s2),)
        # choose the closest using x_data and not the whole data
        # x_closest = self.closest(self.x_scaling_model.transform(self.x_data),self.x_scaling_model.transform(x)) #get the similar value of x in the real dataset
        
        pos = ((self.x_data)).index((x[0]).tolist())       
        y = (self.y_data)[pos] # get the approximate value of y from the data
        
        if isinstance(y,list):
            y = np.array(y)
        else:
            y = y
        L = 0.5*(y-fhat)**2 + 0.5*s2 #modified for maximization
        return -L

    # def loss_func(self,x,y):
    #     #hybrid optimization because we just search but an efficient search
    #     # we have to always update the model and refresh
    #     # x = x.reshape(1,self.dim)
    #     fhat,s2 = self.gp_old.predict(x) #modify predict to return two values just for this process
    #     fhat = fhat.reshape(len(s2),)
    #     s2 = s2.reshape(len(s2),)
    #     if isinstance(y,list):
    #         y = np.array(y)
    #     #have different formulations to compute the loss function, so I can select 
    #     # EI inclusive
    #     # EI = (self.y_best - y_hat) * (0.5 + 0.5*erf((self.y_best - y_hat)/np.sqrt(2 * SSqr))) + \
    #     #             np.sqrt(0.5*SSqr/np.pi)*np.exp(-0.5*(self.y_best - y_hat)**2/SSqr)
    #     L = 0.5*(y-fhat)**2 + 0.5*s2 #modified for maximization
    #     return -L

    def get_more_samples(self):
        
        # generate initial samples, we can even do that using our previous code
        sprd_model = sprd.Sampling_plan(self.x_data,self.y_data,Ns=self.initial_ns,sequence=self.DOE) #get the initial sample
        # Halton sequence is used as the default DOE method to choose initial samples
        x_initial_sample,y_initial_sample = sprd_model.create_samples() #default is Halton sequence
        # make it easy to get the starting points
        self.x_initial_sample = x_initial_sample
        self.y_initial_sample = y_initial_sample
        count = self.initial_ns #initialization     
        x_sample = np.array(x_initial_sample) #initialization
        y_sample = np.array(y_initial_sample)
        y_sample = y_sample.reshape(len(y_sample),1) #single output 

        #get the location of the initial sample points too and delete from total data
        sample_loc = []
        #convert self.x and self.y to list
        self.x_data = self.x_data.tolist()
        self.y_data = self.y_data.tolist()
        for data_loc in range(x_sample.shape[0]): #get the location
            try:
                x_data_loc = x_sample[data_loc]

                # new_point = self.closest(self.x_scaling_model.transform(self.x_data),self.x_scaling_model.transform(x_data_loc))
                x_data_loc_norm = self.x_scaling_model.transform((np.array(x_data_loc)).reshape(1,self.dim))
                new_point = self.closest(self.x_data_norm,x_data_loc_norm)
                sample_pos = self.x_data.index(new_point[0].tolist())

            except ValueError:
                print("List does not contain value")
            sample_loc.append(sample_pos) 
            del self.x_data[sample_pos]
            del self.y_data[sample_pos]
            del self.x_data_norm[sample_pos]
 
        # y_sample = y_sample.reshape(len(y_sample),1) #single output 

        x_sample = x_initial_sample
        y_sample = y_initial_sample

        #initialize the list of f_min 
        # maximum_improvement_list = (np.linspace(1,4,4)).tolist()#improvement_list = [1.0,2.0,3.0,4.0] 
        maximum_improvement_list = [24,0.0001,1000,1] #random numbers with large variance
        theta0 = [0.01]*self.dim
        count = 0 #initializing count
        self.y_min = min(y_sample) #initialization
        perc = [] #just for plotting
        self.improvement = True # initialized
        while(count<self.max_steps): #stopping criteria
            # main task: get new points
            # build surrogate model
            y_sample = np.array(y_sample)
            self.gp_old = KCORE2.Kriging(x_sample,y_sample,self.kernel,theta0=theta0,optimizer="nelder-mead-c") #initialiize model
            self.gp_old.train() # train model
            #using the value of theta in last training as the initial value to help with faster convergence
            # theta0 = self.gp_old.theta #update the value of theta0 
            theta0 = []
            for i in range(self.dim):
                theta0.append(np.random.uniform(1e-3,100)) #see if randomizing theta in each iteration helps

            # optimize using manual search to get new point
            #choose the starting point randomly within the bound domain
            starting_point = self.get_new_starting_points()

            # put a while loop here to ensure that there is no repetition
            # if minimal point is 0.0 or -0.0, choose new starting position
            expected_improvement = 0.0
            ei_array = (np.linspace(0,5,5)).tolist() #dummy data
            while (expected_improvement == 0.0):
                # optimize using manual search to get new point
                # new_point_loc = np.argmin(self.loss_func(self.x_data,self.y_data))
                # x_new = np.array(self.x_data[new_point_loc])
                # y_new = np.array(self.y_data[new_point_loc])
                res = cNM.constrNM(self.loss_func,starting_point,self.LB,self.UB,full_output=True)
                
                # neg_acq_func = res['fopt']
                expected_improvement = self.compute_EI(res['xopt'])
                ei_array = push_and_pop(expected_improvement[0],ei_array)
                starting_point = self.get_new_starting_points()
                if(np.mean(ei_array)==0): 
                    # set stopping criteria to be True
                    self.improvement = False
                    break

            if (self.improvement == False): break
            # maximum_improvement_list = push_and_pop(maximum_improvement,maximum_improvement_list) #add the new minima to the list
            x_opt = (res['xopt']).reshape(1,self.dim)
            x_new = self.closest(self.x_scaling_model.transform(self.x_data),self.x_scaling_model.transform(x_opt)) #get the similar value of x in the real dataset
            pos = (self.x_data).index(x_new[0].tolist())       
            y_new = (self.y_data)[pos] # get the approximate value of y from the data
        
            # check if the new y is an outlier 
            if (check_outlier(y_new,y_sample,m=5)):continue #ensure that outliers don't fit

            y_sample = y_sample.reshape(len(y_sample),1)


            #add new point to existing sample if it is not an outlier

            x_sample = np.vstack((x_sample,x_new))
            y_sample = np.vstack((y_sample,y_new)) 

            # should we not be deleting the new points from the data progressively?
            new_point_loc = (self.x_data).index(x_new[0].tolist())
            del self.x_data[new_point_loc] #test line
            del self.y_data[new_point_loc] #test line 
            del self.x_data_norm[new_point_loc]
            # x_sample, y_sample = reject_outliers(x_sample,y_sample)

            # self.y_min = min(y_sample) #update
            count +=1


            # attempting to compute the KL_divergence
            # compute gp_new model
            # nelder-mead-c
            self.gp_new = KCORE2.Kriging(x_sample,y_sample,self.kernel,theta0=theta0,optimizer="nelder-mead-c") #initialiize model
            self.gp_new.train() # train new model
            difference = compute_percentage_difference(self.gp_old.likelihood,self.gp_new.likelihood)
            perc.append(difference)
            # print("Old likelihood: {0:3.2f}, New likelihood: {1:3.2f}. The percentage difference: {2:.3f}" \
                # .format(self.gp_old.likelihood,self.gp_new.likelihood, difference))
            maximum_improvement_list = push_and_pop(difference,maximum_improvement_list)
            # predict
            # y_pred = self.gp_new.predict(x_sample)
            # y_var = self.gp_new.predict_variance(x_sample)
            # y_cov = self.gp_new.cov_y
            # pos_new = [y_pred[0],y_cov]
            # prior = [np.zeros((y_pred[0].shape[0],1)),self.gp_new.R_xx]
            # # compute KL_prior
            # KL_prior = calcKL(pos_new, prior)

            # N = len(y_sample) - 1 # placed on pos_old
            # gp_new = GaussianProcessRegressor(kernel=kernel, alpha=0.0, optimizer=None).fit(new_input, new_output) # retrain gp to get new priors
            # K = kernel(x_sample, x_sample)
            # prior = [np.zeros(K.shape[0]), K]
            # pos_new = gp_new.predict(x_sample, return_cov=True)
            # KL_prior = calcKL(pos_new, prior)
            # E_Qnew = (((new_output - pos_new[0]) ** 2).sum() + np.trace(pos_new[1])) / (2 * Var * N) - (1 / 2) * np.log(
            #     2 * np.pi * Var)

            # if(np.var(maximum_improvement_list)<=self.stopping_criteria):break #convergence criteria
            # if(compute_norm_var(maximum_improvement_list)<=self.stopping_criteria):break #convergence criteria\
            # if(max(scaled_model_variance) <= self.stopping_criteria):break #convergence criteria\
            if (check_criteria(maximum_improvement_list,self.stopping_criteria)): 
                print(maximum_improvement_list)
                # delete the extra point
                # data = [x_sample,y_sample] #kenny debug line
                # x_sample,y_sample = remove_extra_entries(data,len(maximum_improvement_list)-1) #kenny debug line
                print(perc)
                break
        self.x_test = self.x_data
        self.y_test = self.y_data
        print('Total training points: {0}'.format(x_sample.shape[0]))
        return x_sample,y_sample
    
    def get_new_starting_points(self):
        starting_point = []
        for j in range(len(self.bounds)):
            starting_point.append(np.random.uniform(self.bounds[j][0],self.bounds[j][1]))
        return starting_point


        
def push_and_pop(value,current_list):
    #delete the first member of the list
    del(current_list[0])
    #add a new member to the list
    current_list.append(value)
    return current_list


def get_pooled_data(size,dim,bounds):
    norm_data = pyd.lhs(dim,size,random_state=42)
    x_pool = scaler.scale(norm_data,bounds)
    return x_pool

def compute_norm_var(numbers):
    #compute the variance of a normalize set of numbers. It is easier to compare different sets that way
    numbers = np.array(numbers)
    numbers = numbers[:,np.newaxis]
    msn = MS()
    norm_numbers = msn.fit_transform(numbers)
    norm_var = np.std(norm_numbers)
    return norm_var

def calcKL(pos1, pos2):
    epsilon = 0.01
    N = pos1[0].shape[0]
    f2 = pos2[0]
    f1 = pos1[0]
    S2 = pos2[1] + epsilon * np.eye(N)
    S1 = pos1[1] + epsilon * np.eye(N)
    S2_inv = np.linalg.inv(S2)
    S = S2_inv @ S1
    trace = np.trace(S) # sum of the variance
    logdet = np.log(np.linalg.det(S))
    se = (f2 - f1).T @ S2_inv @ (f2 - f1)
    KL = 0.5 * (trace - logdet + se - N)
    return KL

def calcPAC(E_Qnew,KL_prior,delta,N,range):

    PAC = E_Qnew + KL_prior / N - np.log(delta) / N + 0.5 * (range[1] - range[0]) ** 2
    return PAC

def compute_percentage_difference(old,new):
    diff = abs(new-old)
    diff /= old
    return diff

def check_criteria(numbers,limit_percentage):
    # condition 1: the last set of improvements must be sorted in descending order
    # condition 2: The improvements must be less than the set stopping limit
    status = False
    if (numbers == sorted(numbers,reverse=True)):
        if (all(i < limit_percentage for i in numbers)):
            status = True
        else:
            status = False
    else:
        status = False
    return status

# write code to get the bounds
# bounds = [[min,max],[min,max]]
def get_bounds(data):
    bounds = []
    for i in range(data.shape[1]):
        bounds.append([min(data[:,i]),max(data[:,i])])    
    return bounds

def remove_extra_entries(data,counts):
    # get x and y data and convert to list
    x = (data[0]).tolist()
    y = (data[1]).tolist()
    for count in range(counts):
        x.pop()
        y.pop()
    # convert x and y back to arrays
    x = np.asarray(x)
    y = np.asarray(y)
    return x,y

def check_outlier(num,data,m=3):
    d_num = np.abs(num-np.median(data))
    mdev = np.median(np.abs(data-np.median(data)))
    s = d_num/mdev
    if s > m:
        status = True #the number is an outlier
    else:
        status = False
    return status

def reject_outliers(x_data,y_data,m=2):
    #the outlier is selected based on the response
    d = np.abs(y_data-np.median(y_data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0
    filtered_pos = np.where(s<m)
    print(len(filtered_pos[0]))
    x_filtered_data = []
    y_filtered_data = []
    for index in range(len(filtered_pos[0])):
        x_filtered_data.append(x_data[filtered_pos[0][index]])
        y_filtered_data.append(y_data[filtered_pos[0][index]])
    return np.array(x_filtered_data),np.array(y_filtered_data)


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