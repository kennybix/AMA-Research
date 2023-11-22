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
The stopping criteria here is the variance of the maximum improvement


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

class active_learn():
    def __init__(self,obj_func,bounds,initial_ns,stopping_criteria=0.3,max_steps=1000,DOE="LHS",kernel="gaussian",tolerance=1e-4):

        self.obj_func = obj_func
        self.stopping_criteria = stopping_criteria
        self.bounds = bounds
        self.LB = []
        self.UB = []
        for i in range(len(self.bounds)):
            self.LB.append(self.bounds[i][0])
            self.UB.append(self.bounds[i][1])
        self.initial_guess = []
        for j in range(len(self.bounds)):
            self.initial_guess.append(np.random.uniform(self.bounds[j][0],self.bounds[j][1]))
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

    def compute_EI(self,x):
        x = x.reshape(1,self.dim)
        fhat,s2 = self.gp_old.predict(x) #modify predict to return two values just for this process
        fhat = fhat.reshape(len(s2),)
        s2 = s2.reshape(len(s2),)
        y = self.obj_func(x)
        if isinstance(y,list):
            y = np.array(y)
        else:
            y = y
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
        fhat,s2 = self.gp_old.predict(x) #modify predict to return two values just for this process
        fhat = fhat.reshape(len(s2),)
        s2 = s2.reshape(len(s2),)
        y = self.obj_func(x)
        if isinstance(y,list):
            y = np.array(y)
        else:
            y = y
        #have different formulations to compute the loss function, so I can select 
        # EI inclusive
        # with np.errstate(divide='warn'):
        #     EI = (self.y_min - fhat) * (0.5 + 0.5*erf((self.y_min - fhat)/np.sqrt(2 * s2))) + \
        #             np.sqrt(0.5*s2/np.pi)*np.exp(-0.5*(self.y_min - fhat)**2/s2)
        #     EI[s2 == 0.0] = 0.0 #no improvement is expected at sample points
        L = 0.5*(y-fhat)**2 + 0.5*s2 #modified for maximization
        # L = EI
        # L = abs(fhat)/s2
        return -L
    def get_more_samples(self):
        
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
        # maximum_improvement_list = (np.linspace(1,4,4)).tolist()#improvement_list = [1.0,2.0,3.0,4.0] 
        maximum_improvement_list = [24,0.0001,1000,1] #random numbers with large variance
        theta0 = [0.01]*self.dim
        count = 0 #initializing count
        starting_point = self.initial_guess
        self.y_min = min(y_sample) #initialization
        self.bounds = np.array(self.bounds)

        perc = [] #just for plotting
        self.improvement = True # initialized
        while(count<self.max_steps): #stopping criteria
            # main task: get new points
            # build surrogate model
            self.gp_old = KCORE2.Kriging(x_sample,y_sample,self.kernel,theta0=theta0,optimizer="nelder-mead-c") #initialiize model
            self.gp_old.train() # train model
            # print(self.gp_old.likelihood)
            # pooled_data = get_pooled_data(20,self.dim,self.bounds) # get 5 pooled data
            # it seems keeping the pooled data constant is important to the convergence of the active learning model
            # model_prediction = (self.gp_old.predict(pooled_data))[0]
            # model_variance = self.gp_old.predict_variance(pooled_data) #compute model variance
            # scaled_model_variance = self.gp_old.y_scaler.transform(model_variance)
            #using the value of theta in last training as the initial value to help with faster convergence
            # theta0 = self.gp_old.theta #update the value of theta0 
            theta0 = []
            for i in range(self.dim):
                theta0.append(np.random.uniform(1e-2,5)) #see if randomizing theta in each iteration helps

            # optimize using manual search to get new point
            #choose the starting point randomly within the bound domain
            starting_point = self.get_new_starting_points()
            # starting_point = (np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(1, self.dim)))[0]
            # put a while loop here to ensure that there is no repetition
            # if minimal point is 0.0 or -0.0, choose new starting position
            expected_improvement = 0.0
            ei_array = (np.linspace(0,30,30)).tolist() #dummy data
            while (expected_improvement == 0.0):
                res = cNM.constrNM(self.loss_func,starting_point,self.LB,self.UB,full_output=True)
                # neg_acq_func = res['fopt']
                expected_improvement = self.compute_EI(res['xopt'])
                ei_array = push_and_pop(expected_improvement[0],ei_array)
                # res1 = minimize(self.loss_func,starting_point,method='L-BFGS-B',bounds=self.bounds)
                # res = {}
                # res['xopt'] = res1.x 
                # neg_acq_func = res1.fun
                # model = PSO.PSO(self.loss_func,starting_point,self.bounds,num_particles=100,maxiter=30)
                # res = {} #empty dictionary
                # res['xopt'] = model.optimize()
                # neg_acq_func = model.func
                starting_point = self.get_new_starting_points()
                if(np.mean(ei_array)==0): 
                    # set stopping criteria to be True
                    self.improvement = False
                    break

            if (self.improvement == False): break
            # maximum_improvement = -(res['fopt']) #update maximum improvement
            x_new = res['xopt']
            # starting_point = x_new # commenting out to facilitate debugging
            x_new = x_new.reshape(1,self.dim)
            y_new = self.obj_func(x_new)
            # maximum_improvement_list = push_and_pop(maximum_improvement,maximum_improvement_list) #add the new minima to the list
            y_new = y_new.reshape(1,1)
            # check if the new y is an outlier 
            # if (check_outlier(y_new,y_sample,m=5)):continue #ensure that outliers don't fit
            # important note --- maybe the outlier detection is more necessary for real world data

            y_sample = y_sample.reshape(len(y_sample),1)


            #add new point to existing sample if it is not an outlier

            x_sample = np.vstack((x_sample,x_new))
            y_sample = np.vstack((y_sample,y_new))  

            # x_sample, y_sample = reject_outliers(x_sample,y_sample)

            # self.y_min = min(y_sample) #update
            count +=1


            # attempting to compute the KL_divergence
            # compute gp_new model
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
                data = [x_sample,y_sample]
                x_sample,y_sample = remove_extra_entries(data,len(maximum_improvement_list)-1)
                print(perc)
                break
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