# import necessary libraries

import numpy as np 
from scipy.optimize import minimize
from sampling_algorithm import kriging_ego as KCORE
from sampling_algorithm import kriging_ego_2 as KCORE2
from sampling_algorithm import kriging_ego_3 as KCORE3
from sampling_algorithm import map_sampling_plan as mps 
from sampling_algorithm import sampling_plan_for_real_data as sprd

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

class Adaptive_Sampling():
    def __init__(self,x_data,y_data,initial_ns,final_ns,DOE="halton",kernel="gaussian"):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.dim = (self.x_data).shape[1]
        if (initial_ns > (self.x_data).shape[0]): #hack to ensure the initial sample size is lesser than the data size
            self.initial_ns = initial_ns / 1.5
        else:
            self.initial_ns = initial_ns
        self.final_ns = final_ns
        self.DOE = DOE # halton or lhs
        self.kernel = kernel

    def loss_func(self,x,y):
        #hybrid optimization because we just search but an efficient search
        # we have to always update the model and refresh
        # x = x.reshape(1,self.dim)
        fhat,s2 = self.model.predict(x) #modify predict to return two values just for this process
        fhat = fhat.reshape(len(s2),)
        s2 = s2.reshape(len(s2),)
        if isinstance(y,list):
            y = np.array(y)
        #have different formulations to compute the loss function, so I can select 
        # EI inclusive
        # EI = (self.y_best - y_hat) * (0.5 + 0.5*erf((self.y_best - y_hat)/np.sqrt(2 * SSqr))) + \
        #             np.sqrt(0.5*SSqr/np.pi)*np.exp(-0.5*(self.y_best - y_hat)**2/SSqr)
        L = 0.5*(y-fhat)**2 + 0.5*s2 #modified for maximization
        return L

    def adapt_samples(self):
        # generate initial samples, we can even do that using our previous code
        # if (self.DOE=="LHS"):
        #     x_initial_sample = lhs.sample(self.dim,self.initial_ns)
        # elif (self.DOE=="HS"):
        #     x_initial_sample = hs.halton_sequence(self.initial_ns,self.dim)
        # # get the corresponding f values       
        # y_initial_sample = self.obj_func(x_initial_sample)

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
                # x_data_loc = np.round(x_data_loc.astype(np.float64),5) #dealing with floating point inconsistencies
                '''
                    The floating point inconsistencies arises when converting numpy array to list
                    you might get 0.444449 for a value of 0.45
                    this behaviour becomes problematic to the code
                '''
                new_point = closest(self.x_data,x_data_loc.tolist())
                sample_pos = self.x_data.index(new_point[0].tolist())

            except ValueError:
                print("List does not contain value")
            sample_loc.append(sample_pos) 
            del self.x_data[sample_pos]
            del self.y_data[sample_pos]
        
        theta0 = [] #empty list
        for i in range(x_sample.shape[1]):
            theta0.append(np.random.uniform(1e-2,5)) # guess the starting point at random

        while(count<self.final_ns): #stopping criteria
            # main task: get new points
            # build surrogate model
            # self.model = KCORE.Kriging(x_sample,y_sample,self.kernel,[0.5]*self.dim,"nelder-mead-c") #initialiize model
            # self.model = KCORE3.Kriging(x_sample,y_sample,self.kernel,theta0=theta0,optimizer="nelder-mead-c") #initialiize model
            self.model = KCORE2.Kriging(x_sample,y_sample,self.kernel,theta0=theta0,optimizer="nelder-mead-c") #initialiize model
            self.model.train() # train model
            #using the value of theta in last training as the initial value to help with faster convergence
            theta0 = self.model.theta #update the value of theta0 

            # optimize using manual search to get new point
            new_point_loc = np.argmax(self.loss_func(self.x_data,self.y_data))
            x_new = np.array(self.x_data[new_point_loc])
            y_new = np.array(self.y_data[new_point_loc])
            #add new point to existing sample
            x_sample = np.vstack((x_sample,x_new.reshape(1,self.dim)))
            y_sample = np.vstack((y_sample,y_new.reshape(1,1)))

            # should we not be deleting the new points from the data progressively?
            del self.x_data[new_point_loc] #test line
            del self.y_data[new_point_loc] #test line
            
            count +=1

        # get the remaining data as testset with self
        # because we would have deleted all the training points
        self.x_test = self.x_data
        self.y_test = self.y_data
        return x_sample,y_sample
    
