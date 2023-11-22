# import necessary libraries

import numpy as np 
from scipy.optimize import minimize
from interpolation_models.core import kriging_ego as KCORE
from interpolation_models.core import lhs
from interpolation_models.core import halton_sequence as hs
import map_sampling_plan as mps 
import sampling_plan_for_real_data as sprd
'''
Input:
        1. Objective function
        2. Starting point counts
        3. Stopping criteria or final counts
        4. Number of validation points

Output:
        1. Imporved Kriging model
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


class Adaptive_Sampling():
    def __init__(self,x_data,y_data,initial_ns,final_ns,DOE="LHS",kernel="gaussian"):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.dim = (self.x_data).shape[1]
        if (initial_ns > (self.x_data).shape[0]): #hack to ensure the initial sample size is lesser than the data size
            self.initial_ns = initial_ns / 1.5
        else:
            self.initial_ns = initial_ns
        self.final_ns = final_ns
        self.DOE = DOE
        self.kernel = kernel

    def loss_func(self,x):
        #hybrid optimization because we just search but an efficient search
        # we have to always update the model and refresh
        '''
        get the nearest x from the data using the euclidean distance
        also get the corresponding y from the data

        '''
        x = x.reshape(1,self.dim)
        plan_model = mps.map_data(x,self.x_data)
        nearest_x = plan_model.create_sample() #get the point
        nearest_x = np.array(nearest_x) #convert to array
        nearest_y = get_data(plan_model.pos,self.y_data)

        fhat,s2 = self.model.predict(nearest_x) #modify predict to return two values just for this process
        L = 0.5*(nearest_y-fhat) + 0.5*s2
        return L

    def adapt_samples(self):
        # generate initial samples, we can even do that using our previous code
        # if (self.DOE=="LHS"):
        #     x_initial_sample = lhs.sample(self.dim,self.initial_ns)
        # elif (self.DOE=="HS"):
        #     x_initial_sample = hs.halton_sequence(self.initial_ns,self.dim)
        # # get the corresponding f values       
        # y_initial_sample = self.obj_func(x_initial_sample)

        sprd_model = sprd.Sampling_plan(self.x_data,self.y_data,Ns=self.initial_ns) #get the initial sample
        x_initial_sample,y_initial_sample = sprd_model.create_samples()
        count = self.initial_ns #initialization     
        x = np.array(x_initial_sample) #initialization
        y = np.array(y_initial_sample)
        while(count<self.final_ns): #stopping criteria
            # main task: get new points
            # build surrogate model
            self.model = KCORE.Kriging(x,y,self.kernel,[0.5]*self.dim,"nelder-mead-c") #initialiize model
            self.model.train() # train model
            # optimize using SLSQP to get new point
            res = minimize(self.loss_func,np.ones((self.dim,)),method="nelder-mead")
            new_x = 1.0



    # def get_optimal_samples(self):

    #     return x_optimal, y_optimal

    # def get_optimal_kriging_model(self):


    #     return model

    

# model test
import pandas as pd 

dataframe = pd.read_csv("yatch_data.csv")
x = dataframe.drop(['y'],axis=1) #efficient way to get the input data
y = dataframe['y']

adsp = Adaptive_Sampling(x,y,initial_ns=20,final_ns=25)
adsp.adapt_samples()