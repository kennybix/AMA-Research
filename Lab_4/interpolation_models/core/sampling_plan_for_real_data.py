import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
import map_sampling_plan as mps 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import halton_sequence as hs
import lhs 


def get_data(index,data_array):
    m = len(index)
    data_array = np.array(data_array)
    data = []
    for i in range(m):
        data.append(data_array[int(index[i])])
    return data

class Sampling_plan():
    def __init__(self,x,y=[],Ns=1,sequence='halton'):
        self.x = x
        self.y = y 
        self.Ns = Ns
        self.Nk = (self.x).shape[1] # get the dimension of the data
        self.sequence = sequence

    def create_samples(self):
        # Normalize data in preparation for mapping
        x_scale = MinMaxScaler()
        x_scale.fit(self.x)
        self.x_norm = x_scale.transform(self.x)
        # Generate points
        if self.sequence == 'halton':
            x_hs = hs.halton_sequence(self.Ns,self.Nk)
        elif self.sequence == 'lhs':
            x_hs = lhs.sample(self.Nk,self.Ns) #not tested
        x_hs = x_hs.T
        plan_model = mps.map_data(x_hs,self.x_norm)
        x_plan_norm = plan_model.create_sample()
        x_plan_norm = np.array(x_plan_norm)
        # plt.scatter(x_plan_norm[:,0],x_plan_norm[:,1])
        # plt.show()
        x_plan = x_scale.inverse_transform(x_plan_norm)
        x_sample = x_plan
        y_sample = get_data(plan_model.pos,self.y)

        return x_sample,y_sample



# First test 
'''
Designed to see the behaviour of the sampling plans to 
ensure consistency

Test is carried out on a 2D data
'''
'''
x = hs.halton_sequence(50,2)
x = x.T
plt.figure(3,figsize=(20,18))
plt.scatter(x[:,0],x[:,1],color='blue')
plt.tight_layout()
plt.show()
'''

'''
Test status: successful
data = pd.read_excel(r'../data/single_2D.xlsx')
x = data.drop('cd',axis=1)
y = data['cd']

whole_data = np.array(x)
plan = Sampling_plan(x,y,Ns=100)
x_sample,y_sample = plan.create_samples()
plt.scatter(whole_data[:,0],whole_data[:,1],color='red',alpha=0.5)
plt.scatter(x_sample[:,0],x_sample[:,1],color='green')


plt.tight_layout()
plt.show()
'''