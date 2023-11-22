import numpy as np 
import scipy as sc 

def closest(points, x):
    return sorted(points, key=lambda p: np.linalg.norm(p-x))[:1] # 1 for top one
    
class map_data():
    '''
    Get the nearest points from a dataset using some other data.
    Distance metric is euclidean

    '''
    def __init__(self,sample_data,whole_data,distance_metric="euclidean"):
        self.sample_data = sample_data # get the sample data
        self.whole_data = whole_data # get the whole data
        '''
        The whole and sample data must be in the form Ns X Ndv
        where Ns is the sample size
        and Ndv is the dimension of the data
        It is important to check and ensure the dimension...
        of both data are same

        By default, the euclidean distance is used however,
        ...make code flexible by adding more distance metrics
        Manhattan distance is said to be best for high dimensions

        Manhattan distance uses the L1 - norm
        and Euclidean distance uses the L2 - norm
        '''

        self.sample_size = (self.sample_data).shape[0] 
        self.sample_dim = (self.sample_data).shape[1]
        self.whole_size = (self.whole_data).shape[0]
        self.whole_dim = (self.whole_data).shape[1]
        self.distance_metric = distance_metric
        if(self.sample_dim != self.whole_dim):
            print("Data mismatch, please revise!")
            pass


    def create_sample(self):
        self.sampled_data = []
        self.pos = []
        whole_data = np.copy(self.whole_data)
        for i in range(self.sample_size):
            #get new point
            #get the position
            #delete the point without adjusting its index
            whole_data = np.array(whole_data)
            self.whole_data = np.array(self.whole_data)
            new_point = closest(whole_data,self.sample_data[i])
            new_point = (new_point[0]).tolist() #converts the newpoint to a list for better handling
            self.sampled_data.append(new_point) # add the new point to the sampled data set
            whole_data = whole_data.tolist()
            self.whole_data = self.whole_data.tolist()
            '''
            To prevent the repeating data points, once a point is selected,
            it is deleted from the whole data
            The program will get the position of the data from the whole data...
            and append to the pos - list

            '''
            (self.pos).append((self.whole_data).index(new_point)) 
            del(whole_data[(whole_data).index(new_point)])
        return self.sampled_data
    
'''       

# unit test for code 
whole_data = [[2,3],[9,1],[4,2],[0,3],[9,9]]
sample = [[8,1],[1,3],[4,1],[6,6]]
whole_data = np.array(whole_data)
sample = np.array(sample)
m = map_data(sample,whole_data)
l = m.create_sample()
print(l)
print(m.pos)

'''