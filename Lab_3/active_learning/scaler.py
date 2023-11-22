from sklearn.preprocessing import MinMaxScaler as MS
import numpy as np

def scale(x,limit):
# '''
# Function that scales data using given limits
# '''
    try:
        dimension = x.shape[1]
    except:
        x = np.array(x)
        dimension = x.shape[1]
    
    x_new = np.zeros((x.shape[0],x.shape[1])) # initialize the new x
    for i in range(dimension):
        scaler = MS()
        xi = (x[:,i]).reshape(x.shape[0],1)
        xi = scaler.fit_transform(xi) # forcefully normalize x
        a,b = get_parameters(limit[i][0],limit[i][1])
        scaler.min_ = a
        scaler.scale_ = b
        x_new[:,i] = (scaler.inverse_transform(xi)).reshape(x.shape[0],)
    return x_new

def get_parameters(x_min,x_max):
    scale_ = 1/(x_max - x_min)
    min_ = -(x_min * scale_)
    return min_,scale_
