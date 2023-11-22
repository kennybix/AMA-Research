import numpy as np

def normalize(x):
    y = (x-min(x))/(max(x)-min(x))
    return y

# def denormalize(x,ymin,ymax):
#     if(x.shape[1]==1):
#         n = len(x)
#         y = np.ones(n)
#         for i in range(n):
#             y[i] = (x[i]*(ymax-ymin))+ymin
#     else:
#         k = x.shape[1]
#         n = x.shape[0]
#         y = np.zeros((n,k))
#         for 
#     return y

def normalize_values(x,xmin,xmax):
    y = (x-xmin)/(xmax-xmin)
    return y


def denormalize(x,ymin,ymax):
    n = len(x)
    y = np.ones(n)
    for i in range(n):
        y[i] = (x[i]*(ymax-ymin))+ymin
    return y

def standardize(X,y):
    '''
    Returns 
    1.X_normal, 2.y_normal
    3.X_mean, 4.y_mean
    5.X_std, 6.y_std

    X is the multidimensional input
        y is the output
    Method returns
                1.the normalized values
                2. the mean
                3. the standard deviation
   of the parameters 
    '''
    X = np.array(X)

    n = X.shape[0]
    k = X.shape[1] #this returns the dimension of X
    X_normal = np.zeros((n,k))
    X_mean = np.zeros(k)
    X_std = np.zeros(k)
    y_normal = np.zeros(n) #initialization
    y_mean = 1
    y_std = 1
    
    for  i in range(k):
        X_normal[:,i] = normalize(X[:,i])
        X_mean[i] = np.sum(X[:,i]) / n
        X_std[i] = np.std(X[:,i])

    y_normal = normalize(y)
    y_mean = np.mean(y)
    y_std = np.std(y)
    return X_normal,y_normal,X_mean,y_mean,X_std,y_std

def norm(X,y):
    '''
    Returns 
    1.X_normal, 2.y_normal
    3.X_min, 4.y_min
    5.X_max, 6.y_max

    X is the multidimensional input
        y is the output
    Method returns
                1.the normalized values
                2. the mean
                3. the standard deviation
   of the parameters 
    '''
    X = np.array(X)

    n = X.shape[0]
    k = X.shape[1] #this returns the dimension of X
    X_normal = np.zeros((n,k))
    X_min = np.zeros(k)
    X_max = np.zeros(k)
    y_normal = np.zeros(n) #initialization
    y_min = 1
    y_max = 1
    
    for  i in range(k):
        X_normal[:,i] = normalize(X[:,i])
        X_min[i] = np.min(X[:,i])
        X_max[i] = np.max(X[:,i])

    y_normal = normalize(y)
    y_min = np.min(y)
    y_max = np.max(y)
    return X_normal,y_normal,X_min,y_min,X_max,y_max