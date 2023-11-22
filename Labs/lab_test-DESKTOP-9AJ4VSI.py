# import the libraries
import numpy as np
import pandas as pd 
from active_learning import adsp_r as AL

# data = pd.read_excel("Concrete_Data.xls")
data = np.loadtxt("concrete_data.txt")
# print(data.head(5))
x_data = data[:,:-1]
y_data = data[:,-1]

whole_data = {"x_data":x_data, "y_data":y_data}



model = AL.active_learn(whole_data,150,0.05,1000)

x,y = model.get_more_samples()

print(x.shape)

np.savetxt("optimal_x.csv",x,delimiter=",")
np.savetxt("optimal_y.csv",y,delimiter=",")

np.savetxt("x_test.csv",model.x_test,delimiter=",")
np.savetxt("y_test.csv",model.y_test,delimiter=",")