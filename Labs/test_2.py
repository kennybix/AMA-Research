# import the libraries
import math
import numpy as np
import pandas as pd 
from active_learning import adsp as AL

def branin_2D(x):
    x1 = x[:,0]
    x2 = x[:,1]
    PI = math.pi
    t = (1/(8*PI))
    t = float(t)
    b = 5.1/(4*math.pow(PI,2))
    b = float(b)
    return (np.power((x2 - (np.power(x1,2) * b) + (5*x1/PI) - 6),2) + 10*(1 - t)*np.cos(x1) + 10)

# limits = [[-5,10],[0,15]]
# model = AL.active_learn(branin_2D,limits,30,0.05,1000)
# x,f = model.get_more_samples()

from smt.surrogate_models import KRG as SMT_KRG
import pandas as pd

x_train = pd.read_csv("optimal_x.csv")
y_train = pd.read_csv("optimal_y.csv")
x_test = pd.read_csv("x_test.csv")
y_test = pd.read_csv("y_test.csv")

smt_model = SMT_KRG(corr='squar_exp')
smt_model.set_training_values(np.array(x_train),np.array(y_train))
smt_model.train()
smt_model_y_test = smt_model.predict_values(np.array(x_test))