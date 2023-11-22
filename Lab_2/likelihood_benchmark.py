# algebraic problem benchmark
# TestLab for likelihood studies
# We will explore the influence of hyperparameter values on the likelihood estimate
# We will also explore the influence of the hyperparameter values on model accuracy
# Lastly, we will investigate the relationship between likelihood estimate and model accuracy

# latest copy as at 30-Jan-2022

# import necessary libraries and models
from random import random
from numpy import * 
import numpy as np 
import sys
from pandas import * 
import pyDOE2 as pyd 
import time 
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import shuffle
from matplotlib import pyplot as plt #might not use this now 
from matplotlib import rc
rc('font',**{'family':'serif'})
rc('text', usetex=True)

import math
from numpy.random import sample 
from interpolation_models.core import scaler as SC
# import inspect
# from active_learning import adsp

# my libraries
from interpolation_models.core import scaler 
from interpolation_models.core import kriging as KRG
from smt.surrogate_models import KRG as SMT_KRG
# from interpolation_models import ensemble as KEM, CKL as KCKL, MCKL as KMCKL, MIKL as KMIKL, MIKLP as KMIKL_PLS
# from interpolation_models.core_free_form import kriging_improved as KVI, kriging_vni as KVU, kriging_vnu_pls as KVU_PLS
# from interpolation_models.core_free_form import kriging_improved_smart as KVI_s, kriging_vni_smart as KVU_s, kriging_vnu_pls_smart as KVU_PLS_s
 
from interpolation_models.core import Benchmark_Problems as BP


def savedata(x_train,y_train,x_test,y_test,data_name):
    # for data in range(len(y_train)):
    savetxt("learned_data/"+data_name+"_train_x"+".csv",x_train,delimiter=",")
    savetxt("learned_data/"+data_name+"_train_y"+".csv",y_train,delimiter=",")
    savetxt("learned_data/"+data_name+"_test_x"+".csv",x_test,delimiter=",")
    savetxt("learned_data/"+data_name+"_test_y"+".csv",y_test,delimiter=",")

    return 0

def getdata(data_name):
    # get the data
    # declare an empty list and fill it up
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    x_train.append(read_csv("learned_data/"+data_name+"_train_x"+".csv",header=None))
    x_test.append(read_csv("learned_data/"+data_name+"_test_x"+".csv",header=None))
    y_train.append(read_csv("learned_data/"+data_name+"_train_y"+".csv",header=None))
    y_test.append(read_csv("learned_data/"+data_name+"_test_y"+".csv",header=None))

    # #convert to array for better data handling
    x_train = (np.array(x_train))[0]
    x_test = (np.array(x_test))[0]
    y_train = (np.array(y_train))[0]
    y_test = (np.array(y_test))[0]
    return x_train,y_train,x_test,y_test

def save_likelihood_data(Z,info):
    # Z is a list of lists
    # convert to array for easy handling
    Z = np.array(Z)
    savetxt("likelihood_data/"+info['name']+"_"+str(info['sample_size'])+"_"+str(info['resolution'])+"_g"+".csv",Z[0],delimiter=",")
    savetxt("likelihood_data/"+info['name']+"_"+str(info['sample_size'])+"_"+str(info['resolution'])+"_e"+".csv",Z[1],delimiter=",")
    savetxt("likelihood_data/"+info['name']+"_"+str(info['sample_size'])+"_"+str(info['resolution'])+"_m3"+".csv",Z[2],delimiter=",")
    savetxt("likelihood_data/"+info['name']+"_"+str(info['sample_size'])+"_"+str(info['resolution'])+"_m5"+".csv",Z[3],delimiter=",")
    return 0


def get_likelihood_data(info):

    G = read_csv("likelihood_data/"+info['name']+"_"+str(info['sample_size'])+"_"+str(info['resolution'])+"_g"+".csv",header=None)
    E = read_csv("likelihood_data/"+info['name']+"_"+str(info['sample_size'])+"_"+str(info['resolution'])+"_e"+".csv",header=None)
    M3 = read_csv("likelihood_data/"+info['name']+"_"+str(info['sample_size'])+"_"+str(info['resolution'])+"_m3"+".csv",header=None)
    M5 = read_csv("likelihood_data/"+info['name']+"_"+str(info['sample_size'])+"_"+str(info['resolution'])+"_m5"+".csv",header=None)

    return G,E,M3,M5


def save_accuracy_data(M,info):
    # M is a list of lists
    # convert to array for easy handling
    M = np.array(M)
    savetxt("accuracy_data/"+info['name']+"_"+str(info['sample_size'])+"_"+str(info['resolution'])+"_g"+".csv",M[0],delimiter=",")
    savetxt("accuracy_data/"+info['name']+"_"+str(info['sample_size'])+"_"+str(info['resolution'])+"_e"+".csv",M[1],delimiter=",")
    savetxt("accuracy_data/"+info['name']+"_"+str(info['sample_size'])+"_"+str(info['resolution'])+"_m3"+".csv",M[2],delimiter=",")
    savetxt("accuracy_data/"+info['name']+"_"+str(info['sample_size'])+"_"+str(info['resolution'])+"_m5"+".csv",M[3],delimiter=",")

    return 0 


def get_accuracy_data(info):
    G = read_csv("accuracy_data/"+info['name']+"_"+str(info['sample_size'])+"_"+str(info['resolution'])+"_g"+".csv",header=None)
    E = read_csv("accuracy_data/"+info['name']+"_"+str(info['sample_size'])+"_"+str(info['resolution'])+"_e"+".csv",header=None)
    M3 = read_csv("accuracy_data/"+info['name']+"_"+str(info['sample_size'])+"_"+str(info['resolution'])+"_m3"+".csv",header=None)
    M5 = read_csv("accuracy_data/"+info['name']+"_"+str(info['sample_size'])+"_"+str(info['resolution'])+"_m5"+".csv",header=None)
    return G,E,M3,M5




def get_errors(model,y_pred,y_true):
    errors = []
    if (len(y_pred) == 1): # account for LOOCVE
        y_true = np.array(y_true)
        e = np.abs(y_pred - y_true)
        NRMSE = 100 * np.abs((y_true - y_pred)/y_true)
        errors = [NRMSE, e[0]]
    else:
        RMSE = 100 * model.computeNRMSE(y_true)
        MAE = mean_absolute_error(y_true,y_pred)
        R2 = r2_score(y_true,y_pred)
        MSE = mean_squared_error(y_true,y_pred)
        errors = [RMSE[0],MAE,R2,MSE]
    return errors

def get_smt_errors(y_pred,y_true):
    errors = []
    MAE = mean_absolute_error(y_true,y_pred)
    R2 = r2_score(y_true,y_pred)
    MSE = mean_squared_error(y_true,y_pred)
    errors = [MAE,R2,MSE]
    return errors


def get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG',kernels="", model_params={}):
    # eval('KRG') == KRG # smart way to select without having to use if-statements
    model = str_to_class(model).Kriging(x_train,y_train,kernels,optimizer=model_params['optimizer'],preprocessing=model_params['preprocessing'])
    start_time = time.time()
    model.train()
    elapsed_time = time.time() - start_time
    y = model.predict(x_test)
    y = y.reshape(len(y_test),1)
    model.y_output = y
    errors = get_errors(model,y,y_test)
    model_error = errors
    training_time = elapsed_time
    training_info = model.info
    return model_error , training_time , training_info

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)



# put all the necessary codes here

def model_likelihood_benchmark(data,params,name=""):

    '''
    data = {'training_data':train_data, 'test_data':test_data}
    params = {'theta_range':theta_range}
    '''

    kernels = ['gaussian','exponential','matern3_2','matern5_2'] #kernels available
    #write case for both 1D and 2D for now
    x_training_data = data['training_data'][0]
    y_training_data = data['training_data'][1]

    x_test_data = data['test_data'][0] 
    y_test_data = data['test_data'][1]

    Nk = x_training_data.shape[1] # get problem dimension
    Ns = x_training_data.shape[0] # get sample size

    Ts = x_test_data.shape[0]

    resolution = params['resolution'] #fixed for now # add to params after successful test
    theta_range = params['theta_range']
    theta_d = np.linspace(theta_range[0],theta_range[1],resolution)    
    
    # dictionary of information for accurate handling
    info = {'name':name,
            'theta_range':theta_range,
            'resolution': resolution,
            'sample_size': Ns,
            'test_size': Ts,
            'problem_dimension':Nk}

    Z = []

    if (Nk==1):
        for kernel_type in range(len(kernels)):
            model = KRG.Kriging(x_training_data,y_training_data,kernel=kernels[kernel_type],preprocessing="standardize")
            #initialize Z
            Z_single = np.zeros(resolution,resolution)
            for kk in range(resolution):
                thet = theta_d[kk]
                Z_single = model.NLL(thet)
            Z.append(Z_single) # append to the already prepared list    

    elif (Nk == 2):
        X_th,Y_th = np.meshgrid(theta_d,theta_d)

        for kernel_type in range(len(kernels)):
            model = KRG.Kriging(x_training_data,y_training_data,kernel=kernels[kernel_type],preprocessing="standardize")
            #initialize Z
            Z_single = np.zeros((resolution,resolution))
            for kk in range(resolution):
                for jj in range(resolution):
                    thet = [Y_th[kk][jj],X_th[kk][jj]]
                    thet = np.array(thet)
                    Z_single[kk][jj] = model.NLL(thet)
            Z.append(Z_single) # append to the already prepared list       
            
    else:
        a = 3 # filler lines

    # store data --- ease of reuse and data handling
    save_likelihood_data(Z,info)

    # method returns info 
    return info


def model_benchmark(data,params,name=""):

    '''
    data = {'training_data':train_data, 'test_data':test_data}
    params = {'theta_range':theta_range}
    '''
    output_data = {'name':name}
    kernels = ['gaussian','exponential','matern3_2','matern5_2'] #kernels available
    #write case for both 1D and 2D for now
    x_training_data = data['training_data'][0]
    y_training_data = data['training_data'][1]

    x_test_data = data['test_data'][0] 
    y_test_data = data['test_data'][1]

    Nk = x_training_data.shape[1] # get problem dimension
    Ns = x_training_data.shape[0] # get sample size

    Ts = x_test_data.shape[0]


    resolution = params['resolution'] #fixed for now
    theta_range = params['theta_range']
    theta_d = np.linspace(theta_range[0],theta_range[1],resolution)

    # dictionary of information for accurate handling
    info = {'name':name,
            'theta_range':theta_range,
            'resolution': resolution,
            'sample_size': Ns,
            'test_size': Ts,
            'problem_dimension':Nk}

    M = [] # measured accuracy


    if (Nk==1):
        for kernel_type in range(len(kernels)):
            model = KRG(x_training_data,y_training_data,kernel=kernels[kernel_type],preprocessing="standardize") #create object
            #initialize Z
            M_single = np.zeros(resolution,resolution)
            for kk in range(resolution):
                thet = theta_d[kk]
                # hack the default kriging code here
                # extract beta and gamma parameters from model
                model.NLL(thet) #check to ensure that the model parameters change to reflect the true value of theta
                model_prediction = model.predict(x_test_data) # get the model prediction
                model_accuracy = get_errors(model,model_prediction,y_test_data) # evaluate the accuracy of the model
                # there is need to uncomplicate the process by selecting just a single metric
                # NRMSE -- Index is 0
                M_single = model_accuracy[0]
            M.append(M_single) # append to the already prepared list    

    elif (Nk == 2):
        X_th,Y_th = np.meshgrid(theta_d,theta_d)

        for kernel_type in range(len(kernels)):
            model = KRG.Kriging(x_training_data,y_training_data,kernel=kernels[kernel_type],preprocessing="standardize")
            #initialize Z
            M_single = np.zeros((resolution,resolution))
            for kk in range(resolution):
                for jj in range(resolution):
                    thet = [Y_th[kk][jj],X_th[kk][jj]]
                    # convert to array for better handling
                    thet = np.array(thet)
                    model.theta = thet # hacking major code to inject theta into Kriging class
                    # hack the default kriging code here
                    # extract beta and gamma parameters from model
                    model.NLL(thet) #check to ensure that the model parameters change to reflect the true value of theta
                    model_prediction = model.predict(x_test_data) # get the model prediction
                    model_accuracy = get_errors(model,model_prediction,y_test_data) # evaluate the accuracy of the model
                    # there is need to uncomplicate the process by selecting just a single metric
                    # NRMSE -- Index is 0
                    M_single[kk][jj] = model_accuracy[0]
            M.append(M_single) # append to the already prepared list       
            
    else:
        a = 3

    # store data --- ease of reuse and data handling
    save_accuracy_data(M,info)
    # method returns info 
    return info

def plot_likelihood_benchmark(info,range):
    # info -- detailed information about the original data
    # range -- the desired range we are interested in visualizing
    # fetch data
    theta_range = info['theta_range']
    theta_d = np.linspace(theta_range[0],theta_range[1],info['resolution'])
    X_th,Y_th = np.meshgrid(theta_d,theta_d)
    G,E,M3,M5 = get_likelihood_data(info)

    # for now, the parameter 'range' does absolutely nothing
    # # check if range is same with the range in info
    # if (range != info['theta_range']):
    #     # do something
    #     c = 0
    # else: #leave as it is 
    #     a = 1 # do nothing

    # reuse plotting codes for different kernels
    plot_contour(X_th,Y_th,G,'gaussian',info,type='likelihood')
    plot_contour(X_th,Y_th,E,'exponential',info,type='likelihood')
    plot_contour(X_th,Y_th,M3,'matern3_2',info,type='likelihood')
    plot_contour(X_th,Y_th,M5,'matern5_2',info,type='likelihood')
    return 0

def plot_accuracy_benchmark(info,range):    
    # info -- detailed information about the original data
    # range -- the desired range we are interested in visualizing
    # fetch data
    theta_range = info['theta_range']
    theta_d = np.linspace(theta_range[0],theta_range[1],info['resolution'])
    X_th,Y_th = np.meshgrid(theta_d,theta_d)
    G,E,M3,M5 = get_likelihood_data(info)
    G,E,M3,M5 = get_accuracy_data(info)
    # reuse plotting codes for different kernels
    plot_contour(X_th,Y_th,G,'gaussian',info,type='accuracy')
    plot_contour(X_th,Y_th,E,'exponential',info,type='accuracy')
    plot_contour(X_th,Y_th,M3,'matern3_2',info,type='accuracy')
    plot_contour(X_th,Y_th,M5,'matern5_2',info,type='accuracy')
    return 0


def plot_contour(X,Y,Z,kernel,info,type='likelihood'):
    # type can either be likelihood or accuracy
    #loop through tract
    fig_count = random.randint(1,9999)
    fig = plt.figure(fig_count,figsize=(20,15))
    ax = fig.add_subplot(111)
    #viridis
    img = ax.contourf(X,Y,Z,50,cmap="jet")
    cbar = fig.colorbar(img)
    cbar.ax.tick_params(labelsize=50)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.xlabel(r"$\theta_1$",fontsize=60)
    plt.ylabel(r"$\theta_2$", fontsize = 60)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    # plt.grid(True)
    if (type == 'likelihood'):
        filename = "likelihood_results/"+info['name']+"_"+str(info['sample_size'])+"_"+str(info['resolution'])+"_"+kernel + ".pdf"
        # contour bar name
        cbar.set_label('NLL', fontsize= 60, rotation=90)
    else:
        filename = "accuracy_results/"+info['name']+"_"+str(info['sample_size'])+"_"+str(info['resolution'])+"_"+kernel + ".pdf"
        # contour bar name
        cbar.set_label(r"$NRMSE (\%)$", fontsize= 60, rotation=90)
    fig.tight_layout()
    plt.savefig(filename,dpi=600)
    # fig_count += 1
    # plt.show()
    return 0


# building list of functions with a decorator
def fix_input(value):
    def apply(f):
        return f(value)
    return apply

function_list = []
def register(function):
    function_list.append(function)

def fix_input_train(TRAIN_SIZE):
    def apply_train(f):
        return f(TRAIN_SIZE)
    return apply_train


training_function_list = []
def register_for_training(function):
    training_function_list.append(function)


# train data registration
# @register_for_training
# def Griewank_nD(TRAIN_SIZE):
#     limits = [[-5,5]]*2
#     x = pyd.lhs(2,75)#hardcode
#     x = scaler.scale(x,limits)
#     f = BP.Griewank_nD(x) 
#     return f,x 

# @register_for_training
# def New_Function_1(TRAIN_SIZE):
#     limits = [[0,2]]*2
#     x = pyd.lhs(2,20)#hardcode
#     x = scaler.scale(x,limits)
#     f = BP.New_Function(x) 
#     return f,x 


# @register_for_training
# def New_Function_2(TRAIN_SIZE):
#     limits = [[0,2]]*2
#     x = pyd.lhs(2,40)#hardcode
#     x = scaler.scale(x,limits)
#     f = BP.New_Function(x) 
#     return f,x 


# @register_for_training
# def New_Function_3(TRAIN_SIZE):
#     limits = [[0,2]]*2
#     x = pyd.lhs(2,60)#hardcode
#     x = scaler.scale(x,limits)
#     f = BP.New_Function(x) 
#     return f,x 

# @register_for_training
# def New_Function_4(TRAIN_SIZE):
#     limits = [[0,2]]*2
#     x = pyd.lhs(2,80)#hardcode
#     x = scaler.scale(x,limits)
#     f = BP.New_Function(x) 
#     return f,x 

# @register_for_training
# def New_Function_5(TRAIN_SIZE):
#     limits = [[0,2]]*2
#     x = pyd.lhs(2,100)#hardcode
#     x = scaler.scale(x,limits)
#     f = BP.New_Function(x) 
#     return f,x 

@register_for_training
def CamelBack_1(TRAIN_SIZE):
    limits = [[-2,2]]*2
    x = pyd.lhs(2,20)#hardcode
    x = scaler.scale(x,limits)
    f = BP.camelBack(x) 
    return f,x 

@register_for_training
def CamelBack_2(TRAIN_SIZE):
    limits = [[-2,2]]*2
    x = pyd.lhs(2,30)#hardcode
    x = scaler.scale(x,limits)
    f = BP.camelBack(x) 
    return f,x 


@register_for_training
def CamelBack_3(TRAIN_SIZE):
    limits = [[-2,2]]*2
    x = pyd.lhs(2,40)#hardcode
    x = scaler.scale(x,limits)
    f = BP.camelBack(x) 
    return f,x 

@register_for_training
def CamelBack_4(TRAIN_SIZE):
    limits = [[-2,2]]*2
    x = pyd.lhs(2,50)#hardcode
    x = scaler.scale(x,limits)
    f = BP.camelBack(x) 
    return f,x 


# test data registration
# @register
# def Griewank_nD(TEST_SIZE):
#     limits = [[-5,5]]*2
#     x = pyd.lhs(2,TEST_SIZE)
#     x = scaler.scale(x,limits)
#     f = BP.Griewank_nD(x) 
#     return f,x 

# @register
# def New_Function_1(TEST_SIZE):
#     limits = [[0,2]]*2
#     x = pyd.lhs(2,TEST_SIZE)
#     x = scaler.scale(x,limits)
#     f = BP.New_Function(x) 
#     return f,x 

# @register
# def New_Function_2(TEST_SIZE):
#     limits = [[0,2]]*2
#     x = pyd.lhs(2,TEST_SIZE)
#     x = scaler.scale(x,limits)
#     f = BP.New_Function(x) 
#     return f,x 


# @register
# def New_Function_3(TEST_SIZE):
#     limits = [[0,2]]*2
#     x = pyd.lhs(2,TEST_SIZE)
#     x = scaler.scale(x,limits)
#     f = BP.New_Function(x) 
#     return f,x 

# @register
# def New_Function_4(TEST_SIZE):
#     limits = [[0,2]]*2
#     x = pyd.lhs(2,TEST_SIZE)
#     x = scaler.scale(x,limits)
#     f = BP.New_Function(x) 
#     return f,x 


# @register
# def New_Function_5(TEST_SIZE):
#     limits = [[0,2]]*2
#     x = pyd.lhs(2,TEST_SIZE)
#     x = scaler.scale(x,limits)
#     f = BP.New_Function(x) 
#     return f,x 

@register
def CamelBack_1(TEST_SIZE):
    limits = [[-2,2]]*2
    x = pyd.lhs(2,TEST_SIZE)
    x = scaler.scale(x,limits)
    f = BP.camelBack(x) 
    return f,x 

@register
def CamelBack_2(TEST_SIZE):
    limits = [[-2,2]]*2
    x = pyd.lhs(2,TEST_SIZE)
    x = scaler.scale(x,limits)
    f = BP.camelBack(x) 
    return f,x 

@register
def CamelBack_3(TEST_SIZE):
    limits = [[-2,2]]*2
    x = pyd.lhs(2,TEST_SIZE)
    x = scaler.scale(x,limits)
    f = BP.camelBack(x) 
    return f,x 

@register
def CamelBack_4(TEST_SIZE):
    limits = [[-2,2]]*2
    x = pyd.lhs(2,TEST_SIZE)
    x = scaler.scale(x,limits)
    f = BP.camelBack(x) 
    return f,x 


TRAIN_SIZE = 40
TEST_SIZE = 500 #200

for set in range(len(function_list)):#create empty dictionaries for both training and test sets
    locals()[str((function_list[set]).__name__)+'_training_data'] = {'x':[], 'y':[]}
    locals()[str((function_list[set]).__name__)+'_test_data'] = {'x':[], 'y':[]}

# '''
# STAGE 1: sample and test points generation for the functions

#generate training and test samples

#generate the training point: LHS and the test set: random selection
# x_train_norm = (pyd.lhs(problem_dimension,sample_size)).T
apply_train = fix_input_train(TRAIN_SIZE)
evaluated_functions_train = [apply_train(f) for f in training_function_list] #contains functional values and scaled inputs (f,x)



apply_test = fix_input(TEST_SIZE)
evaluated_functions_test = [apply_test(f) for f in function_list] #contains functional values and scaled inputs (f,x)


for function_set in range(len(function_list)):
    # save the data for each train and test sets for the functions in separate csv files
    locals()[str((function_list[function_set]).__name__)+'_training_data']['x'].append(evaluated_functions_train[function_set][1])
    locals()[str((function_list[function_set]).__name__)+'_training_data']['y'].append(evaluated_functions_train[function_set][0])

    locals()[str((function_list[function_set]).__name__)+'_test_data']['x'].append(evaluated_functions_test[function_set][1])
    locals()[str((function_list[function_set]).__name__)+'_test_data']['y'].append(evaluated_functions_test[function_set][0])

    # can I log the data for future use?
    # save data
    #combine the name and dimension for storage
    data_name = str((function_list[function_set]).__name__)
    savedata(evaluated_functions_train[function_set][1],evaluated_functions_train[function_set][0],evaluated_functions_test[function_set][1],evaluated_functions_test[function_set][0],data_name) #done for easy reuse
# '''



# '''
# STAGE 2: fetch data and use for benchmark
#benchmark

# standard parameters
kernels = ['gaussian','exponential','matern3_2','matern5_2']
preprocessing = "standardize"
# theta_range = [1e-3,1e1] # works for only camelBack function
theta_range = [1e-1,1e2]
resolution = 200
 # just for tests
# model parameters
params = {'kernels':kernels,'preprocessing':preprocessing, 'theta_range':theta_range, 'resolution': resolution}
for benchmark_set in range(len(function_list)):
    #get the data
    # data = {'training_data': [locals()[str((function_list[benchmark_set]).__name__)+'_training_data']['x'],\
    #     locals()[str((function_list[benchmark_set]).__name__)+'_training_data']['y']],
    #     'test_data' : [locals()[str((function_list[benchmark_set]).__name__)+'_test_data']['x'],\
    #     locals()[str((function_list[benchmark_set]).__name__)+'_test_data']['y']] 
    #     }
    data_name = str((function_list[benchmark_set]).__name__) #get the function name
    x_train,y_train,x_test,y_test = getdata(data_name)

    data = {'training_data': [x_train,y_train],
        'test_data' : [x_test,y_test]
        }
    likelihood_info = model_likelihood_benchmark(data,params,name=str((function_list[benchmark_set]).__name__))
    accuracy_info = model_benchmark(data,params,name=str((function_list[benchmark_set]).__name__))

    # info already contains name of the benchmark set ---- fetch it from there

    desired_range = theta_range # by default
    # plot likelihood
    plot_likelihood_benchmark(likelihood_info,desired_range)
    # # plot accuracy
    plot_accuracy_benchmark(accuracy_info,desired_range)
# '''



# future improvement
# implement within code framework
def trim_matrix(input_matrix,bounds):
    # ensure the conditions are met
    # test -- remove values less than 3 (i.e 1 and 2)
    # loop through each sub-array
    # we assume squared resolution i.e. input_matrix.shape[0] == input_matrix.shape[1]
    # bounds = [LB,UB]
    output_matrix = []
    for i in range(input_matrix.shape[0]):
        # check for the two criteria simultaneously
        checked_array = input_matrix[i] # get the array
        temp_array = checked_array[checked_array >= bounds[0]] # first check
        temp_array_2 = temp_array[temp_array <= bounds[1]] # second check
        output_matrix.append(temp_array_2)
    output_matrix = np.array(output_matrix)   
    return output_matrix