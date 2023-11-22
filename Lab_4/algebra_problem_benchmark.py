# algebraic problem benchmark
# latest copy as at 9-Nov-2021
import numpy as np
from numpy import *
import math
from numpy.random import sample 
from interpolation_models.core import scaler as SC
import pyDOE2 as pyd
import inspect
from active_learning import adsp
from active_learning import adsp_smt
from active_learning import adsp_cve
# import necessary libraries and models
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


# my libraries
from interpolation_models.core import scaler 
from interpolation_models.core import kriging as KRG
from smt.surrogate_models import KRG as SMT_KRG
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
    # x_train = array(x_train)
    # x_test = array(x_test)
    # y_train = array(y_train)
    # y_test = array(y_test)  
    return x_train,y_train,x_test,y_test


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



def model_benchmark(data,params,name=""):

    '''
    data = {'training_data':train_data, 'test_data':test_data}
    '''

    filename = name
    file1 = open(filename+".txt","a") 

    file1.write('\n')
    file1.write('\n')

    # empty list for model accuracy
    # can I group the single kernels?
    G = []
    E = []
    M3 = []
    M5 = []
    
    SMT_G = []
    SMT_E = []
    SMT_M3 = []
    SMT_M5 = []
    

    # empty list for model training time
    G_tt = []
    E_tt = []
    M3_tt = []
    M5_tt = []


    # empty list for model information
    G_i = []
    E_i = []
    M3_i = []
    M5_i = []


    x_training_data = data['training_data'][0]
    y_training_data = data['training_data'][1]

    x_test_data = data['test_data'][0]
    y_test_data = data['test_data'][1]
    
    # after separating both the training and test sets using various methods
    # run the benchmark once
    for set_data in range(len(x_training_data)): #loops through the problem dimension
        # get the set data for each iteration
        x_train = (x_training_data[set_data])
        y_train = y_training_data[set_data]
        x_test = (x_test_data[set_data])
        y_test = y_test_data[set_data]

        # declare model_parameters
        single_kernel_params = {'kernels':params['kernels'], 'optimizer': 'COBYLA', 'preprocessing':params['preprocessing']}
        # build model
        # # '''
        # errors, elapsed_time, info= get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG',kernels=params['kernels'][0],model_params=single_kernel_params)
        # G.append(errors)
        # G_tt.append(elapsed_time)
        # G_i.append(info)

        # errors, elapsed_time, info= get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG',kernels=params['kernels'][1],model_params=single_kernel_params)
        # E.append(errors)
        # E_tt.append(elapsed_time)
        # E_i.append(info)

        # errors, elapsed_time, info= get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG',kernels=params['kernels'][2],model_params=single_kernel_params)
        # M3.append(errors)
        # M3_tt.append(elapsed_time)
        # M3_i.append(info)


        # errors, elapsed_time, info= get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG',kernels=params['kernels'][3],model_params=single_kernel_params)
        # M5.append(errors)
        # M5_tt.append(elapsed_time)   
        # M5_i.append(info)

	    # [‘abs_exp’, ‘squar_exp’, ‘act_exp’, ‘matern52’, ‘matern32’]
        #smt predictions for validations
        smt_gaussian_model = SMT_KRG(corr='squar_exp')
        smt_exponential_model = SMT_KRG(corr='abs_exp')
        smt_m3_model = SMT_KRG(corr='matern32')
        smt_m5_model = SMT_KRG(corr='matern52')

        smt_gaussian_model.set_training_values(np.array(x_train),np.array(y_train))
        smt_exponential_model.set_training_values(np.array(x_train),np.array(y_train))
        smt_m3_model.set_training_values(np.array(x_train),np.array(y_train))
        smt_m5_model.set_training_values(np.array(x_train),np.array(y_train))

        smt_gaussian_model.train()
        smt_exponential_model.train()
        smt_m3_model.train()
        smt_m5_model.train()

        print(smt_gaussian_model.optimal_theta)
 
        smt_g_ytest = smt_gaussian_model.predict_values(np.array(x_test))
        smt_e_ytest = smt_exponential_model.predict_values(np.array(x_test))
        smt_m3_ytest = smt_m3_model.predict_values(np.array(x_test))
        smt_m5_ytest = smt_m5_model.predict_values(np.array(x_test))

        SMT_G = get_smt_errors(smt_g_ytest,np.array(y_test))
        SMT_E = get_smt_errors(smt_e_ytest,np.array(y_test))
        SMT_M3 = get_smt_errors(smt_m3_ytest,np.array(y_test))
        SMT_M5 = get_smt_errors(smt_m5_ytest,np.array(y_test))

        
    file1.write("\n")
    file1.write("\n")

    # '''

    file1.writelines("G = {0}".format(G))
    file1.write("\n")
    file1.writelines("E = {0}".format(E))
    file1.write("\n")
    file1.writelines("M3 = {0}".format(M3))
    file1.write("\n")
    file1.writelines("M5 = {0}".format(M5))
    file1.write("\n")
    file1.writelines("SMT_G = {0}".format(SMT_G))
    file1.write("\n")
    file1.writelines("SMT_E = {0}".format(SMT_E))
    file1.write("\n")
    file1.writelines("SMT_M3 = {0}".format(SMT_M3))
    file1.write("\n")
    file1.writelines("SMT_M5 = {0}".format(SMT_M5))
    file1.write("\n")
    file1.writelines('Time comparison')
    file1.write("\n")

    file1.writelines("G_tt = {0}".format(G_tt))
    file1.write("\n")
    file1.writelines("E_tt = {0}".format(E_tt))
    file1.write("\n")
    file1.writelines("M3_tt = {0}".format(M3_tt))
    file1.write("\n")
    file1.writelines("M5_tt = {0}".format(M5_tt))
    file1.write("\n")
    file1.write("\n")
    file1.writelines('Model information')
    file1.write("\n")

    file1.writelines("G_i = {0}".format(G_i))
    file1.write("\n")
    file1.writelines("E_i = {0}".format(E_i))
    file1.write("\n")
    file1.writelines("M3_i = {0}".format(M3_i))
    file1.write("\n")
    file1.writelines("M5_i = {0}".format(M5_i))
    file1.write("\n")
    file1.close()
    return 0


# building list of functions with a decorator
def fix_input(value):
    def apply(f):
        return f(value)
    return apply

function_list = []
def register(function):
    function_list.append(function)

def fix_input_train(params):
    def apply_train(f):
        return f(params)
    return apply_train


training_function_list = []
def register_for_training(function):
    training_function_list.append(function)


# train data registration



@register_for_training
def camelBack(params):  #to use
    limits = [[-2,2]]*2
    model = adsp_cve.active_learn(BP.camelBack,limits,params['initial_ns'],params['stopping_criterion_value'],params['max_iter'])
    start_time = time.time()
    x,f = model.get_more_samples()
    elapsed_time = time.time() - start_time
    info = "Learning time with "+str(params['stopping_criterion_value'])+" :"+str(elapsed_time)
    file1 = open("camelBack.txt","a") 
    file1.writelines(info)
    file1.write('\n')
    file1.close
    return f,x


# @register_for_training
# def Himmelblau(params):
#     limits = [[-5,5]]*2
#     model = adsp_cve.active_learn(BP.himmelblau,limits,params['initial_ns'],params['stopping_criterion_value'],params['max_iter'])
#     start_time = time.time()
#     x,f = model.get_more_samples()
#     elapsed_time = time.time() - start_time
#     info = "Learning time with "+str(params['stopping_criterion_value'])+" :"+str(elapsed_time)
#     file1 = open("Himmelblau.txt","a") 
#     file1.writelines(info)
#     file1.write('\n')
#     file1.close
#     return f,x

# @register_for_training
# def branin_2D(params):
#     limits = [[-5,10],[0,15]]
#     model = adsp_cve.active_learn(BP.branin_2D,limits,params['initial_ns'],params['stopping_criterion_value'],params['max_iter'])
#     start_time = time.time()
#     x,f = model.get_more_samples()
#     elapsed_time = time.time() - start_time
#     info = "Learning time with "+str(params['stopping_criterion_value'])+" :"+str(elapsed_time)
#     file1 = open("branin_2D.txt","a") 
#     file1.writelines(info)
#     file1.write('\n')
#     file1.close
#     return f,x



# @register_for_training
# def Ackley(params):  #to use
#     limits = [[-5,5]]*2
#     model = adsp_cve.active_learn(BP.Ackley,limits,params['initial_ns'],params['stopping_criterion_value'],params['max_iter'])
#     start_time = time.time()
#     x,f = model.get_more_samples()
#     elapsed_time = time.time() - start_time
#     info = "Learning time with "+str(params['stopping_criterion_value'])+" :"+str(elapsed_time)
#     file1 = open("Ackley.txt","a") 
#     file1.writelines(info)
#     file1.write('\n')
#     file1.close
#     return f,x


# @register_for_training
# def TPHT_nD(TRAIN_SIZE):
#     limits = [[-1,1]]*2
#     model = adsp.active_learn(BP.TPHT_nD,limits,params['initial_ns'],params['stopping_criterion_value'],params['max_iter'])
#     x,f = model.get_more_samples()
#     return f,x

# @register_for_training
# def RobotArm_nD(TRAIN_SIZE):
#     limits = [[0,2*np.pi]]*2
#     model = adsp_smt.active_learn(BP.RobotArm_nD,limits,params['initial_ns'],params['stopping_criterion_value'],params['max_iter'])
#     x,f = model.get_more_samples()
#     return f,x


# @register_for_training
# def Haupt(TRAIN_SIZE):
#     limits = [[0,4]]*2
#     model = adsp.active_learn(BP.Haupt,limits,params['initial_ns'],params['stopping_criterion_value'],params['max_iter'])
#     x,f = model.get_more_samples()  
#     return f,x

# @register_for_training
# def sasena(params):
#     limits = [[0,5]]*2
#     model = adsp_smt.active_learn(BP.sasena,limits,params['initial_ns'],params['stopping_criterion_value'],params['max_iter'])
#     x,f = model.get_more_samples()  
#     return f,x

# @register_for_training
# def hosaki(params):    
#     limits = [[0,5]]*2
#     model = adsp_smt.active_learn(BP.hosaki,limits,params['initial_ns'],params['stopping_criterion_value'],params['max_iter'])
#     x,f = model.get_more_samples()  
#     return f,x


# @register_for_training
# def New_Function(params):
#     limits = [[0,2]]*2
#     model = adsp_smt.active_learn(BP.New_Function,limits,params['initial_ns'],params['stopping_criterion_value'],params['max_iter'])
#     x,f = model.get_more_samples()
#     return f,x



# test data registration


@register
def camelBack(TEST_SIZE):
    limits = [[-2,2]]*2
    x = (np.random.rand(TEST_SIZE,len(limits))) #generate random samples for testing
    x = SC.scale(x,limits)
    f = BP.camelBack(x) 
    return f,x

# @register
# def Himmelblau(TEST_SIZE):
#     limits = [[-5,5]]*2
#     x = (np.random.rand(TEST_SIZE,len(limits))) #generate random samples for testing
#     x = SC.scale(x,limits)
#     f = BP.himmelblau(x) 
#     return f,x

# @register
# def branin_2D(TEST_SIZE):    
#     limits = [[-5,10],[0,15]]
#     x = (np.random.rand(TEST_SIZE,len(limits))) #generate random samples for testing
#     x = SC.scale(x,limits)
#     f = BP.branin_2D(x) 
#     return f,x


# @register
# def Ackley(TEST_SIZE):
#     limits = [[-5,5]]*2
#     x = (np.random.rand(TEST_SIZE,len(limits))) #generate random samples for testing
#     x = SC.scale(x,limits)
#     f = BP.Ackley(x) 
#     return f,x


# @register
# def New_Function(TEST_SIZE):
#     limits = [[0,2]]*2
#     x = (np.random.rand(TEST_SIZE,len(limits))) #generate random samples for testing
#     x = SC.scale(x,limits)
#     f = BP.New_Function(x) 
#     return f,x

# @register
# def TPHT_nD(TEST_SIZE):
#     limits = [[-1,1]]*2
#     x = (np.random.rand(TEST_SIZE,len(limits))) #generate random samples for testing
#     x = SC.scale(x,limits)
#     f = BP.TPHT_nD(x) 
#     return f,x

# @register
# def RobotArm_nD(TEST_SIZE):
#     limits = [[0,2*np.pi]]*2
#     x = (np.random.rand(TEST_SIZE,len(limits))) #generate random samples for testing
#     x = SC.scale(x,limits)
#     f = BP.RobotArm_nD(x) 
#     return f,x


# @register
# def Haupt(TEST_SIZE):
#     limits = [[0,4]]*2
#     x = (np.random.rand(TEST_SIZE,len(limits))) #generate random samples for testing
#     x = SC.scale(x,limits)
#     f = BP.Haupt(x) 
#     return f,x

# @register
# def sasena(TEST_SIZE):
#     limits = [[0,5]]*2
#     x = (np.random.rand(TEST_SIZE,len(limits))) #generate random samples for testing
#     x = SC.scale(x,limits)
#     f = BP.sasena(x) +


#     return f,x

# @register
# def hosaki(TEST_SIZE):    
#     limits = [[0,5]]*2
#     x = (np.random.rand(TEST_SIZE,len(limits))) #generate random samples for testing
#     x = SC.scale(x,limits)
#     f = BP.hosaki(x) 
#     return f,x


TEST_SIZE = 500

for set in range(len(function_list)):#create empty dictionaries for both training and test sets
    locals()[str((function_list[set]).__name__)+'_training_data'] = {'x':[], 'y':[]}
    locals()[str((function_list[set]).__name__)+'_test_data'] = {'x':[], 'y':[]}

# '''
# STAGE 1: acitve learning of the functions

sample_size = 20 #fixing this

# sample_size = 20 
#generate training and test samples

#generate the training point: LHS and the test set: random selection
# x_train_norm = (pyd.lhs(problem_dimension,sample_size)).T
params = {'initial_ns':sample_size,'stopping_criterion_value':0.01,'max_iter':1000}
apply_train = fix_input_train(params)
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



'''
# STAGE 2: fetch data and use for benchmark
#benchmark

# standard parameters
kernels = ['gaussian','exponential','matern3_2','matern5_2']
preprocessing="standardize"
###
# optimizer = "nelder-mead-c"
# optimizer = "CMA-ES"
optimizer = "COBYLA"


# model parameters
params = {'kernels':kernels,'preprocessing':preprocessing}
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
    model_benchmark(data,params,name=str((function_list[benchmark_set]).__name__))

'''