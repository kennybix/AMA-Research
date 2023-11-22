# algebraic problem benchmark for research paper

# latest copy as at 05-April-2022

# import necessary libraries and models
import imp
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
from interpolation_models.core import kriging_pls as KRG_PLS
from interpolation_models.core import kriging_mic as KRG_KMIC
from interpolation_models.core import kriging_jmim_c_1 as KRG_KJMIM
from smt.surrogate_models import KRG as SMT_KRG
from smt.surrogate_models import KPLS as SMT_KPLS
# from interpolation_models import ensemble as KEM, CKL as KCKL, MCKL as KMCKL, MIKL as KMIKL, MIKLP as KMIKL_PLS
# from interpolation_models.core_free_form import kriging_improved as KVI, kriging_vni as KVU, kriging_vnu_pls as KVU_PLS
# from interpolation_models.core_free_form import kriging_improved_smart as KVI_s, kriging_vni_smart as KVU_s, kriging_vnu_pls_smart as KVU_PLS_s
 
from interpolation_models.core import Benchmark_Problems as BP


# print(data.head(5))
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


def computeNRMSE(y_pred,y_exact):
    m = len(y_pred) 
    sum = 0.0
    for i in range(m):
        try:
            sum += np.power((y_exact[i] - y_pred[i]),2)
        except:
            # self.y_predict = self.y_predict.reshape(m,)
            y_exact = np.asarray(y_exact)
            sum += np.power((y_exact[i] - y_pred[i]),2)
    RMSE = np.sqrt(sum / m)
    RMSE /= (np.max(y_exact)-np.min(y_exact))
    return RMSE

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
    RMSE = 100*(computeNRMSE(y_true,y_pred)[0])
    MAE = mean_absolute_error(y_true,y_pred)
    R2 = r2_score(y_true,y_pred)
    MSE = mean_squared_error(y_true,y_pred)
    errors = [RMSE,MAE,R2,MSE]
    return errors


def get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG',kernels="", pls_n_comp="", model_params={}):
    # eval('KRG') == KRG # smart way to select without having to use if-statements
    if model=='KRG_PLS':
        model = str_to_class(model).Kriging(x_train,y_train,kernels,optimizer=model_params['optimizer'],preprocessing=model_params['preprocessing'], pls_n_comp=pls_n_comp)
    else:
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

    KPLS_1 = []
    KPLS_2 = []
    KPLS_3 = []
    KPLS_4 = []   

    SMT_G = []
    SMT_K1 = []
    SMT_K2 = []
    SMT_K3 = []
    

    # empty list for model training time
    G_tt = []
    E_tt = []
    M3_tt = []
    M5_tt = []
    

    KPLS_1_tt = []
    KPLS_2_tt = []
    KPLS_3_tt = []
    KPLS_4_tt = []   

    KMIC_tt = [] #my version
    KJMIM_tt = [] #my version
    
    SMT_G_tt = []
    SMT_K1_tt = []
    SMT_K2_tt = []
    SMT_K3_tt = []

    # empty list for model information
    G_i = []
    E_i = []
    M3_i = []
    M5_i = []

    KPLS_1_i = []
    KPLS_2_i = []
    KPLS_3_i = []
    KPLS_4_i = []   

    KMIC_i = [] #my version
    KJMIM_i = [] #my version
    
    SMT_G_i  = []
    SMT_K1_i  = []
    SMT_K2_i  = []
    SMT_K3_i  = []


    KJMIM_G = [] #my version
    KJMIM_E= [] #my version
    KJMIM_M3 = [] #my version
    KJMIM_M5 = [] #my version

    # empty list for model training time
    KJMIM_G_tt = [] #my version
    KJMIM_E_tt = [] #my version
    KJMIM_M3_tt = [] #my version
    KJMIM_M5_tt = [] #my version

    # empty list for model information
    KJMIM_G_i = [] #my version
    KJMIM_E_i = [] #my version
    KJMIM_M3_i = [] #my version
    KJMIM_M5_i = [] #my version


    KMIC_G = [] #my version
    KMIC_E= [] #my version
    KMIC_M3 = [] #my version
    KMIC_M5 = [] #my version

    KMIC_G_tt = [] #my version
    KMIC_E_tt = [] #my version
    KMIC_M3_tt = [] #my version
    KMIC_M5_tt = [] #my version

    KMIC_G_i = [] #my version
    KMIC_E_i = [] #my version
    KMIC_M3_i = [] #my version
    KMIC_M5_i = [] #my version

    x_training_data = data['training_data'][0]
    y_training_data = data['training_data'][1]

    x_test_data = data['test_data'][0]
    y_test_data = data['test_data'][1]
    
    # after separating both the training and test sets using various methods
    # run the benchmark once
    # for set_data in range(len(x_training_data)): #loops through the problem dimension
        # get the set data for each iteration
    x_train = x_training_data
    y_train = y_training_data
    x_test = x_test_data
    y_test = y_test_data

    # declare model_parameters
    single_kernel_params = {'kernels':params['kernels'], 'optimizer': 'COBYLA', 'preprocessing':params['preprocessing']}
    # build model
    # # '''
    errors, elapsed_time, info = get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG',kernels=params['kernels'][0],model_params=single_kernel_params)
    G.append(errors)
    G_tt.append(elapsed_time)
    G_i.append(info)

    # errors, elapsed_time, info = get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG',kernels=params['kernels'][1],model_params=single_kernel_params)
    # E.append(errors)
    # E_tt.append(elapsed_time)
    # E_i.append(info)

    # errors, elapsed_time, info = get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG',kernels=params['kernels'][2],model_params=single_kernel_params)
    # M3.append(errors)
    # M3_tt.append(elapsed_time)
    # M3_i.append(info)

    # errors, elapsed_time, info = get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG',kernels=params['kernels'][3],model_params=single_kernel_params)
    # M5.append(errors)
    # M5_tt.append(elapsed_time)
    # M5_i.append(info)  


    # errors, elapsed_time, info = get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG_PLS',kernels=params['kernels'][0], pls_n_comp = 1, model_params=single_kernel_params)
    # KPLS_1.append(errors)
    # KPLS_1_tt.append(elapsed_time)
    # KPLS_1_i.append(info)

    # errors, elapsed_time, info = get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG_PLS',kernels=params['kernels'][0], pls_n_comp = 2, model_params=single_kernel_params)
    # KPLS_2.append(errors)
    # KPLS_2_tt.append(elapsed_time)
    # KPLS_2_i.append(info)


    # errors, elapsed_time, info = get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG_PLS',kernels=params['kernels'][0], pls_n_comp = 3, model_params=single_kernel_params)
    # KPLS_3.append(errors)
    # KPLS_3_tt.append(elapsed_time)
    # KPLS_3_i.append(info)

    # errors, elapsed_time, info = get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG_PLS',kernels=params['kernels'][0], pls_n_comp = 4, model_params=single_kernel_params)
    # KPLS_4.append(errors)
    # KPLS_4_tt.append(elapsed_time)
    # KPLS_4_i.append(info)


    # errors, elapsed_time, info = get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG_KJMIM',kernels=params['kernels'][0],model_params=single_kernel_params)
    # KJMIM_G.append(errors)
    # KJMIM_G_tt.append(elapsed_time)
    # KJMIM_G_i.append(info)


    # errors, elapsed_time, info = get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG_KJMIM',kernels=params['kernels'][1],model_params=single_kernel_params)
    # KJMIM_E.append(errors)
    # KJMIM_E_tt.append(elapsed_time)
    # KJMIM_E_i.append(info)


    # errors, elapsed_time, info = get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG_KJMIM',kernels=params['kernels'][2],model_params=single_kernel_params)
    # KJMIM_M3.append(errors)
    # KJMIM_M3_tt.append(elapsed_time)
    # KJMIM_M3_i.append(info)

    # errors, elapsed_time, info = get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG_KJMIM',kernels=params['kernels'][3],model_params=single_kernel_params)
    # KJMIM_M5.append(errors)
    # KJMIM_M5_tt.append(elapsed_time)
    # KJMIM_M5_i.append(info)

    # errors, elapsed_time, info = get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG_KMIC',kernels=params['kernels'][0],model_params=single_kernel_params)
    # KMIC_G.append(errors)
    # KMIC_G_tt.append(elapsed_time)
    # KMIC_G_i.append(info)

    # errors, elapsed_time, info = get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG_KMIC',kernels=params['kernels'][1],model_params=single_kernel_params)
    # KMIC_E.append(errors)
    # KMIC_E_tt.append(elapsed_time)
    # KMIC_E_i.append(info)


    # errors, elapsed_time, info = get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG_KMIC',kernels=params['kernels'][2],model_params=single_kernel_params)
    # KMIC_M3.append(errors)
    # KMIC_M3_tt.append(elapsed_time)
    # KMIC_M3_i.append(info)

    # errors, elapsed_time, info = get_individual_model_performance(x_train,y_train,x_test,y_test,model='KRG_KMIC',kernels=params['kernels'][3],model_params=single_kernel_params)
    # KMIC_M5.append(errors)
    # KMIC_M5_tt.append(elapsed_time)
    # KMIC_M5_i.append(info)

    #much later -- modify individual_model_performance to be able to handle SMT library tools

    # [‘abs_exp’, ‘squar_exp’, ‘act_exp’, ‘matern52’, ‘matern32’]
    #smt predictions for validations
    # smt_gaussian_model = SMT_KRG(corr='squar_exp')
    # smt_k1_model = SMT_KPLS(n_comp=1)
    # smt_k2_model = SMT_KPLS(n_comp=2)
    # smt_k3_model = SMT_KPLS(n_comp=3)


    # smt_exponential_model = SMT_KRG(corr='abs_exp')
    # smt_m3_model = SMT_KRG(corr='matern32')
    # smt_m5_model = SMT_KRG(corr='matern52')

    # smt_gaussian_model.set_training_values(np.array(x_train),np.array(y_train))
    # smt_k1_model.set_training_values(np.array(x_train),np.array(y_train))
    # smt_k2_model.set_training_values(np.array(x_train),np.array(y_train))
    # smt_k3_model.set_training_values(np.array(x_train),np.array(y_train))

    # smt_exponential_model.set_training_values(np.array(x_train),np.array(y_train))
    # smt_m3_model.set_training_values(np.array(x_train),np.array(y_train))
    # smt_m5_model.set_training_values(np.array(x_train),np.array(y_train))

    # write function for this ?
    # start_time = time.time()
    # smt_gaussian_model.train()
    # elapsed_time = time.time() - start_time
    # SMT_G_tt.append(elapsed_time)

    # start_time = time.time()
    # smt_k1_model.train()
    # elapsed_time = time.time() - start_time
    # SMT_K1_tt.append(elapsed_time)

    # start_time = time.time()
    # smt_k2_model.train()
    # elapsed_time = time.time() - start_time
    # SMT_K2_tt.append(elapsed_time)


    # start_time = time.time()
    # smt_k3_model.train()
    # elapsed_time = time.time() - start_time
    # SMT_K3_tt.append(elapsed_time)


    # smt_exponential_model.train()
    # smt_m3_model.train()
    # smt_m5_model.train()
    # SMT_G_i = {'Theta':smt_gaussian_model.optimal_theta,
    #                 'Likelihood':(smt_gaussian_model._reduced_likelihood_function(smt_gaussian_model.optimal_theta))[0]}   

    # SMT_K1_i = {'Theta':smt_k1_model.optimal_theta,
    #                 'Likelihood':(smt_k1_model._reduced_likelihood_function(smt_k1_model.optimal_theta))[0]}   

    # SMT_K2_i = {'Theta':smt_k2_model.optimal_theta,
    #                 'Likelihood':(smt_k2_model._reduced_likelihood_function(smt_k2_model.optimal_theta))[0]}

    # SMT_K3_i = {'Theta':smt_k3_model.optimal_theta,
    #                 'Likelihood':(smt_k3_model._reduced_likelihood_function(smt_k3_model.optimal_theta))[0]}

    # smt_g_ytest = smt_gaussian_model.predict_values(np.array(x_test))
    # smt_k1_ytest = smt_k1_model.predict_values(np.array(x_test))
    # smt_k2_ytest = smt_k2_model.predict_values(np.array(x_test))
    # smt_k3_ytest = smt_k3_model.predict_values(np.array(x_test))


    # smt_e_ytest = smt_exponential_model.predict_values(np.array(x_test))
    # smt_m3_ytest = smt_m3_model.predict_values(np.array(x_test))
    # smt_m5_ytest = smt_m5_model.predict_values(np.array(x_test))

    # SMT_G = get_smt_errors(smt_g_ytest,np.array(y_test))
    # SMT_K1 = get_smt_errors(smt_k1_ytest,np.array(y_test))
    # SMT_K2 = get_smt_errors(smt_k2_ytest,np.array(y_test))
    # SMT_K3 = get_smt_errors(smt_k3_ytest,np.array(y_test))


    # SMT_E = get_smt_errors(smt_e_ytest,np.array(y_test))
    # SMT_M3 = get_smt_errors(smt_m3_ytest,np.array(y_test))
    # SMT_M5 = get_smt_errors(smt_m5_ytest,np.array(y_test))

    file1.write("\n")
    file1.write("\n")

    # '''

    file1.writelines("G = {0}".format(G))
    # file1.write("\n")
    # file1.writelines("E = {0}".format(E))
    # file1.write("\n")
    # file1.writelines("M3 = {0}".format(M3))
    # file1.write("\n")
    # file1.writelines("M5 = {0}".format(M5))

    # file1.writelines("KPLS_1 = {0}".format(KPLS_1))
    # file1.write("\n")
    # file1.writelines("KPLS_2 = {0}".format(KPLS_2))
    # file1.write("\n")
    # file1.writelines("KPLS_3 = {0}".format(KPLS_3))
    # file1.write("\n")
    # file1.writelines("KPLS_4 = {0}".format(KPLS_4))
    # file1.write("\n")
    # file1.writelines("SMT_K1 = {0}".format(SMT_K1))
    # file1.write("\n")
    # file1.writelines("SMT_K2 = {0}".format(SMT_K2))
    # file1.write("\n")
    # file1.writelines("SMT_K3 = {0}".format(SMT_K3))
    # file1.writelines("KJMIM_G = {0}".format(KJMIM_G))
    # file1.write("\n")    
    # file1.writelines("KJMIM_E = {0}".format(KJMIM_E))
    # file1.write("\n")
    # file1.writelines("KJMIM_M3 = {0}".format(KJMIM_M3))
    # file1.write("\n")
    # file1.writelines("KJMIM_M5 = {0}".format(KJMIM_M5))
    # file1.write("\n")
    # file1.write("\n")
    # file1.writelines("KMIC_G = {0}".format(KMIC_G))
    # file1.write("\n")    
    # file1.writelines("KMIC_E = {0}".format(KMIC_E))
    # file1.write("\n")
    # file1.writelines("KMIC_M3 = {0}".format(KMIC_M3))
    # file1.write("\n")
    # file1.writelines("KMIC_M5 = {0}".format(KMIC_M5))
    file1.write("\n")
    file1.writelines('Time comparison')
    file1.write("\n")

    file1.writelines("G_tt = {0}".format(G_tt))
    file1.write("\n")
    # file1.writelines("E_tt = {0}".format(E_tt))
    # file1.write("\n")
    # file1.writelines("M3_tt = {0}".format(M3_tt))
    # file1.write("\n")
    # file1.writelines("M5_tt = {0}".format(M5_tt))

    # file1.writelines("KPLS_1_tt = {0}".format(KPLS_1_tt))
    # file1.write("\n")
    # file1.writelines("KPLS_2_tt = {0}".format(KPLS_2_tt))
    # file1.write("\n")
    # file1.writelines("KPLS_3_tt = {0}".format(KPLS_3_tt))
    # file1.write("\n")
    # file1.writelines("KPLS_4_tt = {0}".format(KPLS_4_tt))
    file1.write("\n")
    # file1.writelines("SMT_K1_tt = {0}".format(SMT_K1_tt))
    # file1.write("\n")
    # file1.writelines("SMT_K2_tt = {0}".format(SMT_K2_tt))
    # file1.write("\n")
    # file1.writelines("SMT_K3_tt = {0}".format(SMT_K3_tt))
    # file1.writelines("KJMIM_G_tt = {0}".format(KJMIM_G_tt))
    # file1.write("\n")
    # file1.writelines("KJMIM_E_tt = {0}".format(KJMIM_E_tt))
    # file1.write("\n")
    # file1.writelines("KJMIM_M3_tt = {0}".format(KJMIM_M3_tt))
    # file1.write("\n")
    # file1.writelines("KJMIM_M5_tt = {0}".format(KJMIM_M5_tt))
    file1.write("\n")
    # file1.writelines("KMIC_G_tt = {0}".format(KMIC_G_tt))
    # file1.write("\n")
    # file1.writelines("KMIC_E_tt = {0}".format(KMIC_E_tt))
    # file1.write("\n")
    # file1.writelines("KMIC_M3_tt = {0}".format(KMIC_M3_tt))
    # file1.write("\n")
    # file1.writelines("KMIC_M5_tt = {0}".format(KMIC_M5_tt))
    file1.write("\n")
    file1.writelines('Model information')
    file1.write("\n")

    file1.writelines("G_i = {0}".format(G_i))
    # file1.write("\n")
    # file1.writelines("E_i = {0}".format(E_i))
    # file1.write("\n")
    # file1.writelines("M3_i = {0}".format(M3_i))
    # file1.write("\n")
    # file1.writelines("M5_i = {0}".format(M5_i))

    file1.write("\n")
    # file1.writelines("KPLS_1_i = {0}".format(KPLS_1_i))
    # file1.write("\n")
    # file1.writelines("KPLS_2_i = {0}".format(KPLS_2_i))
    # file1.write("\n")
    # file1.writelines("KPLS_3_i = {0}".format(KPLS_3_i))
    # file1.write("\n")
    # file1.writelines("KPLS_4_i = {0}".format(KPLS_4_i))
    # file1.write("\n")
    # file1.writelines("SMT_K1_i = {0}".format(SMT_K1_i))
    # file1.write("\n")
    # file1.writelines("SMT_K2_i = {0}".format(SMT_K2_i))
    # file1.write("\n")
    # file1.writelines("SMT_K3_i = {0}".format(SMT_K3_i))
    # file1.writelines("KJMIM_G_i = {0}".format(KJMIM_G_i))
    # file1.write("\n")
    # file1.writelines("KJMIM_E_i = {0}".format(KJMIM_E_i))
    # file1.write("\n")
    # file1.writelines("KJMIM_M3_i = {0}".format(KJMIM_M3_i))
    # file1.write("\n")
    # file1.writelines("KJMIM_M5_i = {0}".format(KJMIM_M5_i))
    file1.write("\n")  
    # file1.writelines("KMIC_G_i = {0}".format(KMIC_G_i))
    # file1.write("\n")
    # file1.writelines("KMIC_E_i = {0}".format(KMIC_E_i))
    # file1.write("\n")
    # file1.writelines("KMIC_M3_i = {0}".format(KMIC_M3_i))
    # file1.write("\n")
    # file1.writelines("KMIC_M5_i = {0}".format(KMIC_M5_i))
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

def fix_input_train(value):
    def apply_train(f):
        return f(value)
    return apply_train


training_function_list = []
def register_for_training(function):
    training_function_list.append(function)


# train data registration


@register_for_training
def real_world_problem(seed):
    #'using seed value to ensure consistency in randomization'
    #split data base on the seed
    dataset = np.loadtxt("concrete_data.txt")
    x_data = dataset[:,:-1]
    y_data = dataset[:,-1] 
    Xtrain, Xtest, ytrain, ytest = train_test_split(x_data, y_data, random_state=seed,train_size=0.80) #split the dataset into training data and truthset for validation
    return ytrain,Xtrain




# test data registration
@register
def real_world_problem(seed):
    #'using seed value to ensure consistency in randomization'
    dataset = np.loadtxt("concrete_data.txt")
    x_data = dataset[:,:-1]
    y_data = dataset[:,-1] 
    Xtrain, Xtest, ytrain, ytest = train_test_split(x_data, y_data, random_state=seed,train_size=0.80) #split the dataset into training data and truthset for validation
    return ytest,Xtest


for i in range(5):#looping to get 20 unique results
        
    # TRAIN_SIZE = 40
    SEED = i+1

    for set in range(len(function_list)):#create empty dictionaries for both training and test sets
        locals()[str((function_list[set]).__name__)+'_training_data'] = {'x':[], 'y':[]}
        locals()[str((function_list[set]).__name__)+'_test_data'] = {'x':[], 'y':[]}

    # '''
    # STAGE 1: sample and test points generation for the functions
    sample_size = 30 #fixing this 
    #generate training and test samples

    #generate the training point: LHS and the test set: random selection
    # x_train_norm = (pyd.lhs(problem_dimension,sample_size)).T
    params = {'initial_ns':sample_size,'stopping_criterion_value':0.05,'max_iter':1000}
    apply_train = fix_input_train(SEED)
    evaluated_functions_train = [apply_train(f) for f in training_function_list] #contains functional values and scaled inputs (f,x)



    apply_test = fix_input(SEED)
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
    # preprocessing = "normalize"
    # theta_range = [1e-3,1e2]
    # theta_range = [1e-4,1e4]
    resolution = 200
    # just for tests
    # model parameters
    # params = {'kernels':kernels,'preprocessing':preprocessing, 'theta_range':theta_range, 'resolution': resolution}
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

# '''
