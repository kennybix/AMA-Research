import random
import numpy as np

def get_index(i,i_list):
    n = len(i_list)
    index = 0
    for j in range(n):
        if(i==i_list[j]):
            index = j
    return index

def get_data(x,data_array):
    m = len(x)
    data_array = np.array(data_array)
    data = []
    for i in range(m):
        data.append(data_array[:,int(x[i])])
    return data


# list = [9,63,64,7,2,3,1,4,8,6,10,45]
# n = 5
# train_data = []
# for i in range(n):
#     number = random.choice(list)
#     index = get_index(number,list)
#     train_data.append(number)
#     del list[index]

# print(train_data)
# print(list)

def create(x_data,y_data,training_size):
    n = len(y_data)
    train_data_index = []
    index_list = np.linspace(0,n-1,n)
    index_list = index_list.tolist()
    for i in range(training_size):
        number = random.choice(index_list)
        index = get_index(number,index_list)
        train_data_index.append(number)
        del index_list[index]
    test_data_index = index_list

    x_training_data = get_data(train_data_index,x_data)
    y_training_data = get_data(train_data_index,(y_data.reshape(len(y_data),1)).T)
    x_test_data = get_data(test_data_index,x_data)
    y_test_data = get_data(test_data_index,(y_data.reshape(len(y_data),1)).T)

    return x_training_data,y_training_data,x_test_data,y_test_data