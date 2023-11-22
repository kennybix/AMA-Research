'''
Modified Halton Sequence
Original contribution from Dawid Laszuk (https://laszukdawid.com/2017/02/04/halton-sequence-in-python/)
Modified by Kehinde Oyetunde (email: oyetundedamilare@gmail.com)

Modifications made:

(1): the output of the sequence is normalized forcing the values
     to range from 0 to 1 inclusive
(2): The output list is converted to a numpy array for better
     handling
(3): the output of the sequence can be made to range by providing the wanted limits

'''
import numpy as np
# from interpolation_models.ensemble import preprocessing as pre
from sklearn.preprocessing import MinMaxScaler

def next_prime():
    def is_prime(num):
        "Checks if num is a prime value"
        for i in range(2,int(num**0.5)+1):
            if(num % i)==0: return False
        return True
 
    prime = 3
    while(1):
        if is_prime(prime):
            yield prime
        prime += 2
def vdc(n, base=2):
    vdc, denom = 0, 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder/float(denom)
    return vdc

def halton_sequence(size, dim,limits=0):
    '''
    halton_sequence(size,dim,limit)
    where size is the number of samples
    and dim is the dimension of the samples

    Ex:
    x = halton_sequence(size=20,dim=2,limit=[[-5,10],[0,15]])
    '''
    seq = []
    primeGen = next_prime()
    next(primeGen)
    for d in range(dim):
        base = next(primeGen)
        seq.append([vdc(i, base) for i in range(size)])
        #modification starts here 
    # ms = MinMaxScaler()    
    seq = np.array(seq)
    if dim == 1:
        seq = seq.reshape(dim,1)
    # ms.fit(seq)
    # seq = ms.transform(seq)
        

    # for i in range(seq.shape[0]):
    #     seq[i] = pre.normalize(seq[i])
    #modification ends here
    if limits!=0:
        limits = np.array(limits)
        if(dim==1):
            limits = limits.reshape(1,2)
        k = limits.shape[0]
        if(k != dim):
            print("Oops! Dimension not same with the limits. Please revise")
        if(k==1):
            seq[i] = pre.denormalize(seq[i],limits[0][0],limits[0][1])
        else:
            for i in range(k):
                seq[i] = pre.denormalize(seq[i],limits[i][0],limits[i][1])
        #denormalize seq with the ranges 
    return seq
