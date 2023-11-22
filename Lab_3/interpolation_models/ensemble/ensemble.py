import numpy as np
from interpolation_models.ensemble import kriging_ensemble as KRE

def compute_AIC(LL,Nf):
    AIC = -2*LL + 2*Nf
    return AIC
def compute_AICc(AIC,Nf,n):
    AICc = AIC + ((2*Nf**2 + 2*Nf)/n-Nf-1)
    return AICc

class ensemble:
    def __init__(self,x,y,kernels,theta0,method="AICc",optimizer="CMA-ES",optimizer_noise=1.0):
        self.x = x
        self.y = y
        self.kernels = kernels
        self.theta0 = theta0
        self.Ns = self.x.shape[0]
        self.Nk = self.x.shape[1]
        self.method = method 
        self.optimizer = optimizer
        self.optimizer_noise = optimizer_noise
        
#separate the train and predict methods later
    def train(self):
        p = len(self.kernels)

        AIC = np.zeros(p)
        AICc = np.zeros(p)
        DAIC = np.zeros(p)
        DAICc = np.zeros(p)
        w_AIC = np.zeros(p)
        w_AICc = np.zeros(p)
        self.hyperparameter = []

        Nf = self.Nk + 2 #number of hyperparameter + 2
        for i in range(p):
            kernel = self.kernels[i]
            model = KRE.Kriging(self.x,self.y,kernel,self.theta0,self.optimizer,optimizer_noise=self.optimizer_noise)
            model.train()
            self.hyperparameter.append(model.theta)
            LL = -(model.likelihood)
            AIC[i] = compute_AIC(LL,Nf)
            AICc[i] = compute_AICc(AIC[i],Nf,self.Ns)
        AIC_min = np.min(AIC)
        AICc_min = np.min(AICc)
        sum_DAIC = 0
        sum_DAICc = 0
        for j in range(p):
            DAIC[j] = AIC[j] - AIC_min
            DAICc[j] = AICc[j] - AICc_min
            sum_DAIC += np.exp(-0.5*DAIC[j])
            sum_DAICc += np.exp(-0.5*DAICc[j])
        w_AIC = np.exp(-0.5*DAIC)/sum_DAIC
        w_AICc = np.exp(-0.5*DAICc)/sum_DAICc
        if(self.method=="AICc"):
            output_weight = w_AICc
        elif(self.method=="AIC"):
            output_weight = w_AIC
        else:
            print("Unknown method!")
        weight = output_weight.reshape(1,p)
        self.weight = weight
        return self

    def predict(self,testdata):
        p = len(self.kernels)
        testdata_size = testdata.shape[0]
        y_array = np.zeros((p,testdata_size))
        for i in range(p):
            model = KRE.Kriging(self.x,self.y,self.kernels[i],self.hyperparameter[i],self.optimizer,optimizer_noise=self.optimizer_noise)
            model.kernel = self.kernels[i]
            model.theta = self.hyperparameter[i]
            y = model.predict(testdata)
            y = y.reshape(testdata_size,)
            y_array[i,:] = y
        y_w = np.dot(self.weight,y_array)
        y_output = y_w.sum(0)
        self.y_predict = y_output
        return y_output

    def computeRMSE(self,y_exact):
        self.RMSE = KRE.computeRMSE(self.y_predict,y_exact)
        return self.RMSE
    def computeNRMSE(self,y_exact):
        self.RMSE = KRE.computeNRMSE(self.y_predict,y_exact)
        return self.RMSE
    def computeRMSD(self,y_exact):
        self.RMSD = KRE.computeRMSD(self.y_predict,y_exact)
        return self.RMSD


        