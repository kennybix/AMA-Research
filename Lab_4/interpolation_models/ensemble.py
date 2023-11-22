import numpy as np
from interpolation_models.core import kriging as KRG
from interpolation_models.core import kriging_pls as KPLS
def compute_AIC(LL,Nf):
    AIC = -2*LL + 2*Nf
    return AIC
def compute_AICc(AIC,Nf,n):
    AICc = AIC + ((2*Nf**2 + 2*Nf)/n-Nf-1)
    return AICc

class Kriging: # changed from ensemble to Kriging to maintain conventionalism
    def __init__(self,x,y,kernels,theta0="",krg_type="kriging",method="AICc",optimizer="nelder-mead-c",optimizer_noise=1.0,preprocessing="normalize"):
        self.x = x
        self.y = y
        self.Ns = self.x.shape[0]
        self.Nk = self.x.shape[1]
        self.kernels = kernels
        if theta0 == "": # to ensure varying initial guesses across board
            theta0 = []
            for i in range(self.Nk):
                theta0.append(np.random.uniform(1e-2,5))
        else: 
            theta0 = theta0
        self.theta0 = theta0
        self.method = method 
        self.optimizer = optimizer
        self.optimizer_noise = optimizer_noise
        self.preprocessing = preprocessing
        if krg_type == "kriging":
            self.Kriging_model = KRG
        elif krg_type == "kpls":
            self.Kriging_model = KPLS

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
        self.gamma = []
        self.beta = []

        Nf = self.Nk + 2 #number of hyperparameter + 2
        for i in range(p):
            kernel = self.kernels[i]
            model = (self.Kriging_model).Kriging(self.x,self.y,kernel,optimizer=self.optimizer,optimizer_noise=self.optimizer_noise,preprocessing=self.preprocessing)
            model.train()
            self.hyperparameter.append(model.theta)
            self.beta.append(model.beta)
            self.gamma.append(model.gamma)
            LL = model.likelihood
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
        self.info = {'kernel weights':self.weight}
        return self

    def predict(self,testdata):
        p = len(self.kernels)
        testdata_size = testdata.shape[0]
        y_array = np.zeros((p,testdata_size))
        for i in range(p):
            model = (self.Kriging_model).Kriging(self.x,self.y,self.kernels[i],self.hyperparameter[i],self.optimizer,optimizer_noise=self.optimizer_noise,preprocessing=self.preprocessing)
            model.kernel = self.kernels[i]
            model.theta = self.hyperparameter[i]
            model.beta = self.beta[i]
            model.gamma = self.gamma[i]
            y = model.predict(testdata)
            y = y.reshape(testdata_size,)
            y_array[i,:] = y
        y_w = np.dot(self.weight,y_array)
        y_output = y_w.sum(0)
        self.y_output = y_output
        return y_output

    def computeRMSE(self,y_exact):
        model = (self.Kriging_model).Kriging(self.x,self.y,self.kernels[0],self.theta0) #dummy initialization
        model.y_predict = self.y_output #ovewrite output of the Kriging method
        self.RMSE = model.computeRMSE(y_exact)
        return self.RMSE
    def computeNRMSE(self,y_exact):
        model = (self.Kriging_model).Kriging(self.x,self.y,self.kernels[0],self.theta0) #dummy initialization
        model.y_predict = self.y_output #ovewrite output of the Kriging method
        self.RMSE = model.computeNRMSE(y_exact)
        return self.RMSE


        