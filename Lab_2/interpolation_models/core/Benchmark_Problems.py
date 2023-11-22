import numpy as np
import math



def branin(x1,x2):
    PI = math.pi
    t = (1/(8*PI))
    t = float(t)
    b = 5.1/(4*math.pow(PI,2))
    b = float(b)
    return (np.power((x2 - (np.power(x1,2) * b) + (5*x1/PI) - 6),2) + 10*(1 - t)*np.cos(x1) + 10)

def branin_2D(x):
    x1 = x[:,0]
    x2 = x[:,1]
    PI = math.pi
    t = (1/(8*PI))
    t = float(t)
    b = 5.1/(4*math.pow(PI,2))
    b = float(b)
    return (np.power((x2 - (np.power(x1,2) * b) + (5*x1/PI) - 6),2) + 10*(1 - t)*np.cos(x1) + 10)

def rosenbrock(x,y):
    """
    This R^2 -> R^1 function should be compatible with algopy.
    http://en.wikipedia.org/wiki/Rosenbrock_function
    A generalized implementation is available
    as the scipy.optimize.rosen function
    """
    a = 1 - x
    b = y - (x*x)
    return a*a + b*b*100


def himmelblau(x,y):
    """
    This R^2 -> R^1 function should be compatible with algopy.
    http://en.wikipedia.org/wiki/Himmelblau%27s_function
    This function has four local minima where the value of the function is 0.
    """
    a = x*x + y - 11
    b = x + y*y - 7
    return a*a + b*b

def wood(x1,x2,x3,x4):
    """
    This R^4 -> R^1 function should be compatible with algopy.
    """
    return sum((
        100*(x1*x1 - x2)**2,
        (x1-1)**2,
        (x3-1)**2,
        90*(x3*x3 - x4)**2,
        10.1*((x2-1)**2 + (x4-1)**2),
        19.8*(x2-1)*(x4-1),
        ))

def nTPHT(n,step=50,xlimit=[-1,1],a=10):
    '''
    n-D Tensor-Product Hyperbolic Tangent Function
    n = number of dimensions
    step = size of input/output of the function
    xlimit = Default boundary : [-1,1]
    a = Abruptness of the step function : 10  
    '''
    x = np.linspace(xlimit[0],xlimit[1],step)
    x = np.array(x)
    f = np.zeros(step)
    for i in range(step):
        temp = 1.0
        for j in range(n):
            temp *= np.tanh(a*x[i])
        f[i] = temp
    return f

def TPHT_1D(x,a=10):
    '''
    1-D Tensor-Product Hyperbolic Tangent Function
    step = size of input/output of the function
    a = Abruptness of the step function : 10  
    '''
    f = np.tanh(a*x)
    return f

def TPHT_nD(x,a=10):
    '''
    x is a multidimensional input
    xlimit[-1 1]
    
    use:
        limit = [-1,1]
        x = halton_sequence(50,1,limit)
        x = np.array(x)
        y = TPHT_nD(x) 
    '''
    x = x.T
    n = x.shape[1] #size of each variable
    m = x.shape[0] #Number of dimensions
    f = np.zeros(n)
    for i in range(n): #variable size
        temp = 1.0
        for j in range(m): #go through the dimension
            temp *= np.tanh(a*x[j][i])
        f[i] = temp
    return f

def Rosenbrock_nD(x):
    '''
    x is a multidimensional input
    xlimit[-2 2]
    ===================================
    limit = [[-2,2], [-2,2]]
    x = halton_sequence(10,2,limit)
    x = np.array(x)
    y = Rosenbrock_nD(x)
    print(y)
    ===================================
    n = 100
    x = np.linspace(-2,2,n)
    X,Y = np.meshgrid(x,x)

    grid_Z = np.zeros((n,n))
    for i in range(n):
        X_all = np.vstack([X[i,:],Y[i,:]])
        y = Rosenbrock_nD(X_all)
        grid_Z[i,:] = y
    =====================================
    plt.figure(1,figsize=(9,6))
    plt.contourf(X,Y,grid_Z,20,cmap="RdGy")
    plt.title('Rosenbrock Function',fontsize=20)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.colorbar()
    plt.show()
    '''
    x = x.T
    n = x.shape[1] #size of each variable
    m = x.shape[0] #Number of dimensions
    f = np.zeros(n)
    #print(x)
    for i in range(n): #variable size
        temp = 0.0
        for j in range(m-1): #go through the dimension
            temp += 100*(x[j+1][i] - (x[j][i])**2)**2 + (1-x[j][i])**2
        f[i] = temp
    return f

def RobotArm_nD(theta,L=1):
    '''
    Input: theta_i : Angle of ith arm segment θ ϵ [0,2π]
           L_i: Length of the ith arm segment L ϵ [0,1]
    Output: r: Distance of tip of robot arm
    Default Length: 1
    Else provide a List of the Length
    L[i] ... L[n+1] e.g. L = [0.5,0.5,0.5]
    if n=2, len(L) = 2+1 = 3
    '''
    theta = np.array(theta)
    theta = theta.T
    n = theta.shape[1] #size of each variable
    m = theta.shape[0] #Number of dimensions
    if (L==1):
        L = np.ones(m+1)
    else:
        if(len(L) != m+1):
            print("Provide sufficient numbers of Length of arm segment")
    r = np.zeros(n)
    #print(x)
    for i in range(n): #variable size
        temp = 0.0
        temp1 = 0.0
        temp2 = 0.0
        for j in range(m+1): #go through the dimension
            thetaSum = 0.0
            for k in range(j):
                thetaSum += theta[k][i]
            temp1 += L[j]*np.cos(thetaSum)
            temp2 += L[j]*np.sin(thetaSum)
        temp = temp1**2 + temp2**2
        r[i] = np.sqrt(temp)
    return r

def CantileverBeam_nD(x,b=0.03,h=0.50,P=50,E=200):
    '''
    Input: x_i : Length of the ith element x ϵ [0.5,1]
           b_i: Width of the ith element b ϵ [0.01,0.05]
           h_i: Height of the ith element h ϵ [0.30,0.65]
           P: Applied force at the tip
           E: Young's modulus
    Output: w: Tip deflection
    Default Width: 0.03
    Default Height: 0.50
    Else provide a List of the Width and Height
    b[i] ... b[n] e.g. b = [0.5,0.5]
    if n=2, len(b) = 2
    '''
    x = np.array(x)
    x = x.T
    n = x.shape[1] #size of each variable
    m = x.shape[0] #Number of dimensions
    

    if (b==0.03):
        b = np.ones(m)
        for i in range(m):
            b[i] = 0.03
    else:
        if(len(b) != m):
            print("Provide sufficient numbers of Width of the elements")
            
    if (h==0.50):
        h = np.ones(m)
        for i in range(m):
            h[i] = 0.5
    else:
        if(len(h) != m):
            print("Provide sufficient numbers of Height of the elements")           
            
    w = np.zeros(n)
    #print(x)
    for i in range(n): #variable size
        temp = 0.0
        for j in range(m): #go through the dimension
            lengthSum1 = 0.0
            lengthSum2 = 0.0
            for k in range(j,m):
                lengthSum1 += x[k][i]
            for k in range(j+1,m):
                lengthSum2 += x[k][i]
            temp += (12/(b[j]*h[j]**3))*(lengthSum1**3 - lengthSum2**3)
        w[i] = (P/(3*E))*temp
    return w

def WeldedBeam_3D(t,h,l):
    '''
    Input: t : Beam thickness t ϵ [5,10]
           h: Beam height h ϵ [0.125,1]
           l: Beam length h ϵ [5,10]
    Output: tau: Shear stress
    '''
    
    tau_p = 6000/(h*l*np.sqrt(2))
    a = 6000*(14 + 0.5*l)
    b = 0.25*(l**2 + (h+t)**2)
    c = 2*(0.707*h*l*(l**2 /12 + 0.25*(h+t)**2))
    tau_pp = (a*np.sqrt(b))/(c)
    tau = np.sqrt(tau_p**2 + tau_pp**2 + (l*tau_p*tau_pp)/np.sqrt(0.25*(l**2 + (h+t)**2)))
    return tau

def waterFlow_3D(rw,r,L,OP=[70000,1000,80,750,10000]):
    '''
    Input: rw: Radius of borehole (m) rw ϵ [0.05,0.15]
           r: Radius of influence (m) r ϵ [100,50000]
           L: Length of borehole (m) L ϵ [1120,1680]
           OP: Takes in a list of the other five parameters
           Tu: Transmissivity of upper aquifer (m2/y) Tu ϵ [63070,115600]: 70000
           Hu: Potentiometric head of upper aquifer (m) Hu ϵ [990,1110]: 1000
           Tl: Transmissivity of lower aquifer (m2/y) Tl ϵ [63.1,116]: 80
           Hl: Potentiometric head of lower aquifer (m) Hl ϵ [700,820]: 750
           Kw: Hydraulic conductivity of borehole (m/y) Kw ϵ [9855,12045]: 10000
    Output: f: Water flow
    '''
    Tu = OP[0]
    Hu = OP[1]
    Tl = OP[2]
    Hl = OP[3]
    Kw = OP[4]
    
    a = 2*np.pi*Tu*(Hu-Hl)
    b = np.log(r/rw)
    c = b*(rw**2)*Kw
    d = (2*L*Tu)/c
    e = Tu/Tl 
    f = a / (b*(1 + d + e))
    return f

def wingWeight_10D(x):
    '''
    Input: Sw: Wing area (ft2) Sw ϵ [150,200]
           As: Aspect ratio A ϵ [6,10]
           Q: Quarter-chord sweep (deg) Q ϵ [-10,10]
           T: Taper ratio T ϵ [0.5,1]
           OP: Takes in a list of the other six parameters
           Wfw: Weight of fuel in the ring (lb) Wfw ϵ [220,300]: 250
           Dp: Dynamic pressure at cruise (lb/ft2) D ϵ [16,45]: 30
           tc: Airfoil thickness to chord ratio tc ϵ [0.08,0.18]: 0.12
           Nz: Ultimate load factor Nz ϵ [2.5,6]: 4
           Wdg: Flight design gross weight (lb) Wdg ϵ [1700,2500]: 2000
           Wp: Paint weight (lb/ft2) Wp ϵ [0.025,0.08]: 0.05
    
    Output: Ww: Wing weight
    '''
    Sw = x[:,0]
    As = x[:,1]
    Q = x[:,2]
    T = x[:,3]
    Wfw = x[:,4]
    Dp = x[:,5]
    tc = x[:,6]
    Nz = x[:,7]
    Wdg = x[:,8]
    Wp = x[:,9]  

    
    a = 0.036*np.power(Sw,0.758)*np.power(Wfw,0.0035)
    b = As / (np.cos(Q))**2
    c = np.power(Dp,0.006)*np.power(T,0.04)
    d = np.power((100*tc/(np.cos(T))),-0.3)
    e = np.power(Nz*Wdg,0.49)
    f = Sw*Wp
    Ww = (a * b * c * d * e) + f
    return Ww

def camelBack(x):
    '''
    Input: x1,x2 [-2,2]
    '''
    x1 = x[:,0]
    x2 = x[:,1]
    y = ((4-(2.1*x1**2)+(x1**4)/3)*x1**2) + x1*x2 + (-4+4*x2**2)*x2**2
    return y



def goldsteinPrice(x):
    '''
    Input: x1,x2 [-2,2]
    '''
    x1 = x[:,0]
    x2 = x[:,1]
    y = (1 + (x1+x2+1)**2 * (19-4*x1+3*x1**2-14*x2+6*x1*x2+3*x2**2))*(30+(2*x1-3*x2)**2 * (18-32*x1+12*x1**2+48*x2-36*x1*x2+27*x2**2))
    return y

def multiModal2D(x):    
    '''
    Input: x1,x2 [-2,2]
    '''
    x1 = x[:,0]
    x2 = x[:,1]
    y = x1*x2*np.sin(x1) + (x1**2)/10 + x1 - 1.5*x2
    return y

def Haupt(x):
    '''
    Input: x1,x2 [0,4]
    '''
    x1 = x[:,0]
    x2 = x[:,1]
    y = x1*np.sin(4*x1) + 1.1*x2*np.sin(2*x2)
    return y

def sasena(x):
    '''
    Input: x1,x2 [0,5]
    '''
    x1 = x[:,0]
    x2 = x[:,1]
    y = 2 + 0.01*(x2-x1**2)**2 + (1-x1)**2 + 2*(2-x2)**2 + 7*np.sin(0.5*x1)*7*np.sin(0.7*x1*x2)
    return y

def hosaki(x):    
    '''
    Input: x1,x2 [0,5]
    '''
    x1 = x[:,0]
    x2 = x[:,1]
    y = (1-8*x1+7*x1**2-(7/3)*x1**3 + (1/4)*x1**4)* x2**2 *np.exp(-x1)
    return y
    
    
def Griewank_nD(x):
    '''
    x is a multidimensional input
    xlimit[-5 5]
    
    use:
        limit = [-5,5]
        x = halton_sequence(50,1,limit)
        x = np.array(x)
        y = Griewank_nD(x) 
    '''
    x = x.T
    n = x.shape[1] #size of each variable
    m = x.shape[0] #Number of dimensions

    a = np.zeros(n)
    b = np.ones(n)
    f = np.zeros(n)
    for i in range(n):
        for j in range(m):
            a[i] += ((x[j][i])**2)/4000

            b[i] *= np.cos((x[j][i])/np.sqrt(m))
        f[i] = a[i] + b[i] + 1.0
    return f

def Hartmann3D(x):
    '''
    x is a multidimensional input
    xlimit[0 1]
    '''

    alpha = [1.0,1.2,3.0,3.2]
    matrixA = [[3.0,10,30],[0.1,10,35],[3.0,10,30],[0.1,10,35]]
    matrixP = 1e-4 *  np.asarray([[3689,1170,2673],[4699,4387,7470],[1091,8732,5547],[381,5743,8828]])
    f = 0.0 #initialization
    for i in range(4):
        AP = 0.0
        for j in range(3):
            AP += matrixA[i][j] * (x[:,j]-matrixP[i][j])**2
        f += alpha[i] * np.exp(-AP) 
    y = -f
    return y


def Ackley(x):
    # '''
    # xlimit [-5,5]
    # Global minima [0,0]
    # '''
    x1 = x[:,0]
    x2 = x[:,1]
    y = -20.0*np.exp(-0.2*np.sqrt(0.5*(x1**2 + x2**2))) - np.exp(0.5*(np.cos(2*np.pi*x1)+np.cos(2*np.pi*x2)))+np.e +20
    return y
 

def multmodal_function_1(x):
    # '''
    # xlimit [-3,3]
    # Global minima [0.228,-1.626]
    # f_min = -6.551
    # Multiple local solutions exist in this defined domain. 
    # The known unique global optimum lies on the point (0.228, –1.626) 
    # where the function value is -6.551. We just use it to show the 
    # performance of our reduced global optimisation methodology.

    # '''
    x1 = x[:,0]
    x2 = x[:,1]
    a = 3*(1-x1)**2
    b = np.exp(-x1**2 - (x2+1)**2)
    c = 10*(0.2*x1 - x1**3 - x2**5)
    d = np.exp(-x1**2 - x2**2)
    e = (1/3)*np.exp(-(x1+1)**2 - x2**2)

    y = a*b - c*d - e

    return y


def New_Function(x):
    # '''
    # xlimit [0,2]
    # '''
    x1 = x[:,0]
    x2 = x[:,1]
    y = 20 + x1**2 + x2**2 - 10*np.cos(2*np.pi*x1) - 10*np.cos(2*np.pi*x2)
    return y