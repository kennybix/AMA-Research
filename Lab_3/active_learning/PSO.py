from __future__ import division
import random
import numpy as np
import math


#--- COST FUNCTION 
# function we are attempting to optimize (minimize)
def func1(x):
    total=0
    for i in range(len(x)):
        total+=x[i]**2
    return total

#--- MAIN 
class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=(costFunc(np.array(self.position_i).T))

        # # check to see if the current position is an individual best
        # if self.err_i[0] < self.err_best_i or self.err_best_i==-1:
        #     self.pos_best_i=self.position_i
        #     self.err_best_i=self.err_i[0]

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i
    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant

        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                
class PSO():
    def __init__(self,costFunc,x0,bounds,num_particles,maxiter):

        self.costFunc = costFunc
        self.x0 = x0
        self.bounds = bounds 
        self.num_particles = num_particles
        self.maxiter = maxiter 

        # global num_dimensions

    def optimize(self):
        global num_dimensions
        num_dimensions=len(self.x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,self.num_particles):
            swarm.append(Particle(self.x0))

        # begin optimization loop
        i=0
        while i < self.maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,self.num_particles):
                swarm[j].evaluate(self.costFunc)

                # determine if current particle is the best (globally)
                # if swarm[j].err_i[0] < err_best_g or err_best_g == -1:
                #     pos_best_g=list(swarm[j].position_i)
                #     pos_best_g=np.array(pos_best_g)
                #     err_best_g=float(swarm[j].err_i[0])

                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    pos_best_g=np.array(pos_best_g)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,self.num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(self.bounds)
            i+=1
        # err_best_g # best error
        func = self.costFunc(pos_best_g)
        self.func = func 
        return pos_best_g
# '''

# if __name__ == "__PSO__":
#     main()

#--- EXECUTE


# initial=[5,5]               # initial starting location [x1,x2...]
# bounds=[(-10,10),(-10,10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
# PSO(func1,initial,bounds,num_particles=15,maxiter=30)

# Testing === successful so far
'''
def branin(x):
    x1 = x[0]
    x2 = x[1]
    PI = math.pi
    t = (1/(8*PI))
    t = float(t)
    b = 5.1/(4*math.pow(PI,2))
    b = float(b)
    result = (np.power((x2 - (np.power(x1,2) * b) + (5*x1/PI) - 6),2) + 10*(1 - t)*np.cos(x1) + 10)
    # result = result.tolist()
    return result


initial=[-5,-5]               # initial starting location [x1,x2...]
bounds=[(-10,10),(-10,10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
model = PSO(branin,initial,bounds,num_particles=100,maxiter=30)
x = model.optimize()
print("optimal point: {0}".format(x))
print("optimal functional value: {0}".format(model.func))



def Rosenbrock_nD(x):
    x = x.T
    n = 1 #size of each variable
    m = x.shape[0] #Number of dimensions
    f = np.zeros(n)
    #print(x)
    for j in range(m-1): #go through the dimension
        temp = 100*(x[j+1] - (x[j])**2)**2 + (1-x[j])**2
    f = temp
    return f


initial=[0,0]               # initial starting location [x1,x2...]
bounds=[(-2,2),(-2,2)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
model = PSO(Rosenbrock_nD,initial,bounds,num_particles=100,maxiter=30)
x = model.optimize()
print("optimal point: {0}".format(x))
print("optimal functional value: {0}".format(model.func))

'''