The goal of this library is to use bayesian optimization for Active learning

- Active learning involves finding the best estimate for an unknown function



'''
res = BayOpt(objective_function,bounds)

optional parameters
a. number of initial sample = 20
b. accuracy, set the total number of test points: 10,000

1. Generate initial samples within the bounds
2. Get the corresponding output
3. Train the model
4. using the fi
'''

13-Sept-2022

- Sometimes it is very difficult to use my model
- It gives a negative likelihood
- I have tried to troubleshoot it
- Now, I want to use SMT since it is better developed than my codes as at this moment
- The changes are made to adsp_smt