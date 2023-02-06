import numpy as np

# tanh function 

tanh = lambda x: (np.tanh(x)+1)/2

# Ramp function 
ramp_function_one_value = lambda x : min(1,max(0,x))

def ramp_function(M:np.array) -> np.array:
    return [[ramp_function_one_value(x) for x in row] for row in M]

# Relu 
relu_function_one_value = lambda x : max(0,x)

def relu(M:np.array) -> np.array:
    return [[relu_function_one_value(x) for x in row] for row in M]

