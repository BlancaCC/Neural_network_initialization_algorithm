import numpy as np
from random import shuffle
from one_layer_neural_network import OneLayerNeuralNetwork
from activation_functions import  ramp_function, relu, tanh
import hat_ya_for_learnt_initialization as hat
def learnt_initialization(X: np.array, y:np.array,
                           number_of_neurons:int, activation_function:str) -> OneLayerNeuralNetwork:
    
    # Initial configuration 
    if activation_function == 'relu':
        M = 1
        hat_ya = hat.ya_for_relu
        activation_function = relu
    elif activation_function == 'ramp':
        M = 1
        activation_function = ramp_function
        hat_ya = hat.ya_for_constant_functions
    elif activation_function == 'tanh':
        M = 4
        activation_function = tanh
        hat_ya = hat.ya_for_constant_functions
    else:
        raise ValueError("Activation function not found")
        
    
    samples, dimension = X.shape
    _, output_dimension = y.shape
    # 1. Take a random p 
    p = np.random.uniform(-1,1,dimension)
    nodes = []
    n = 0 
    samples -= 1 
    candidates = list(np.arange(0,samples))
    shuffle(candidates)
    while n < number_of_neurons and bool(candidates):
        # 3.1 Pick a random 
        i = candidates.pop()

        # 3.2 Select subsect that satisfies orthogonality 
        j = 0
        satisfies_orthogonality  = True
        while j < n and satisfies_orthogonality:
            if p.dot(X[i, :] - nodes[j][0]) == 0:
                satisfies_orthogonality = False
            else:
                j += 1
        if satisfies_orthogonality:
            nodes.append((p.dot(X[i, :]), y[i],X[i, :]))
            n += 1
    # 4 Sort elements 
    nodes.sort(key = lambda x: x[0]) # First item is p.x
    
    # 5 Compute equation 
    A = np.zeros((number_of_neurons, dimension+1))
    B = np.zeros((output_dimension,number_of_neurons))
    
    px_a,y_a, _ = nodes[0]
    A[0,0] = M 
    A[0,1:] = 0 * p
    B[:,0] = y_a #y
    
    for i in range(1,number_of_neurons):
        px_s, y_s, x_s = nodes[i]

        aux = 2*M /(px_s - px_a)
        A[i,0] = M - aux * px_a 
        A[i,1:] = aux * p
        B[:,i] = y_s - hat_ya(y_a, x_s, A, B, i, activation_function )

        px_a = px_s
        y_a = y_s

    return OneLayerNeuralNetwork(A,B,activation_function)

if __name__ == '__main__':
    ideal_function = lambda x : x*x*x - x*x +x 
    W = 100
    X = np.linspace(-5,5,W)
    y = np.array(list(map(lambda x: [x], ideal_function(X))))
    X = np.array(list(map(lambda x: [x], X)))
    for activation_function in ['tanh']:
        for n in range(2,90,10):
            h = learnt_initialization(X, y, n, activation_function)

            import matplotlib.pyplot as plt
            plt.plot(X, h.predict(X))
            plt.plot(X, ideal_function(X))
            plt.title(f'number of neurons = {n}, activation function  = {activation_function}')
            plt.show()


    

