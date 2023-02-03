import numpy as np
from random import shuffle
#from .one_layer_neural_network import OneLayerNeuralNetwork
from one_layer_neural_network import OneLayerNeuralNetwork

def learnt_initialization(X: np.array, y:np.array, number_of_neurons:int, M:float) -> OneLayerNeuralNetwork:
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
            nodes.append((p.dot(X[i, :]), y[i]))
            n += 1
    # 4 Sort elements 
    nodes.sort(key = lambda x: x[0]) # First item is p.x
    
    # 5 Compute equation 
    A = np.zeros((number_of_neurons, dimension+1))
    B = np.zeros((output_dimension,number_of_neurons))
    
    px_a,y_a = nodes[0]
    A[0,0] = M 
    A[0,1:] = 0 * p
    B[:,0] = y_a #y
    
    for i in range(1,number_of_neurons):
        px_s,y_s = nodes[i]

        aux = 2*M /(px_s - px_a)
        A[i,0] = M - aux * px_a 
        A[i,1:] = aux * p
        B[:,i] = y_s - y_a

        px_a = px_s
        y_a = y_s

    return OneLayerNeuralNetwork(A,B,None)


if __name__ == '__main__':
    ideal_function = lambda x : x*x*x - x*x +x 
    W = 100
    X = np.linspace(-5,5,W)
    y = np.array(list(map(lambda x: [x], ideal_function(X))))
    X = np.array(list(map(lambda x: [x], X)))
    for n in range(2,80,10):
        #n = 20
        M = 1
        h = learnt_initialization(X, y, n, M)
        h.activation_function = lambda x : x #min(1,max(0,x))
        import matplotlib.pyplot as plt
        from activation_functions import  ramp_function
        h.activation_function = ramp_function
        plt.plot(X, h.predict(X))
        plt.plot(X, ideal_function(X))
        plt.title(f'n = {n}')
        plt.show()


    

