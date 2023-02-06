from one_layer_neural_network import OneLayerNeuralNetwork
import numpy as np

def ya_for_constant_functions(ya, x, A, B, i, activation_function):
    return ya

def ya_for_relu(_, x, A, B, i, activation_function):
    aux_A = A[0:i, :]
    aux_B = B[:,0:i]
    h = OneLayerNeuralNetwork(aux_A,aux_B,activation_function)
    return h.predict(np.array([x]))
