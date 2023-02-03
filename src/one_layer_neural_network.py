import numpy as np

class OneLayerNeuralNetwork:
    '''
    One Layer neural Network is defined by 
    h(x) = A . sigma*(B . x)
    '''
    def __init__(self, A:np.array, B: np.array, activation_function): 
        self.activation_function = activation_function
        self.number_of_neurons, self.input_size = A.shape
        self.output_size, number_neurons_test = B.shape

        assert self.number_of_neurons == number_neurons_test, \
            f'The number of A rows {self.number_of_neurons} should be the same as B columns {number_neurons_test}'

        self.A = A
        self.B = B
        self.input_size -= 1 #Bias should not count 

    def predict(self, X:np.matrix) -> np.array:
        '''
        X: data to predict,each row is a element

        output: matrix each column is the output for the neural network
        '''
        # Add bias
        X_size, dimensions = X.shape

        assert dimensions == self.input_size, 'Data dimension does not fit'
        X_bias = np.column_stack((np.ones(X_size), X))
        s = self.A @ X_bias.T
        deltas = self.activation_function(s)
        return (self.B @ deltas).T
