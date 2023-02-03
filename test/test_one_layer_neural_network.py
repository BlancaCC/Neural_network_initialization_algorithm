import unittest
import numpy as np

from src.one_layer_neural_network import OneLayerNeuralNetwork


class TestOneLayerNeuralNetwork(unittest.TestCase):
    # Class initialization
    def test_create_instance(self):
        A = np.array([[1,0,1],[0,1,1]])
        B = np.array([[1,0]])
        i = lambda x: x
        h = OneLayerNeuralNetwork(A, B, i)
        self.assertIsInstance(h, OneLayerNeuralNetwork)

    def test_create_square_wrong_matrix_shapes(self):
        A = np.array([[1,0,1],[0,1,1]])
        i = lambda x: x
        with self.assertRaises(AssertionError):
            h = OneLayerNeuralNetwork(A, A, i)
    # Prediction
    def test_prediction(self):
        # Translation 
        t_1 = 1
        t_2 = 2
        A = np.array([[t_1,1,0],[t_2,0,1]])
        I_2 = np.array([[1,0], [0,1]])
        I_fun = lambda x:x
        h = OneLayerNeuralNetwork(A, I_2, I_fun)
        X = np.array([[1,2], [0,0],[-1,4]])
        Y = X + np.array([[t_1, t_2] for i in range(3)])
        np.testing.assert_array_equal(h.predict(X), Y)

    
