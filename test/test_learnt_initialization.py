import unittest
import numpy as np

import src.learnt_initialization

class TestLearntInitialization(unittest.TestCase):
    ideal_function = lambda x : x*x
    def visualTest(self):
        X = ideal_function(np.linspace(-5,5,30))

