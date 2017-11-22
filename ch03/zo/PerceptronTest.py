import numpy as np
from SinglePerceptron import TestSinglePerceptron

TestSinglePerceptron("AND", np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1]))
TestSinglePerceptron("OR", np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 1]))
TestSinglePerceptron("NAND", np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([1, 1, 1, 0]))
TestSinglePerceptron("XOR", np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 0]))