# code: utf-8
# ADD Feedforward using MNIST
import sys, os
import pickle
import numpy as np
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image
from common.functions import sigmoid, softmax


#load mnist
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("../sample_weight.pkl", 'rb') as f:
        ntw = pickle.load(f)
    return ntw


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

#x = data, t= label
x, t = get_data()

#initialize with trained
network = init_network()
accuracy_cnt = 0

#for all x
for i in range(len(x)):
    #predicted for x[i]
    y = predict(network, x[i])
    p = np.argmax(y)
    #compare with label t[i]
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))