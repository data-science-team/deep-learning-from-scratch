# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        # 출력 결과는 one hot encoded 된 각 층의 출력마다 minibatch size만큼 쌓여있으니 axis=1 로 1차원 배열의 index들을 가져와야함
        y = np.argmax(y, axis=1)
        # 상동
        t = np.argmax(t, axis=1)

        # minibatch에서 일치하는 index들을 다 더한 뒤 minibatch size로 나눈다.
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        # 각 층의 numerical_gradient들을 구한다.
        grads = {}
        # grads의 각 노드는 x,t의 갯수만큼 W1의 gradient가 들어있는 배열 (3차원 배열 1차원x,t의갯수 / 2차원 layer input수, 3차원 layer output수)
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
#end of class TwoLayerNet


print('showing parameter shapes')
net = TwoLayerNet(input_size = 784, hidden_size = 100, output_size = 10)
print(net.params['W1'].shape)
print(net.params['b1'].shape)
print(net.params['W2'].shape)
print(net.params['b2'].shape)
#입력 784, hidden layer = 100 nodes, output layer = 10 nodes


print('load mnist')
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
batch_loss_list = []
train_loss_list = []
test_loss_list = []

print(x_train.shape)
print(t_train.shape)

print('setting hyperparameters')
#hyperparameters
iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
print(train_size)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

print("begin train")

def calculate():
    for i in range(iters_num):
        #minibatch에 사용할 index를 batch_size개 고름
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        #gradient
        #grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)

        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        loss_tr = network.loss(x_train, t_train)
        loss_ts = network.loss(x_test, t_test)
        train_loss_list.append(loss_tr)
        test_loss_list.append(loss_ts)

        loss = network.loss(x_batch, t_batch)
        batch_loss_list.append(loss)
        if(i%10 == 0):
            print((float(i) / float(iters_num)) * 100, "% learned")
calculate()
print("finish train")
import matplotlib.pyplot as plt

px = np.arange(0, iters_num, 1)

def ploter(list, ar):
    rtv = []
    for i in ar:
        if(len(list) > i):
            rtv.append(list[int(i)])
        else:
            rtv.append(list[len(list)-1])
    return rtv

def plot():
    py_train = ploter(train_loss_list, px)
    py_test = ploter(test_loss_list, px)
    py_batch = ploter(batch_loss_list, px)
    plt.plot(px, py_train, label="train")
    plt.plot(px, py_test, label="test")
    plt.plot(px, py_batch, label="minibatch")
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.show()

plot()