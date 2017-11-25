import numpy as np
from dataset.mnist import load_mnist
from ..two_layer_net import TwoLayerNet


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
batch_loss_list = []
train_loss_list = []
test_loss_list = []

train_acc_list = []
test_acc_list = []

network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

#hypterparameters
iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

#added
iter_per_epoch = max(train_size / batch_size, 1)

def calculate():
    for i in range(iters_num):
        #minibatch에 사용할 index를 batch_size개 고름
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        #gradient
        grad = network.numerical_gradient(x_batch, t_batch)
        #grad = network.gradient(x_batch, t_batch)

        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        loss_tr = network.loss(x_train, t_train)
        loss_ts = network.loss(x_test, t_test)
        train_loss_list.append(loss_tr)
        test_loss_list.append(loss_ts)

        loss = network.loss(x_batch, t_batch)
        batch_loss_list.append(loss)

        #if(i%10 == 0):
        #    print((float(i) / float(iters_num)) * 100, "% learned")

        #added
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | ", train_acc, ", ", test_acc)

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
    py_train = ploter(train_acc_list, px)
    py_test = ploter(test_acc_list, px)
    plt.plot(px, py_train, label="train")
    plt.plot(px, py_test, label="test")
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.show()

plot()