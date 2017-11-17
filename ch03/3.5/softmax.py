# coding: utf-8
import numpy as np

def softmax(a):
    c = np.max(a)       # softmax overflow 방지를 위해 output layer에서 가장 큰 값을 뽑아냄
    exp_a = np.exp(a-c)   # 가장 큰 값으로 뺌
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y