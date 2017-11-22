# coding: utf-8
import numpy as np

class SinglePerceptron(object):
    def __init__(self, _d, _itr, _initial_w, bias, epoch_print_interval = 10):
        self.itr = _itr             #epoch
        self.learning_rate = _d     #learning_rate
        self.b = bias               #bias
        self.bw = 0                 #weight for bias
        self.w = _initial_w         #initial bias
        self.print_interval = epoch_print_interval  #epoch print interval

    # predict함수. in_data로 값을 재본다
    def inner_predict(self, in_data):
        y = np.sum(self.w * in_data) + self.b * self.bw
        if (y > 0):
            return 1
        else:
            return 0

    #lpredict and log
    def PredictWithLog(self, in_data, label):
        y = self.inner_predict(in_data)
        print("Predict With w:",self.w,self.bw, " x:",in_data," sum(wx):", np.sum(self.w*in_data), " b:", self.b, " / Result : ", y, " label:", label)
        return y


    # traning 함수 in_datas = traning set, labels = traning label set
    def Fit(self, in_datas, labels):
        # 초기 가중치는 전부 0으로 한다
        self.w = np.zeros(in_datas[0].shape)

        # itr번 루프(epoch = itr)
        for i in range(self.itr):
            if(i%self.print_interval == 0):
                print("epoch : ", i)
            #모든 traning set에 대해 1루프 돈다
            for data, label in zip(in_datas, labels):
                y = self.inner_predict(data)
                diff = (label - y) * self.learning_rate
                #update weights
                self.w += diff * data
                self.bw += diff * self.b
                #log
                if(i % self.print_interval == 0):
                    print("after update [w] bw:", self.w, self.bw, " / x:", data, " sum(wx):", np.sum(self.w * data), " b:",self.b, " / y : ", np.sum(self.w * data) + self.b * self.bw ," /  Result : ", y, " label:", label)
            #log for each epoch
            if(i%self.print_interval == 0):
                print("weight = ", self.w, " bias =", self.b, " bias weight = ", self.bw)


def TestSinglePerceptron(name, dataset, labelset):
    print(name, "-----------------------------")
    bp = SinglePerceptron(0.05, 100, np.array([0, 0]), -1, 25)
    print("Predict with INITIAL preceptron")
    for data, label in zip(dataset, labelset):
        bp.PredictWithLog(data, label)
    print("Begin TRANING")
    bp.Fit(dataset, labelset)
    print("Predict with LEARNED preceptron")
    for data, label in zip(dataset, labelset):
        bp.PredictWithLog(data, label)
    print("-----------------------------", name)
    print("")

