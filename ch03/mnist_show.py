# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# normalize : True => 0~1.0 , False => 255유지
# flatten : True => 1차원 배열로 만듬 784개, False => 1 x 28 x 28  3차원 배열
# one_hot_label : label을 one-hot encoding 형태로 저장할지를 정함,
# True => [0, 0, 1, 0, 0, 0] 즉, 정답만 1.
# False => 7, 2 같이 숫자 형태로 레이블 저장함.
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_train.shape)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 형상을 원래 이미지의 크기로 변형
print(img.shape)  # (28, 28)

img_show(img)
