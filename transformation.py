# -*- coding: utf-8 -*-
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Normalize

class Transformation:
    '''
    处理训练数据的类，某些情况下需要对训练的数据再一次的处理。
    如无需处理的话，不用实现该方法。
    '''
    # def __init__(self):

    def transformation_data(self, x_train=None, y_train=None, x_test=None, y_test=None):
        return x_train, y_train, x_test, y_test
