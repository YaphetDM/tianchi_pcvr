# coding:utf-8
from utils import input_data
import xgboost as xgb
import numpy as np

if __name__ == '__main__':
    path = 'data/train.txt'
    train, valuate, test, featmap = input_data(path)
    print(np.sum(train.labels())/len(train.labels()))
