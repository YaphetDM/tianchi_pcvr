#coding:utf-8
from utils import input_data
import xgboost as xgb
if __name__ == '__main__':
    path = 'data/train.txt'
    train, valuate, test, featmap = input_data(path)
    for i in featmap:
        print(i, featmap[i])
    for each in train.features()[0:5]:
        print(len(each))