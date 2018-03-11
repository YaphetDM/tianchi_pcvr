# coding:utf-8
from xgboost_utils import read_input_as_df, add_dict
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    path = 'data/train.txt'
    aa = read_input_as_df(path,'2018-09-18',df=5)
    featmap = aa[0]
    x_train = aa[1]
    print(featmap)
    print(x_train)
    # iris = load_iris()
    # X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target, test_size=0.3)
    # print(y_test)
    # params = {'booster': 'gbtree',
    #           'objective': 'multi:softmax',
    #           'eval_metric': 'mlogloss',
    #           'max_depth': 4,
    #           'lambda': 5,
    #           'subsample': 0.75,
    #           'colsample_bytree': 0.75,
    #           'min_child_weight': 2,
    #           'eta': 0.05,
    #           'seed': 1024,
    #           'num_class': 3,
    #           'nthread': 8,
    #           'silent': 1}
    # dtrain = xgb.DMatrix(data=X_train, label=y_train)
    # dtest = xgb.DMatrix(data=X_test)
    #
    # watchlist = [(dtrain, 'train')]
    # model = xgb.train(params, dtrain, num_boost_round=500, evals=watchlist)
    #
    # pred = model.predict(dtest)
    # print(np.sum(y_test == pred)/len(y_test))
    # print(pred)