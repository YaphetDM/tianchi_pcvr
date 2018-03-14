# coding:utf-8
from xgboost_utils import read_input_as_df, add_dict
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.layers import embed_sequence

features = [[1, 0, 2], [4, 0, 6]]
n_words = 6
outputs = embed_sequence(features, vocab_size=n_words + 1, embed_dim=4)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a = sess.run(outputs)
    print(a)
