# coding:utf-8

from keras.layers import Input, Dense, Reshape, Embedding, Dropout,Concatenate
from xgboost_utils import read_input
from keras import backend as K
import tensorflow as tf


class DeepCrossNetwork(object):
    def __init__(self, field_dim, feature_dim, embedding_size,
                 cross_layer_num, hidden_size, init_std, seed, keep_prob):
        self.field_dim = field_dim
        self.feature_dim = feature_dim
        self.embedding_size = embedding_size

        self.cross_layer_num = cross_layer_num
        self.hidden_size = hidden_size

        self.init_std = init_std
        self.keep_prob = keep_prob

    def _build(self):
        real_value_input = Input(shape=(self.field_dim[0],), dtype=tf.float32)
        discrete_input = Input(shape=(self.field_dim[1],), dtype=tf.int32)
        embeddings = Embedding(input_dim=self.feature_dim + 2, output_dim=self.embedding_size)(discrete_input)
        merge = Concatenate(axis=1)([real_value_input,embeddings])
        return merge

if __name__ == '__main__':
    file_path = '../data/train.txt'
    featmap, train_real_value, train_discrete, train_labels, \
        test_real_value, test_discrete, test_labels = read_input(file_path)
    print(len(featmap))
    dcn = DeepCrossNetwork([4,20],168,10,1,[1,2],0.1,1024,0.5)


