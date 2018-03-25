# coding:utf-8

import os

import keras
import numpy as np
import tensorflow as tf
import xgboost as xgb
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Reshape, Embedding, Concatenate, Add, Lambda, Flatten, Dense
from keras.layers import Layer
from keras.models import Model
from sklearn.metrics import auc

from xgboost_utils import read_input

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


class CrossLayer(Layer):
    def __init__(self, output_dim, num_layer, cross_reg, **kwargs):
        self.output_dim = output_dim
        self.num_layer = num_layer
        self.cross_reg = cross_reg
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[1]
        self.W = []
        self.bias = []
        for i in range(self.num_layer):
            self.W.append(
                self.add_weight(shape=[1, self.input_dim],
                                initializer=keras.initializers.truncated_normal(stddev=0.1), name='w_' + str(i),
                                regularizer=keras.regularizers.l2(self.cross_reg),
                                trainable=True))
            self.bias.append(
                self.add_weight(shape=[1, self.input_dim], initializer='zeros', name='b_' + str(i), trainable=True))
        # super(CrossLayer, self).build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        cross = None
        for i in range(self.num_layer):
            if i == 0:
                cross = Lambda(lambda x: Add()(
                    [K.sum(self.W[i] * K.batch_dot(K.reshape(x, (-1, self.input_dim, 1)),
                                                   K.reshape(x, (-1, 1, self.input_dim))), 1, keepdims=True),
                     self.bias[i], x]))(inputs)
            else:
                cross = Lambda(lambda x: Add()(
                    [K.sum(self.W[i] * K.batch_dot(K.reshape(x, (-1, self.input_dim, 1)),
                                                   K.reshape(inputs, (-1, 1, self.input_dim))), 1, keepdims=True),
                     self.bias[i], cross]))(cross)
        return Flatten()(cross)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class DeepCrossNetwork(object):
    def __init__(self, field_dim, feature_dim, embedding_size, batch_size, cross_layer_num,
                 hidden_size, init_std, seed, embed_reg, cross_reg, dense_reg, output_reg,
                 lr, epoch, keep_prob):
        self.field_dim = field_dim
        self.feature_dim = feature_dim
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.seed = seed

        self.cross_layer_num = cross_layer_num
        self.hidden_size = hidden_size

        self.init_std = init_std
        self.embed_reg = embed_reg
        self.cross_reg = cross_reg
        self.dense_reg = dense_reg
        self.output_reg = output_reg
        self.keep_prob = keep_prob
        self.input_dim = self.field_dim[0] + self.field_dim[1] * self.embedding_size

        self.lr = lr
        self.epoch = epoch

        self.real_value_input = Input(shape=(self.field_dim[0],))
        self.discrete_input = Input(shape=(self.field_dim[1],))

    def dense_loop(self, input):
        output = input
        if len(self.hidden_size) == 0:
            pass
        else:
            for each in self.hidden_size:
                output = Dense(each, activation=keras.activations.relu,
                               kernel_initializer=keras.initializers.truncated_normal(
                                   stddev=self.init_std),
                               kernel_regularizer=keras.regularizers.l2(self.dense_reg))(input)
                # output = K.dropout(output, self.keep_prob)
                # output = Dropout(self.keep_prob)(output)
        return output

    def binary_PFA(self, y_true, y_pred, threshold=K.variable(value=0.5)):
        y_pred = K.cast(y_pred >= threshold, 'float32')
        # N = total number of negative labels
        N = K.sum(1 - y_true)
        # FP = total number of false alerts, alerts from the negative class labels
        FP = K.sum(y_pred - y_pred * y_true)
        return FP / N

    def binary_PTA(self, y_true, y_pred, threshold=K.variable(value=0.5)):
        y_pred = K.cast(y_pred >= threshold, 'float32')
        # P = total number of positive labels
        P = K.sum(y_true)
        # TP = total number of correct alerts, alerts from the positive class labels
        TP = K.sum(y_pred * y_true)
        return TP / P

    def auc(self, y_true, y_pred):
        p_tas = tf.stack([self.binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
        p_fas = tf.stack([self.binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
        p_fas = tf.concat([tf.ones((1,)), p_fas], axis=0)
        binSizes = -(p_fas[1:] - p_fas[:-1])
        s = p_tas * binSizes
        return K.sum(s, axis=0)

    def xgb_auc(self, inputs, labels):
        inputs = xgb.DMatrix(inputs)
        prediction = self.xgb_model.predict(inputs)
        return auc(prediction, labels)

    def build_model(self):
        embeddings = Embedding(self.feature_dim + 1, self.embedding_size,
                               embeddings_initializer=keras.initializers.truncated_normal(stddev=self.init_std),
                               embeddings_regularizer=keras.regularizers.l2(self.embed_reg))(self.discrete_input)
        reshape = Reshape(target_shape=(-1,))(embeddings)
        # features = Concatenate(axis=1)([real_value_input, reshape])
        features = Concatenate(axis=1)([self.real_value_input, reshape])
        dense_network_out = Lambda(self.dense_loop)(features)

        cross_network_out = CrossLayer(self.input_dim,
                                       self.cross_layer_num, self.cross_reg)(features)
        # self.hidden_size[-1]+ self.field_dim[0] + self.field_dim[1] * self.embedding_size
        self.concat = Concatenate(axis=1, name='concat')([dense_network_out, cross_network_out])
        output = Dense(1, activation='sigmoid',
                       kernel_initializer=keras.initializers.truncated_normal(stddev=self.init_std),
                       kernel_regularizer=keras.regularizers.l2(self.output_reg))(self.concat)
        return Model([self.real_value_input, self.discrete_input], [output])

    def train(self, inputs, labels):
        print('dcn training step...')
        self.model = self.build_model()
        self.model.compile(optimizer=keras.optimizers.Adam(self.lr),
                           loss=keras.losses.binary_crossentropy,
                           metrics=[keras.metrics.binary_crossentropy, self.auc])
        self.model.fit(inputs, labels, batch_size=self.batch_size, epochs=self.epoch)

    def evaluate_dcn(self, inputs, labels):
        print('dcn evaluate step...')
        _, cross_entropy = self.model.evaluate(inputs, labels, batch_size=self.batch_size * 2)
        print('dcn evaluation loss ', cross_entropy)
        #
        # prediction = self.model.predict(inputs, batch_size=self.batch_size)
        # cross_entropy = self._cross_entropy(prediction, labels)
        # print("dcn cross entropy: ", cross_entropy)

    def get_concat(self, inputs):
        concat_model = Model(self.model.inputs, [self.concat])
        return concat_model.predict(inputs, batch_size=self.batch_size * 2)

    def _cross_entropy(self, prediction, labels):
        return -np.mean(np.log(prediction) * labels) - np.mean(np.log(1 - prediction) - np.log(1 - prediction) * labels)

    def xgb_train_with_concat(self, features, labels, params):
        print('xgb training...')
        dtrain = xgb.DMatrix(self.get_concat(features), labels)
        self.xgb_model = xgb.train(dtrain=dtrain, num_boost_round=30, params=params)

    def evaluate_xgb(self, inputs, labels):
        print('xgb evaluate step ...')
        concat = self.get_concat(inputs)
        dtest = xgb.DMatrix(concat)
        self.output = self.xgb_model.predict(dtest)
        cross_entropy = self._cross_entropy(self.output, labels)
        print('cross entropy added with xgb: ', cross_entropy)


if __name__ == '__main__':
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'logloss',
              'max_depth': 3,
              'alpha': 5,
              'subsample': 0.75,
              'colsample_bytree': 0.75,
              'min_child_weight': 2,
              'eta': 0.2,
              'seed': 1024,
              'nthread': 8,
              'silent': 1}
    file_path = '../data/train.txt'
    featmap, train_real_value, train_discrete, train_labels, test_real_value, test_discrete, test_labels = read_input(
        file_path)
    features_len = len(featmap)
    print('feature length ', features_len)
    # dcn = DeepCrossNetwork([4, 20], features_len, 32, 256, 6, [32, 32, 64, 64], 0.1, 1024, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3,
    #                        5, 0.5)
    dcn = DeepCrossNetwork([4, 20], features_len, 8, 256, 6, [32, 32], 0.1, 1024, 1e-3, 1e-3, 1e-3, 1e-3, 5e-2,
                           5, 0.5)
    dcn.train([train_real_value, train_discrete], train_labels)
    dcn.evaluate_dcn([train_real_value, train_discrete], train_labels)
    dcn.xgb_train_with_concat([train_real_value, train_discrete], train_labels, params)
    dcn.evaluate_xgb([train_real_value, train_discrete], train_labels)
    # print(model.output)
