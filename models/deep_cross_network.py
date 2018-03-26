# coding:utf-8

import os

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.initializers import truncated_normal
from keras.layers import Input, Embedding, Concatenate, Add, Lambda, Flatten, Dense
from keras.layers import Layer
from keras.models import Model
from keras.regularizers import l2
from sklearn.metrics import log_loss

from utils import read_input, _Reshape

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


class CrossLayer(Layer):
    def __init__(self, output_dim, num_layer, reg, **kwargs):
        self.output_dim = output_dim
        self.num_layer = num_layer
        self.reg = reg
        self.supports_masking = True
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[1]
        self.W = []
        self.bias = []
        for i in range(self.num_layer):
            self.W.append(
                self.add_weight(shape=[1, self.input_dim],
                                initializer=truncated_normal(stddev=0.01),
                                regularizer=l2(self.reg),
                                name='w_' + str(i)))
            self.bias.append(
                self.add_weight(shape=[1, self.input_dim], initializer='zeros',
                                name='b_' + str(i)))
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

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class DeepCrossNetwork(object):
    def __init__(self, field_dim, feature_dim, embedding_size, batch_size, cross_layer_num,
                 hidden_size, lr, reg, epoch, init_std, seed):
        self.field_dim = field_dim
        self.feature_dim = feature_dim
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.seed = seed

        self.cross_layer_num = cross_layer_num
        self.hidden_size = hidden_size

        self.init_std = init_std
        self.input_dim = self.field_dim[0] + self.field_dim[1] * self.embedding_size

        self.lr = lr
        self.reg = reg
        self.epoch = epoch

        # self.real_value_input = Input(shape=(self.field_dim[0],))
        # self.discrete_input = Input(shape=(self.field_dim[1],))

    def build_model(self):
        real_value_input = Input(shape=(self.field_dim[0],))
        discrete_input = Input(shape=(self.field_dim[1],))
        embeddings = Embedding(self.feature_dim + 1, self.embedding_size,
                               embeddings_initializer=truncated_normal(stddev=self.init_std),
                               embeddings_regularizer=l2(self.reg),
                               mask_zero=True, trainable=True)(discrete_input)
        reshape = _Reshape(target_shape=(-1,))(embeddings)
        # features = Concatenate(axis=1)([real_value_input, reshape])
        features = Concatenate(axis=1)([real_value_input, reshape])
        dense_network_out = features
        for each in self.hidden_size:
            dense_network_out = Dense(each,
                                      activation='relu',
                                      kernel_initializer=truncated_normal(stddev=self.init_std),
                                      kernel_regularizer=l2(self.reg))(dense_network_out)

        cross_network_out = CrossLayer(self.input_dim, self.cross_layer_num, self.reg)(features)
        # self.hidden_size[-1]+ self.field_dim[0] + self.field_dim[1] * self.embedding_size
        concat = Concatenate(axis=1, name='concat')([dense_network_out, cross_network_out])

        output = Dense(1, activation='sigmoid',
                       kernel_initializer=truncated_normal(stddev=self.init_std),
                       kernel_regularizer=l2(self.reg))(concat)
        return Model([real_value_input, discrete_input], [output])

    def train(self, train_inputs, train_labels):
        print('dcn training step...')
        self.model = self.build_model()
        self.model.compile(optimizer=keras.optimizers.Adam(self.lr),
                           loss=keras.losses.binary_crossentropy,
                           metrics=[keras.metrics.binary_crossentropy])
        self.model.fit(train_inputs, train_labels, batch_size=self.batch_size, epochs=self.epoch)

    def train_with_valid(self, train_inputs, train_labels, valid_inputs, valid_labels):
        print('dcn training step...')
        self.model = self.build_model()
        self.model.compile(optimizer=keras.optimizers.Adam(self.lr),
                           loss=keras.losses.binary_crossentropy,
                           metrics=[keras.metrics.binary_crossentropy])
        self.model.fit(train_inputs, train_labels, validation_data=(valid_inputs, valid_labels),
                       batch_size=self.batch_size, epochs=self.epoch)

    def dcn_predict(self, inputs):
        predictions = self.model.predict(inputs).reshape((-1,))
        print('max predictions: ', np.max(predictions))
        print('min predictions: ', np.min(predictions))
        return predictions


if __name__ == '__main__':
    train_file_path = '../data/train.txt'
    test_file_path = 'data/test'
    output_file_path = 'output.txt'
    is_train = True
    drop_pct = 0.95
    num_iterations = 1000
    if is_train:
        featmap, train_real_value, train_discrete, train_labels, \
        valid_real_value, valid_discrete, valid_labels = read_input(train_file_path, test_file_path=None,
                                                                    is_train=is_train, drop_pct=drop_pct)
        features_len = len(featmap)
        print('features length: ', features_len)
        dcn = DeepCrossNetwork([4, 20], features_len, 4, 1024, 10, [128, 128, 128, 128, 128],
                               6e-4, 5e-5, 20, 0.01, 1024)
        dcn.train_with_valid([train_real_value, train_discrete], train_labels,
                             [valid_real_value, valid_discrete], valid_labels)
        valid_pred = dcn.dcn_predict([valid_real_value, valid_discrete])
        print('valid log loss: ', log_loss(valid_labels, valid_pred))
        # dcn.train_with_valid([train_real_value, train_discrete], train_labels,
        #                      [train_real_value, train_discrete], train_labels)
    else:
        featmap, train_real_value, train_discrete, train_labels, \
        test_real_value, test_discrete, test_instance_id = read_input(train_file_path, test_file_path=test_file_path,
                                                                      is_train=False, drop_pct=drop_pct)
        features_len = len(featmap)
        print('features length: ', features_len)
        dcn = DeepCrossNetwork([4, 20], features_len, 4, 1024, 10, [128, 128, 128, 128, 128],
                               6e-4, 5e-5, 20, 0.01, 1024)
        dcn.train([train_real_value, train_discrete], train_labels)
        predictions = dcn.dcn_predict([test_real_value, test_discrete])
        with open(output_file_path, 'w') as f:
            f.write('instance_id predicted_score\n')
            for id, score in zip(test_instance_id, predictions):
                f.write(str(id) + ' ' + str(score) + '\n')
