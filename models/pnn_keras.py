# coding:utf-8
import os

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.initializers import truncated_normal
from keras.layers import Layer, Input, Dense, Concatenate, Embedding, LeakyReLU, Dropout, Conv1D, Flatten
from keras.models import Model
from keras.regularizers import l2
from sklearn.metrics import log_loss

from utils import read_input

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


class ZLayer(Layer):
    def __init__(self, output_dim, reg, **kwargs):
        self.output_dim = output_dim
        self.reg = reg
        self.supports_masking = True
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.field_dim = input_shape[1]
        self.embed_size = input_shape[2]
        # self.weight = self.add_weight(shape=(self.batch_size,self.field_dim),
        #                               name='z_weight',
        #                               initializer=truncated_normal(stddev=0.01),
        #                               regularizer=l2(self.reg))
        self.built = True

    def call(self, inputs, **kwargs):
        return Flatten()(Conv1D(self.output_dim, (self.field_dim,))(inputs))

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class InnerProductLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class OuterProductLayer(Layer):
    def __init__(self, output_dim, reg, **kwargs):
        self.output_dim = output_dim
        self.reg = reg
        self.supports_masking = True
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.field_dim = input_shape[1]
        self.embed_size = input_shape[2]
        # self.weight = self.add_weight(shape=[self.embed_size * self.embed_size, self.output_dim],
        #                               name='p_weight',
        #                               initializer=truncated_normal(stddev=0.01),
        #                               regularizer=l2(self.reg))
        self.built = True

    def call(self, inputs, **kwargs):
        f_sigma = K.sum(inputs, axis=1)
        p = K.map_fn(lambda x: K.dot(K.reshape(x, (-1, 1)), K.reshape(x, (1, -1))), elems=f_sigma)
        return Flatten()(Conv1D(self.output_dim, (self.embed_size,))(p))
        # p = K.map_fn(lambda x: K.dot(K.reshape(x, (-1, 1)), K.reshape(x, (1, -1))), elems=f_sigma)
        # return K.dot(K.reshape(p, (-1, self.embed_size * self.embed_size)), self.weight)

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class ProductNetwork(object):
    def __init__(self, field_dim, feature_dim, embedding_size, output_dim, fully_list,
                 epoch, batch_size, lr, reg, keep_prob, init_std, mode='outer'):
        self.field_dim = field_dim
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.embedding_size = embedding_size

        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.reg = reg
        self.keep_prob = keep_prob
        self.init_std = init_std

        self.fully_list = fully_list
        self.mode = mode

    def build_model(self):
        inputs = Input((self.field_dim,))
        embeddings = Embedding(self.feature_dim + 1, self.embedding_size,
                               embeddings_initializer=truncated_normal(self.init_std),
                               embeddings_regularizer=l2(self.reg),
                               mask_zero=False, trainable=True)(inputs)
        z = ZLayer(self.output_dim, self.reg)(embeddings)
        p = None
        if self.mode == 'outer':
            p = OuterProductLayer(self.output_dim, self.reg)(embeddings)
        else:
            pass
        features = Concatenate(axis=1)([z, p])
        outputs = LeakyReLU(1.0)(features)
        for i in range(len(self.fully_list)):
            if i < len(self.fully_list) - 1:
                outputs = Dropout(self.keep_prob)(Dense(self.fully_list[i],
                                                        activation='relu',
                                                        kernel_initializer=truncated_normal(stddev=self.init_std),
                                                        kernel_regularizer=l2(self.reg))(outputs))
            else:
                outputs = Dense(1, activation='sigmoid',
                                kernel_initializer=truncated_normal(stddev=self.init_std),
                                kernel_regularizer=l2(self.reg))(outputs)
        return Model([inputs], outputs)

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

    def pnn_predict(self, inputs):
        predictions = self.model.predict(inputs).reshape((-1))
        print('max predictions: ', np.max(predictions))
        print('min predictions: ', np.min(predictions))
        return predictions


if __name__ == '__main__':
    train_file_path = '../data/train.txt'
    test_file_path = 'data/test'
    output_file_path = 'output.txt'
    is_train = True
    drop_pct = 0.96
    featmap, train_real_value, train_discrete, train_labels, \
    valid_real_value, valid_discrete, valid_labels = read_input(train_file_path, test_file_path=None,
                                                                is_train=is_train, drop_pct=drop_pct)
    features_len = len(featmap)
    print('features length: ', features_len)
    opnn = ProductNetwork(20, features_len, 6, 50, [128, 128, 1], 20, 256, 1e-3, 1e-4, 0.5, 0.01)
    opnn.train_with_valid(train_discrete, train_labels, train_discrete, train_labels)
    valid_pred = opnn.pnn_predict(train_discrete)
    print('valid log loss: ', log_loss(train_labels, valid_pred))
