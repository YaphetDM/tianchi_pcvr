# coding:utf-8
import keras
from keras import backend as K
from keras.layers import Layer, Input, Dense, Concatenate, Embedding, LeakyReLU
from keras.models import Model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import os

from utils import read_input

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


class ZLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.supports_masking = True
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.field_dim = input_shape[1]
        self.embed_size = input_shape[2]
        self.weight = self.add_weight(shape=[self.field_dim * self.embed_size, self.output_dim],
                                      name='z_weight',
                                      initializer=keras.initializers.truncated_normal(stddev=0.01))
        self.built = True

    def call(self, inputs, mask=None):
        return K.dot(K.reshape(inputs, (-1, self.field_dim * self.embed_size)), self.weight)

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
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.supports_masking = True
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.field_dim = input_shape[1]
        self.embed_size = input_shape[2]
        self.weight = self.add_weight(shape=[self.embed_size * self.embed_size, self.output_dim],
                                      name='p_weight',
                                      initializer=keras.initializers.truncated_normal(stddev=0.01))
        self.built = True

    def call(self, inputs, **kwargs):
        f_sigma = K.sum(inputs, axis=1)
        p = K.map_fn(lambda x: K.dot(K.reshape(x, (-1, 1)), K.reshape(x, (1, -1))), elems=f_sigma)
        return K.dot(K.reshape(p, (-1, self.embed_size * self.embed_size)), self.weight)

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class ProductNetwork(object):
    def __init__(self, field_dim, feature_dim, embedding_size, output_dim, fully_list,
                 epoch, batch_size, lr, mode='outer'):
        self.field_dim = field_dim
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.embedding_size = embedding_size

        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr

        self.fully_list = fully_list
        self.mode = mode

    def build_model(self):
        inputs = Input((self.field_dim,))
        embeddings = Embedding(self.feature_dim + 1, self.embedding_size, mask_zero=True)(inputs)
        z = ZLayer(self.output_dim)(embeddings)
        p = None
        if self.mode == 'outer':
            p = OuterProductLayer(self.output_dim)(embeddings)
        features = Concatenate(axis=1)([z, p])
        outputs = LeakyReLU(1.0)(features)
        for i in range(len(self.fully_list)):
            if i < len(self.fully_list) - 1:
                outputs = Dense(self.fully_list[i], activation='relu')(outputs)
            else:
                outputs = Dense(1, activation='sigmoid')(outputs)
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
        predictions = self.model.predict(inputs)
        return predictions.reshape((-1,))


if __name__ == '__main__':
    train_file_path = '../data/train.txt'
    test_file_path = 'data/test'
    output_file_path = 'output.txt'
    is_train = True
    drop_pct = 0.95
    featmap, train_real_value, train_discrete, train_labels, \
    valid_real_value, valid_discrete, valid_labels = read_input(train_file_path, test_file_path=None,
                                                                is_train=is_train, drop_pct=drop_pct)
    features_len = len(featmap)
    print('features length: ', features_len)
    opnn = ProductNetwork(20, features_len + 1, 5, 10, [3, 1], 2, 64, 1e-3)
    opnn.train_with_valid(train_discrete, train_labels, valid_discrete, valid_labels)
