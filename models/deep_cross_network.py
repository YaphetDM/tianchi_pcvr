# coding:utf-8

import os

import keras
import tensorflow as tf
import xgboost as xgb
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping
from keras.layers import Input, Embedding, Concatenate, Add, Lambda, Flatten, Dense
from keras.layers import Layer
from keras.models import Model
from sklearn.metrics import log_loss

from utils import read_input, _Reshape

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


class CrossLayer(Layer):
    def __init__(self, output_dim, num_layer, cross_reg, **kwargs):
        self.output_dim = output_dim
        self.num_layer = num_layer
        self.cross_reg = cross_reg
        self.supports_masking = True
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[1]
        self.W = []
        self.bias = []
        for i in range(self.num_layer):
            self.W.append(
                self.add_weight(shape=[1, self.input_dim],
                                initializer=keras.initializers.truncated_normal(stddev=0.01), name='w_' + str(i),
                                regularizer=keras.regularizers.l2(self.cross_reg),
                                trainable=True))
            self.bias.append(
                self.add_weight(shape=[1, self.input_dim], initializer='zeros', name='b_' + str(i), trainable=True))
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
                               kernel_initializer=keras.initializers.truncated_normal(stddev=self.init_std),
                               kernel_regularizer=keras.regularizers.l2(self.dense_reg))(output)
        return output

    def build_model(self):
        embeddings = Embedding(self.feature_dim + 1, self.embedding_size,
                               embeddings_initializer=keras.initializers.truncated_normal(stddev=self.init_std),
                               embeddings_regularizer=keras.regularizers.l2(self.embed_reg),
                               mask_zero=True)(self.discrete_input)
        reshape = _Reshape(target_shape=(-1,))(embeddings)
        # features = Concatenate(axis=1)([real_value_input, reshape])
        features = Concatenate(axis=1)([self.real_value_input, reshape])
        dense_network_out = features
        for each in self.hidden_size:
            dense_network_out = Dense(each, activation=keras.activations.relu,
                                      kernel_initializer=keras.initializers.truncated_normal(stddev=self.init_std),
                                      kernel_regularizer=keras.regularizers.l2(self.dense_reg))(dense_network_out)

        cross_network_out = CrossLayer(self.input_dim, self.cross_layer_num, self.cross_reg)(features)
        # self.hidden_size[-1]+ self.field_dim[0] + self.field_dim[1] * self.embedding_size
        concat = Concatenate(axis=1, name='concat')([dense_network_out, cross_network_out])

        output = Dense(1, activation='sigmoid',
                       kernel_initializer=keras.initializers.truncated_normal(stddev=self.init_std),
                       kernel_regularizer=keras.regularizers.l2(self.output_reg))(concat)
        return Model([self.real_value_input, self.discrete_input], [output]), Model(
            [self.real_value_input, self.discrete_input], [concat])

    def train(self, train_inputs, train_labels):
        print('dcn training step...')
        early_stopping = EarlyStopping(monitor='loss', patience=1, min_delta=0.001)
        self.model = self.build_model()[0]
        self.model.compile(optimizer=keras.optimizers.Adam(self.lr),
                           loss=keras.losses.binary_crossentropy,
                           metrics=[keras.metrics.binary_crossentropy])
        self.model.fit(train_inputs, train_labels, batch_size=self.batch_size,
                       callbacks=[early_stopping], epochs=self.epoch)

    def train_with_valid(self, train_inputs, train_labels, valid_inputs, valid_labels):
        print('dcn training step...')
        early_stopping = EarlyStopping(monitor='val_binary_crossentropy', patience=1, min_delta=0.001, mode='min')
        self.model = self.build_model()[0]
        self.model.compile(optimizer=keras.optimizers.Adam(self.lr),
                           loss=keras.losses.binary_crossentropy,
                           metrics=[keras.metrics.binary_crossentropy])
        self.model.fit(train_inputs, train_labels, validation_data=(valid_inputs, valid_labels),
                       batch_size=self.batch_size, callbacks=[early_stopping], epochs=self.epoch)

    def dcn_predict(self, inputs):
        predictions = self.model.predict(inputs)
        return predictions.reshape((-1,))

    def get_concat(self, inputs):
        concat_model = self.build_model()[1]
        return concat_model.predict(inputs, batch_size=self.batch_size * 2)

    def xgb_train_with_concat(self, features, labels, num_boost_round=100, params=None):
        print('xgb training...')
        dtrain = xgb.DMatrix(self.get_concat(features), labels)
        print('num rows', dtrain.num_row())
        print('num columns', dtrain.num_col())
        self.xgb_model = xgb.train(dtrain=dtrain, num_boost_round=num_boost_round, params=params)

    def xgb_predict(self, features):
        dtest = xgb.DMatrix(self.get_concat(features))
        return self.xgb_model.predict(dtest, ntree_limit=self.xgb_model.best_ntree_limit)

    def xgb_logloss(self, features, labels):
        predictions = self.xgb_predict(features)
        return log_loss(labels, predictions)


if __name__ == '__main__':
    train_file_path = 'data/train'
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
        dcn = DeepCrossNetwork([4, 20], features_len, 16, 1024, 6, [128, 128, 128], 0.01, 1024,
                               1e-2, 1e-2, 1e-2, 1e-2, 1e-3, 20, 0.4)
        dcn.train_with_valid([train_real_value, train_discrete], train_labels,
                             [valid_real_value, valid_discrete], valid_labels)
    else:
        featmap, train_real_value, train_discrete, train_labels, \
        test_real_value, test_discrete, test_instance_id = read_input(train_file_path, test_file_path=test_file_path,
                                                                      is_train=False, drop_pct=drop_pct)
        features_len = len(featmap)
        print('features length: ', features_len)
        dcn = DeepCrossNetwork([4, 20], features_len, 8, 256, 4, [32, 32], 0.1, 1024,
                               1e-2, 1e-2, 1e-2, 1e-2, 5e-4, 2, 0.4)
        dcn.train([train_real_value, train_discrete], train_labels)
        predictions = dcn.dcn_predict([test_real_value, test_discrete])
        with open(output_file_path, 'w') as f:
            f.write('instance_id predicted_score\n')
            for i in range(len(test_instance_id)):
                f.write(str(test_instance_id[i]) + ' ' + str(predictions[i]) + '\n')
