# coding:utf-8

import keras
from keras import backend as K
from keras.layers import Input, Reshape, Embedding, Concatenate, Add, Lambda, Flatten, Dense, Dropout
from keras.layers import Layer
from keras.models import Model

from xgboost_utils import read_input


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
    def __init__(self, field_dim, feature_dim, embedding_size, cross_layer_num,
                 hidden_size, init_std, seed, embed_reg, cross_reg, dense_reg, output_reg,
                 lr, epoch, keep_prob):
        self.field_dim = field_dim
        self.feature_dim = feature_dim
        self.embedding_size = embedding_size
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
                output = Dropout(self.keep_prob)(output)
        return output

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
        # self.hidden_size[-1]+self.input_size
        self.concat = Concatenate(axis=1, name='concat')([dense_network_out, cross_network_out])
        self.output = Dense(1, activation='sigmoid',
                            kernel_initializer=keras.initializers.truncated_normal(stddev=self.init_std),
                            kernel_regularizer=keras.regularizers.l2(self.output_reg))(self.concat)
        return Model([self.real_value_input, self.discrete_input], [self.output])

    def train(self, inputs, labels):
        self.model = self.build_model()
        self.model.compile(optimizer=keras.optimizers.Adam(self.lr),
                           loss=keras.losses.binary_crossentropy,
                           metrics=[keras.metrics.binary_crossentropy])
        self.model.fit(inputs, labels, epochs=self.epoch)
        return self.model

    def get_concat(self):
        pass

    def loss(self):
        pass


if __name__ == '__main__':
    file_path = '../data/train.txt'
    featmap, train_real_value, train_discrete, train_labels, test_real_value, test_discrete, test_labels = read_input(
        file_path)
    dcn = DeepCrossNetwork([4, 20], 168, 10, 2, [10, 10, 10], 0.1, 1024, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 3, 0.5)
    model = dcn.train([train_real_value, train_discrete], train_labels)
    print(dcn.get_concat())
