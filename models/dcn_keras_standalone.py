# coding:utf-8

import keras
from keras.layers import Input, Reshape, Embedding, Concatenate, Multiply, Add, Lambda, Flatten, Dense, Dropout
from keras.models import Model
from keras.layers import Layer
from keras import backend as K

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
                self.add_weight(shape=[1, self.input_shape],
                                initializer=keras.initializers.truncated_normal(stddev=0.1), name='w_' + str(i),
                                regularizer=keras.regularizers.l2(self.cross_reg),
                                trainable=True))
            self.bias.append(
                self.add_weight(shape=[1, self.input_dim], initializer=keras.initializers.zero, name='b_' + str(i),
                                trainable=True)
            )
        self.built = True

    def call(self, x):
        cross = None
        for i in range(self.num_layer):
            cross = None
            if i == 0:
                cross = Lambda(lambda x: Add()(
                    [K.sum(self.W[i] * K.batch_dot(K.reshape(x, (-1, self.input_dim, 1)), x), 1, keepdims=True),
                     self.bias[i], x]))(x)
            else:
                cross = Lambda(lambda x: Add()(
                    [K.sum(self.W[i] * K.batch_dot(K.reshape(x, (-1, self.input_dim, 1)), x), 1, keepdims=True),
                     self.bias[i], x]))(cross)
        return Flatten()(cross)

    def compute_output_shape(self, input_shape):
        return None, self.output_dim


class DeepCrossNetwork(object):
    def __init__(self, field_dim, feature_dim, embedding_size, cross_layer_num,
                 hidden_size, init_std, seed, embed_reg, cross_reg, dense_reg, keep_prob):
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
        self.keep_prob = keep_prob

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
        real_value_input = Input(shape=(self.field_dim[0],))
        discrete_input = Input(shape=(self.field_dim[1],))
        embeddings = Embedding(self.feature_dim + 1, self.embedding_size,
                               embeddings_initializer=keras.initializers.truncated_normal(stddev=self.init_std),
                               embeddings_regularizer=keras.regularizers.l2(self.embed_reg))(discrete_input)
        reshape = Reshape(target_shape=(-1,))(embeddings)
        features = Concatenate(axis=1)([real_value_input, reshape])
        dense_network_out = Lambda(self.dense_loop)(features)
        cross_network_out = CrossLayer(features.shape[1], self.cross_layer_num, self.cross_reg)(features)
        # concat = Concatenate(axis=1)([dense_network_out, cross_network_out])

        return Model(inputs=[real_value_input, discrete_input], outputs=[cross_network_out])


if __name__ == '__main__':
    file_path = '../data/train.txt'
    featmap, train_real_value, train_discrete, train_labels, test_real_value, test_discrete, test_labels = read_input(
        file_path)
    dcn = DeepCrossNetwork([4, 20], 168, 10, 2, [10, 10, 10], 0.1, 1024, 1e-3, 1e-3, 1e-3, 0.5)
    model = dcn.build_model()
    print(model.predict([train_real_value, train_discrete]))
