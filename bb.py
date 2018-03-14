# coding:utf-8
import keras
# from keras.layers import Embedding
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Lambda, concatenate, Reshape,Masking
from keras.layers.core import Masking

import tensorflow as tf


def mean(x, axis=1):
    return K.mean(x, axis=axis)


def activation(x):
    return keras.activations.relu(x)


def softmax(x):
    def get_array(y):
        none_zero_num = tf.count_nonzero(y, axis=0)
        # params = K.map_fn()
        params = K.reshape(K.map_fn(lambda i: K.exp(-i), elems=K.arange(none_zero_num, dtype=tf.float32)),
                           (none_zero_num, 1))
        return K.sum(tf.multiply(params, y), axis=1)

    K.map_fn(lambda y: get_array(y), elems=x, dtype=tf.float32)


a = np.array([[2, 0, 0, 0], [3, 4, 5, 0]], dtype=np.int32)
print(a.shape)
b = np.array([[1, 3, 1, 0], [1, 5, 3, 0]], dtype=np.int32)

input1 = Input(shape=(4,))
input2 = Input(shape=(4,))

e1 = Embedding(6, 4, embeddings_initializer=keras.initializers.truncated_normal(stddev=0.1), mask_zero=True)(input1)
e2 = Embedding(8, 4, embeddings_initializer=keras.initializers.glorot_normal(seed=1024), mask_zero=True)(input2)
e3 = Masking(mask_value=0,)

# mean1 = Lambda(mean)(e1)
mean1 = Reshape((-1,1))(e1)
mean2 = Lambda(mean)(e2)
# mean2 = Lambda(K.mean,output_shape=(2,))(e2)
# output = concatenate([mean1,mean2],axis=1)
# mean2 = Lambda(mean)(e2)
# mean2 = K.mean(e2,axis=1)
# c = Lambda
# concat = K.concatenate([mean1,mean2],axis=1)
model = Model(inputs=[input1], outputs=[e1,mean1])
print(model.predict(a)[0])
print(model.predict(a)[1])

