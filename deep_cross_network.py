# coding:utf-8
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, l2_regularizer

def tf_weighted_sigmoid_ce_with_logits(labels=None, logits=None, sample_weight=None):
    return tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits), sample_weight)


class DeepCrossNetwork(object):
    def __init__(self, filed_dim, feature_dim, embedding_size=64, cross_layer_num=1,
                 hidden_size=None, use_batch_norm=True, deep_l2_reg=0.0, sample_weight=0.5,
                 init_std=0.01, seed=1024, keep_prob=0.5):
        if hidden_size is None:
            hidden_size = []
        self.filed_dim = filed_dim
        self.feature_dim = feature_dim
        self.embedding_size = embedding_size
        self.deep_l2_reg = deep_l2_reg
        self.sample_weight = sample_weight
        self.init_std = init_std
        self.keep_prob = keep_prob
        self.seed = seed
        self.cross_layer_num = cross_layer_num
        self.hidden_size = hidden_size
        self.use_batch_norm = use_batch_norm

    def f_cross_l(self, x_l, w_l, b_l):
        _dot = tf.matmul(self.x_0, x_l, transpose_b=True)
        return tf.nn.xw_plus_b(_dot, w_l, b_l)

    def _create_placeholder(self, ):
        with tf.name_scope('placeholder'):
            self.X = tf.placeholder(
                dtype=tf.int32, shape=[None, self.filed_dim], name='input_X')
            self.y = tf.placeholder(tf.float32, shape=[None, ], name='input_y')
            self.train_flag = tf.placeholder(tf.bool, name='train_flag')

    def _create_variable(self, ):

        self.total_size = self.filed_dim * self.embedding_size
        with tf.name_scope('embedding'):
            self.embedding = tf.Variable(tf.truncated_normal(
                [self.feature_dim, self.embedding_size], stddev=self.init_std, seed=self.seed),
                name='cross_embed_weight')

        with tf.name_scope('cross_layer_weight'):
            self.cross_layer_weight = [
                tf.Variable(tf.truncated_normal(
                    [self.total_size, 1], stddev=self.init_std, seed=self.seed),
                    name='cross_layer_weight_' + str(i)) for i in range(self.cross_layer_num)]

        with tf.name_scope('cross_layer_bias'):
            self.cross_layer_bias = [
                tf.Variable(tf.constant(0.0, tf.float32, [self.total_size, 1]),
                            name='cross_layer_bias_' + str(i)) for i in range(self.cross_layer_num)]

    def _forward_pass(self, ):
        fc_input = None

        def inverted_dropout(fc, keep_prob):
            return tf.divide(tf.nn.dropout(fc, keep_prob), keep_prob)

        with tf.name_scope('cross_network'):
            with tf.name_scope('embeddings'):
                embeddings = tf.nn.embedding_lookup(
                    self.embedding, self.X, partition_strategy='div')
            self.x_0 = tf.reshape(embeddings, (-1, self.total_size, 1))
            x_l = self.x_0
            for l in range(self.cross_layer_num):
                x_l = self.f_cross_l(x_l, self.cross_layer_weight[l], self.cross_layer_bias[l]) + x_l
            cross_network_out = tf.reshape(x_l, (-1, self.total_size))

        with tf.name_scope('deep_network'):
            if len(self.hidden_size) > 0:
                fc_input = tf.reshape(embeddings, (-1, self.total_size))
                for l in range(len(self.hidden_size)):
                    if self.use_batch_norm:
                        weight = tf.get_variable(
                            name='deep_weight_' + str(l), shape=[fc_input.get_shape().aslist()[1], self.hidden_size[l]])
                        bias = tf.get_variable(
                            name='deep_bias_' + str(l), shape=[fc_input.get_shape().aslist()[1], 1],
                            initializer=tf.zeros_initializer)
                        h_l = tf.nn.xw_plus_b(fc_input, weight, bias)
                        # h_l_bn = tf.nn.batch_normalization(h_l,)
                        h_l_bn = tf.layers.batch_normalization(h_l, training=self.train_flag)
                        fc = tf.nn.relu(h_l_bn)
                    else:
                        fc = fully_connected(fc_input, self.hidden_size[l],
                                             activation_fn=tf.nn.relu,
                                             weights_initializer=tf.truncated_normal_initializer(
                                                 stddev=self.init_std),
                                             weights_regularizer=l2_regularizer(self.deep_l2_reg))
                if l < len(self.hidden_size) - 1:
                    fc = tf.cond(self.train_flag, lambda: inverted_dropout(
                        fc, self.keep_prob), lambda: fc)
                fc_input = fc
            deep_network_out = fc_input

        with tf.name_scope('combination_output_layer'):
            x_stack = cross_network_out
            if len(self.hidden_size) > 0:
                x_stack = tf.concat([x_stack, deep_network_out], axis=1)
            self.logit = fully_connected(x_stack, 1, activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=self.init_std),
                                         weights_regularizer=None)
            self.logit = tf.reshape(self.logit, (-1,))

    def _create_loss(self, ):
        self.log_loss = tf.reduce_sum(tf_weighted_sigmoid_ce_with_logits(labels=self.y, logits=self.logit,
                                                                         sample_weight=self.sample_weight))
        self.loss = self.log_loss
