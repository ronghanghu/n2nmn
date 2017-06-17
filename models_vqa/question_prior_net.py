from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import convert_to_tensor as to_T

from tensorflow.python.ops.nn import dropout as drop
from util.cnn import fc_layer as fc, fc_relu_layer as fc_relu

# The network that takes in the hidden state of the
def question_prior_net(encoder_states, num_choices, qpn_dropout, hidden_dim=500,
    scope='question_prior_net', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # concate the LSTM states from all layers
        assert(isinstance(encoder_states, tuple))
        h_list = []
        for s in encoder_states:
            assert(isinstance(s, tf.contrib.rnn.LSTMStateTuple))
            h_list.append(s.h)
        # h_concat has shape [N, D_lstm1 + ... + D_lstm_n]
        h_concat = tf.concat(h_list, axis=1)

        if qpn_dropout:
            h_concat = drop(h_concat, 0.5)
        fc1 = fc_relu('fc1', h_concat, output_dim=hidden_dim)
        if qpn_dropout:
            fc1 = drop(fc1, 0.5)
        fc2 = fc('fc2', fc1, output_dim=num_choices)
        return fc2
