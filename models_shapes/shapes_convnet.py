from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import convert_to_tensor as to_T

from util.cnn import conv_relu_layer as conv_relu

def shapes_convnet(input_batch, hidden_dim=64, output_dim=64,
    scope='shapes_convnet', reuse=None):
    # input_batch has shape [N, H_im, W_im, 3]
    with tf.variable_scope(scope, reuse=reuse):
        conv_1 = conv_relu('conv_1', input_batch, kernel_size=10, stride=10,
            output_dim=hidden_dim, padding='VALID')
        conv_2 = conv_relu('conv_2', conv_1, kernel_size=1, stride=1,
            output_dim=output_dim)

    return conv_2
