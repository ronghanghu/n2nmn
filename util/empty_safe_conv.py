from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import convert_to_tensor as to_T

from util.cnn import conv_layer as conv

def empty_safe_1x1_conv(name, bottom, output_dim, reuse=None):
    # TensorFlow Fold can generate zero-size batch for conv layer
    # which will crash cuDNN on backward pass. So use this
    # for 1x1 convolution in modules to avoid the crash.
    bottom_shape = tf.shape(bottom)
    N = bottom_shape[0]
    H = bottom_shape[1]
    W = bottom_shape[2]
    input_dim = bottom.get_shape().as_list()[-1]
    bottom_flat = tf.reshape(bottom, [-1, input_dim])

    # weights and biases variables
    with tf.variable_scope(name, reuse=reuse):
        # initialize the variables
        weights_initializer = tf.contrib.layers.xavier_initializer()
        biases_initializer = tf.constant_initializer(0.)
        weights = tf.get_variable('weights', [input_dim, output_dim],
            initializer=weights_initializer)
        biases = tf.get_variable('biases', output_dim,
            initializer=biases_initializer)

        conv_flat = tf.nn.xw_plus_b(bottom_flat, weights, biases)
        conv = tf.reshape(conv_flat, to_T([N, H, W, output_dim]))

    return conv

# TensorFlow Fold can generate zero-size batch for conv layer
# which will crash cuDNN on backward pass. So use this
# for arbitrary convolution in modules to avoid the crash.
def empty_safe_conv(name, bottom, kernel_size, stride, output_dim, padding='SAME',
          bias_term=True, weights_initializer=None,
          biases_initializer=None, reuse=None):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Conv2D': 'Conv2D_handle_empty_batch'}):
        return conv(name, bottom, kernel_size, stride, output_dim,
                    padding, bias_term, weights_initializer,
                    biases_initializer, reuse=reuse)

@tf.RegisterGradient('Conv2D_handle_empty_batch')
def _Conv2DGrad(op, grad):
    with tf.device('/cpu:0'):
        return [tf.nn.conv2d_backprop_input(
                tf.shape(op.inputs[0]), op.inputs[1], grad, op.get_attr('strides'),
                op.get_attr('padding'), op.get_attr('use_cudnn_on_gpu'),
                op.get_attr('data_format')),
                tf.nn.conv2d_backprop_filter(op.inputs[0],
                                             tf.shape(op.inputs[1]), grad,
                                             op.get_attr('strides'),
                                             op.get_attr('padding'),
                                             op.get_attr('use_cudnn_on_gpu'),
                                             op.get_attr('data_format'))]
# @tf.RegisterGradient('Conv2D_handle_empty_batch')
# def _Conv2DGrad(op, grad):
#     def _input_nonempty():
#         return tf.nn.conv2d_backprop_input(
#             tf.shape(op.inputs[0]), op.inputs[1], grad, op.get_attr('strides'),
#             op.get_attr('padding'), op.get_attr('use_cudnn_on_gpu'),
#             op.get_attr('data_format'))
#     def _filter_nonempty():
#         return tf.nn.conv2d_backprop_filter(op.inputs[0],
#                                             tf.shape(op.inputs[1]), grad,
#                                             op.get_attr('strides'),
#                                             op.get_attr('padding'),
#                                             op.get_attr('use_cudnn_on_gpu'),
#                                             op.get_attr('data_format'))
#     def _input_empty():
#         return tf.zeros_like(op.inputs[0])
#     def _filter_empty():
#         return tf.zeros_like(op.inputs[1])
#     is_nonempty = tf.greater(tf.size(op.inputs[0]), 0)
#     return [tf.cond(is_nonempty, _input_nonempty, _input_empty),
#             tf.cond(is_nonempty, _filter_nonempty, _filter_empty)]
