from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor as to_T

from util.cnn import fc_layer as fc, conv_layer as conv
from util.empty_safe_conv import empty_safe_1x1_conv as _1x1_conv
from util.empty_safe_conv import empty_safe_conv as _conv

def add_spatial_coordinate_map(image_feat_grid):
    image_feat_shape = tf.shape(image_feat_grid)
    N = image_feat_shape[0]
    H = image_feat_shape[1]
    W = image_feat_shape[2]
    x_map = tf.tile(
        tf.reshape(tf.linspace(-1., 1., W), [1, 1, -1, 1]),
        to_T([N, H, 1, 1]))
    y_map = tf.tile(
        tf.reshape(tf.linspace(-1., 1., H), [1, -1, 1, 1]),
        to_T([N, 1, W, 1]))
    # stop gradient on coords_map (needed to fix the tile grad error on TF 1.0.0)
    coords_map = tf.stop_gradient(tf.concat([x_map, y_map], axis=3))
    image_feat_with_coords = tf.concat([image_feat_grid, coords_map], axis=3)
    # set shapes of the new feature maps
    image_feat_static_shape = image_feat_grid.get_shape().as_list()
    image_feat_static_shape[3] += 2
    image_feat_with_coords.set_shape(image_feat_static_shape)
    image_feat_static_shape[3] = 2
    coords_map.set_shape(image_feat_static_shape)
    return image_feat_with_coords, coords_map

class Modules:
    def __init__(self, image_feat_grid, word_vecs, encoder_states, num_choices):
        self.image_feat_grid_with_coords, self.coords_map = \
            add_spatial_coordinate_map(image_feat_grid)
        self.word_vecs = word_vecs
        self.encoder_states = encoder_states
        self.num_choices = num_choices
        # Capture the variable scope for creating all variables
        with tf.variable_scope('module_variables') as module_variable_scope:
            self.module_variable_scope = module_variable_scope
        # Flatten word vecs for efficient slicing
        # word_vecs has shape [T_decoder, N, D]
        word_vecs_shape = tf.shape(word_vecs)
        T_full = word_vecs_shape[0]
        self.N_full = word_vecs_shape[1]
        D_word = word_vecs.get_shape().as_list()[-1]
        self.word_vecs_flat = tf.reshape(
            word_vecs, to_T([T_full*self.N_full, D_word]))

        # create each dummy modules here so that weights won't get initialized again
        att_shape = self.image_feat_grid_with_coords.get_shape().as_list()[:-1] + [1]
        self.att_shape = att_shape
        input_att = tf.placeholder(tf.float32, att_shape)
        time_idx = tf.placeholder(tf.int32, [None])
        batch_idx = tf.placeholder(tf.int32, [None])
        self.FindModule(time_idx, batch_idx, reuse=False)
        self.TransformModule(input_att, time_idx, batch_idx, reuse=False)
        self.AndModule(input_att, input_att, time_idx, batch_idx, reuse=False)
        self.DescribeModule(input_att, time_idx, batch_idx, reuse=False)

    def _slice_image_feat_grid(self, batch_idx):
        # In TF Fold, batch_idx is a [N_batch, 1] tensor
        return tf.gather(self.image_feat_grid_with_coords, batch_idx)

    def _slice_coords_grid(self, batch_idx):
        # In TF Fold, batch_idx is a [N_batch, 1] tensor
        return tf.gather(self.coords_map, batch_idx)

    def _slice_word_vecs(self, time_idx, batch_idx):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors
        # time is highest dim in word_vecs
        joint_index = time_idx*self.N_full + batch_idx
        return tf.gather(self.word_vecs_flat, joint_index)

    def _slice_encoder_states(self, batch_idx):
        # In TF Fold, batch_idx is a [N_batch, 1] tensor
        if self.encoder_states is not None:
            return tf.gather(self.encoder_states, batch_idx)
        else:
            return None

    def FindModule(self, time_idx, batch_idx, map_dim=1024, scope='FindModule',
        reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        image_feat_grid = self._slice_image_feat_grid(batch_idx)
        text_param = self._slice_word_vecs(time_idx, batch_idx)
        # Mapping: image_feat_grid x text_param -> att_grid
        # Input:
        #   image_feat_grid: [N, H, W, D_im]
        #   text_param: [N, D_txt]
        # Output:
        #   att_grid: [N, H, W, 1]
        #
        # Implementation:
        #   1. Elementwise multiplication between image_feat_grid and text_param
        #   2. L2-normalization
        #   3. Linear classification
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                image_shape = tf.shape(image_feat_grid)
                N = tf.shape(time_idx)[0]
                H = image_shape[1]
                W = image_shape[2]
                D_im = image_feat_grid.get_shape().as_list()[-1]
                D_txt = text_param.get_shape().as_list()[-1]

                # image_feat_mapped has shape [N, H, W, map_dim]
                image_feat_mapped = _1x1_conv('conv_image', image_feat_grid,
                                              output_dim=map_dim)

                text_param_mapped = fc('fc_text', text_param, output_dim=map_dim)
                text_param_mapped = tf.reshape(text_param_mapped, to_T([N, 1, 1, map_dim]))

                eltwise_mult = tf.nn.l2_normalize(image_feat_mapped * text_param_mapped, 3)
                att_grid = _1x1_conv('conv_eltwise', eltwise_mult, output_dim=1)

        att_grid.set_shape(self.att_shape)
        return att_grid

    def TransformModule(self, input_0, time_idx, batch_idx, kernel_size=5,
        map_dim=1024, scope='TransformModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        image_feat_grid = self._slice_image_feat_grid(batch_idx)
        text_param = self._slice_word_vecs(time_idx, batch_idx)
        # Mapping: att_grid x text_param -> att_grid
        # Input:
        #   input_0: [N, H, W, 1]
        #   text_param: [N, D_txt]
        # Output:
        #   att_grid: [N, H, W, 1]
        #
        # Implementation (Same as FindSamePropertyModule):
        #   1. Extract visual features using the input attention map, and
        #      linear transform to map_dim
        #   2. linear transform language features to map_dim
        #   3. Convolve image features to map_dim
        #   4. Element-wise multiplication of the three, l2_normalize, linear transform.
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                image_shape = tf.shape(image_feat_grid)
                N = tf.shape(time_idx)[0]
                H = image_shape[1]
                W = image_shape[2]
                D_im = image_feat_grid.get_shape().as_list()[-1]
                D_txt = text_param.get_shape().as_list()[-1]

                # image_feat_mapped has shape [N, H, W, map_dim]
                image_feat_mapped = _1x1_conv('conv_image', image_feat_grid,
                                              output_dim=map_dim)

                text_param_mapped = fc('fc_text', text_param, output_dim=map_dim)
                text_param_mapped = tf.reshape(text_param_mapped, to_T([N, 1, 1, map_dim]))

                att_softmax = tf.reshape(
                    tf.nn.softmax(tf.reshape(input_0, to_T([N, H*W]))),
                    to_T([N, H, W, 1]))
                # att_feat has shape [N, D_vis]
                att_feat = tf.reduce_sum(image_feat_grid * att_softmax, axis=[1, 2])
                att_feat_mapped = tf.reshape(
                    fc('fc_att', att_feat, output_dim=map_dim), to_T([N, 1, 1, map_dim]))

                eltwise_mult = tf.nn.l2_normalize(
                    image_feat_mapped * text_param_mapped * att_feat_mapped, 3)
                att_grid = _1x1_conv('conv_eltwise', eltwise_mult, output_dim=1)

        att_grid.set_shape(self.att_shape)
        return att_grid

    def AndModule(self, input_0, input_1, time_idx, batch_idx,
        scope='AndModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        # Mapping: att_grid x att_grid -> att_grid
        # Input:
        #   input_0: [N, H, W, 1]
        #   input_1: [N, H, W, 1]
        # Output:
        #   att_grid: [N, H, W, 1]
        #
        # Implementation:
        #   Take the elementwise-min
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                att_grid = tf.minimum(input_0, input_1)

        att_grid.set_shape(self.att_shape)
        return att_grid

    def DescribeModule(self, input_0, time_idx, batch_idx,
        map_dim=1024, scope='DescribeModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        image_feat_grid = self._slice_image_feat_grid(batch_idx)
        text_param = self._slice_word_vecs(time_idx, batch_idx)
        encoder_states = self._slice_encoder_states(batch_idx)
        # Mapping: att_grid -> answer probs
        # Input:
        #   input_0: [N, H, W, 1]
        # Output:
        #   answer_scores: [N, self.num_choices]
        #
        # Implementation:
        #   1. Extract visual features using the input attention map, and
        #      linear transform to map_dim
        #   2. linear transform language features to map_dim
        #   3. Element-wise multiplication of the two, l2_normalize, linear transform.
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                image_shape = tf.shape(image_feat_grid)
                N = tf.shape(time_idx)[0]
                H = image_shape[1]
                W = image_shape[2]
                D_im = image_feat_grid.get_shape().as_list()[-1]
                D_txt = text_param.get_shape().as_list()[-1]

                text_param_mapped = fc('fc_text', text_param, output_dim=map_dim)

                att_softmax = tf.reshape(
                    tf.nn.softmax(tf.reshape(input_0, to_T([N, H*W]))),
                    to_T([N, H, W, 1]))

                # att_feat, att_feat_1 has shape [N, D_vis]
                att_feat = tf.reduce_sum(image_feat_grid * att_softmax, axis=[1, 2])
                att_feat_mapped = tf.reshape(
                    fc('fc_att', att_feat, output_dim=map_dim),
                    to_T([N, map_dim]))

                if encoder_states is not None:
                    # Add in encoder states in the elementwise multiplication
                    encoder_states_mapped = fc('fc_encoder_states', encoder_states, output_dim=map_dim)
                    eltwise_mult = tf.nn.l2_normalize(text_param_mapped * att_feat_mapped * encoder_states_mapped, 1)
                else:
                    eltwise_mult = tf.nn.l2_normalize(text_param_mapped * att_feat_mapped, 1)
                scores = fc('fc_eltwise', eltwise_mult, output_dim=self.num_choices)

        return scores
