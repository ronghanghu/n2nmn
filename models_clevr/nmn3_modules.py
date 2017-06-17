from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor as to_T

from util.cnn import fc_layer as fc, conv_layer as conv
from util.empty_safe_conv import empty_safe_1x1_conv as _1x1_conv
from util.empty_safe_conv import empty_safe_conv as _conv

class Modules:
    def __init__(self, image_feat_grid, word_vecs, num_choices):
        self.image_feat_grid = image_feat_grid
        self.word_vecs = word_vecs
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
        att_shape = image_feat_grid.get_shape().as_list()[:-1] + [1]
        self.att_shape = att_shape
        input_att = tf.placeholder(tf.float32, att_shape)
        time_idx = tf.placeholder(tf.int32, [None])
        batch_idx = tf.placeholder(tf.int32, [None])
        self.SceneModule(time_idx, batch_idx, reuse=False)
        self.FindModule(time_idx, batch_idx, reuse=False)
        self.FindSamePropertyModule(input_att, time_idx, batch_idx, reuse=False)
        self.TransformModule(input_att, time_idx, batch_idx, reuse=False)
        self.AndModule(input_att, input_att, time_idx, batch_idx, reuse=False)
        self.FilterModule(input_att, time_idx, batch_idx, reuse=False)
        self.OrModule(input_att, input_att, time_idx, batch_idx, reuse=False)
        self.ExistModule(input_att, time_idx, batch_idx, reuse=False)
        self.CountModule(input_att, time_idx, batch_idx, reuse=False)
        self.EqualNumModule(input_att, input_att, time_idx, batch_idx, reuse=False)
        self.MoreNumModule(input_att, input_att, time_idx, batch_idx, reuse=False)
        self.LessNumModule(input_att, input_att, time_idx, batch_idx, reuse=False)
        self.SamePropertyModule(input_att, input_att, time_idx, batch_idx, reuse=False)
        self.DescribeModule(input_att, time_idx, batch_idx, reuse=False)

    def _slice_image_feat_grid(self, batch_idx):
        # In TF Fold, batch_idx is a [N_batch, 1] tensor
        return tf.gather(self.image_feat_grid, batch_idx)

    def _slice_word_vecs(self, time_idx, batch_idx):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors
        # time is highest dim in word_vecs
        joint_index = time_idx*self.N_full + batch_idx
        return tf.gather(self.word_vecs_flat, joint_index)

    # All the layers are wrapped with td.ScopedLayer
    def SceneModule(self, time_idx, batch_idx, pos_val=3, scope='SceneModule',
        reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        # Mapping: None -> att_grid
        # Output:
        #   att_grid: [N, H, W, 1]
        #
        # Implementation:
        #   1. Just output a positive attention everywhere
        N = tf.shape(time_idx)[0]
        att_grid = pos_val*tf.ones(to_T([N]+self.att_shape[1:]))
        return att_grid

    def FindModule(self, time_idx, batch_idx, map_dim=250, scope='FindModule',
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

    def FilterModule(self, input_0, time_idx, batch_idx, map_dim=250,
        scope='FilterModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors
        image_feat_grid = self._slice_image_feat_grid(batch_idx)
        text_param = self._slice_word_vecs(time_idx, batch_idx)

        # Mapping: att_grid x image_feat_grid x text_param -> att_grid
        # Input:
        #   input_0: [N, H, W, 1]
        #   image_feat_grid: [N, H, W, D_im]
        #   text_param: [N, D_txt]
        # Output:
        #   att_grid: [N, H, W, 1]
        #
        # Implementation:
        #   This is just Find + And
        find_result = self.FindModule(time_idx, batch_idx, reuse=True)
        att_grid = self.AndModule(input_0, find_result, None, None, reuse=True)
        att_grid.set_shape(input_0.get_shape())
        return att_grid

    def FindSamePropertyModule(self, input_0, time_idx, batch_idx, map_dim=250,
        scope='FindSamePropertyModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        image_feat_grid = self._slice_image_feat_grid(batch_idx)
        text_param = self._slice_word_vecs(time_idx, batch_idx)
        # Mapping: att_grid x image_feat_grid x text_param -> att_grid
        # Input:
        #   input_0: [N, H, W, 1]
        #   image_feat_grid: [N, H, W, D_im]
        #   text_param: [N, D_txt]
        # Output:
        #   att_grid: [N, H, W, 1]
        #
        # Implementation:
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

    def TransformModule(self, input_0, time_idx, batch_idx, kernel_size=5,
        map_dim=250, scope='TransformModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        text_param = self._slice_word_vecs(time_idx, batch_idx)
        # Mapping: att_grid x text_param -> att_grid
        # Input:
        #   input_0: [N, H, W, 1]
        #   text_param: [N, D_txt]
        # Output:
        #   att_grid: [N, H, W, 1]
        #
        # Implementation:
        #   Convolutional layer that also involve text_param
        #   A 'soft' convolutional kernel that is modulated by text_param
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                att_shape = tf.shape(input_0)
                N = att_shape[0]
                H = att_shape[1]
                W = att_shape[2]
                att_maps = _conv('conv_maps', input_0, kernel_size=kernel_size,
                    stride=1, output_dim=map_dim)

                text_param_mapped = fc('text_fc', text_param, output_dim=map_dim)
                text_param_mapped = tf.reshape(text_param_mapped, to_T([N, 1, 1, map_dim]))

                eltwise_mult = tf.nn.l2_normalize(att_maps * text_param_mapped, 3)
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

    def OrModule(self, input_0, input_1, time_idx, batch_idx,
        scope='OrModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        # Mapping: att_grid x att_grid -> att_grid
        # Input:
        #   input_0: [N, H, W, 1]
        #   input_1: [N, H, W, 1]
        # Output:
        #   att_grid: [N, H, W, 1]
        #
        # Implementation:
        #   Take the elementwise-max
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                att_grid = tf.maximum(input_0, input_1)

        att_grid.set_shape(self.att_shape)
        return att_grid

    def ExistModule(self, input_0, time_idx, batch_idx,
        scope='ExistModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        # Mapping: att_grid -> answer probs
        # Input:
        #   att_grid: [N, H, W, 1]
        # Output:
        #   answer_scores: [N, self.num_choices]
        #
        # Implementation:
        #   1. Max-pool over att_grid
        #   2. a linear mapping layer (without ReLU)
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                att_min = tf.reduce_min(input_0, axis=[1, 2])
                att_avg = tf.reduce_mean(input_0, axis=[1, 2])
                att_max = tf.reduce_max(input_0, axis=[1, 2])
                # att_reduced has shape [N, 3]
                att_reduced = tf.concat([att_min, att_avg, att_max], axis=1)
                scores = fc('fc_scores', att_reduced, output_dim=self.num_choices)

        return scores

    def CountModule(self, input_0, time_idx, batch_idx,
        scope='CountModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        # Mapping: att_grid -> answer probs
        # Input:
        #   input_0: [N, H, W, 1]
        # Output:
        #   answer_scores: [N, self.num_choices]
        #
        # Implementation:
        #   1. linear transform of the attention map (also including max and min)
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                H, W = self.att_shape[1:3]
                att_all = tf.reshape(input_0, to_T([-1, H*W]))
                att_min = tf.reduce_min(input_0, axis=[1, 2])
                att_max = tf.reduce_max(input_0, axis=[1, 2])
                # att_reduced has shape [N, 3]
                att_concat = tf.concat([att_all, att_min, att_max], axis=1)
                scores = fc('fc_scores', att_concat, output_dim=self.num_choices)

        return scores

    def EqualNumModule(self, input_0, input_1, time_idx, batch_idx,
        scope='EqualNumModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        # Mapping: att_grid x att_grid -> answer probs
        # Input:
        #   input_0: [N, H, W, 1]
        #   input_1: [N, H, W, 1]
        # Output:
        #   answer_scores: [N, self.num_choices]
        #
        # Implementation:
        #   1. linear transform of the attention map (also including max and min)
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                att_shape = tf.shape(input_0)

                H, W = self.att_shape[1:3]
                att_all_0 = tf.reshape(input_0, to_T([-1, H*W]))
                att_min_0 = tf.reduce_min(input_0, axis=[1, 2])
                att_max_0 = tf.reduce_max(input_0, axis=[1, 2])
                att_all_1 = tf.reshape(input_1, to_T([-1, H*W]))
                att_min_1 = tf.reduce_min(input_1, axis=[1, 2])
                att_max_1 = tf.reduce_max(input_1, axis=[1, 2])
                # att_reduced has shape [N, 3]
                att_concat = tf.concat([att_all_0, att_min_0, att_max_0,
                                        att_all_1, att_min_1, att_max_1],
                                       axis=1)
                scores = fc('fc_scores', att_concat, output_dim=self.num_choices)

        return scores

    def MoreNumModule(self, input_0, input_1, time_idx, batch_idx,
        scope='MoreNumModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        # Mapping: att_grid x att_grid -> answer probs
        # Input:
        #   input_0: [N, H, W, 1]
        #   input_1: [N, H, W, 1]
        # Output:
        #   answer_scores: [N, self.num_choices]
        #
        # Implementation:
        #   1. linear transform of the attention map (also including max and min)
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                att_shape = tf.shape(input_0)

                H, W = self.att_shape[1:3]
                att_all_0 = tf.reshape(input_0, to_T([-1, H*W]))
                att_min_0 = tf.reduce_min(input_0, axis=[1, 2])
                att_max_0 = tf.reduce_max(input_0, axis=[1, 2])
                att_all_1 = tf.reshape(input_1, to_T([-1, H*W]))
                att_min_1 = tf.reduce_min(input_1, axis=[1, 2])
                att_max_1 = tf.reduce_max(input_1, axis=[1, 2])
                # att_reduced has shape [N, 3]
                att_concat = tf.concat([att_all_0, att_min_0, att_max_0,
                                        att_all_1, att_min_1, att_max_1],
                                       axis=1)
                scores = fc('fc_scores', att_concat, output_dim=self.num_choices)

        return scores

    def LessNumModule(self, input_0, input_1, time_idx, batch_idx,
        scope='LessNumModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        # Mapping: att_grid x att_grid -> answer probs
        # Input:
        #   input_0: [N, H, W, 1]
        #   input_1: [N, H, W, 1]
        # Output:
        #   answer_scores: [N, self.num_choices]
        #
        # Implementation:
        #   1. linear transform of the attention map (also including max and min)
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                att_shape = tf.shape(input_0)

                H, W = self.att_shape[1:3]
                att_all_0 = tf.reshape(input_0, to_T([-1, H*W]))
                att_min_0 = tf.reduce_min(input_0, axis=[1, 2])
                att_max_0 = tf.reduce_max(input_0, axis=[1, 2])
                att_all_1 = tf.reshape(input_1, to_T([-1, H*W]))
                att_min_1 = tf.reduce_min(input_1, axis=[1, 2])
                att_max_1 = tf.reduce_max(input_1, axis=[1, 2])
                # att_reduced has shape [N, 3]
                att_concat = tf.concat([att_all_0, att_min_0, att_max_0,
                                        att_all_1, att_min_1, att_max_1],
                                       axis=1)
                scores = fc('fc_scores', att_concat, output_dim=self.num_choices)

        return scores

    def SamePropertyModule(self, input_0, input_1, time_idx, batch_idx,
        map_dim=250, scope='SamePropertyModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        image_feat_grid = self._slice_image_feat_grid(batch_idx)
        text_param = self._slice_word_vecs(time_idx, batch_idx)
        # Mapping: att_grid x att_grid -> answer probs
        # Input:
        #   input_0: [N, H, W, 1]
        #   input_1: [N, H, W, 1]
        # Output:
        #   answer_scores: [N, self.num_choices]
        #
        # Implementation:
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

                text_param_mapped = fc('fc_text', text_param, output_dim=map_dim)

                att_softmax_0 = tf.reshape(
                    tf.nn.softmax(tf.reshape(input_0, to_T([N, H*W]))),
                    to_T([N, H, W, 1]))
                att_softmax_1 = tf.reshape(
                    tf.nn.softmax(tf.reshape(input_1, to_T([N, H*W]))),
                    to_T([N, H, W, 1]))
                # att_feat_0, att_feat_1 has shape [N, D_vis]
                att_feat_0 = tf.reduce_sum(image_feat_grid * att_softmax_0, axis=[1, 2])
                att_feat_1 = tf.reduce_sum(image_feat_grid * att_softmax_1, axis=[1, 2])
                att_feat_mapped_0 = tf.reshape(
                    fc('fc_att_0', att_feat_0, output_dim=map_dim),
                    to_T([N, map_dim]))
                att_feat_mapped_1 = tf.reshape(
                    fc('fc_att_1', att_feat_1, output_dim=map_dim),
                    to_T([N, map_dim]))

                eltwise_mult = tf.nn.l2_normalize(
                    att_feat_mapped_0 * text_param_mapped * att_feat_mapped_1, 1)
                scores = fc('fc_eltwise', eltwise_mult, output_dim=self.num_choices)

        return scores

    def DescribeModule(self, input_0, time_idx, batch_idx,
        map_dim=250, scope='DescribeModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        image_feat_grid = self._slice_image_feat_grid(batch_idx)
        text_param = self._slice_word_vecs(time_idx, batch_idx)
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

                eltwise_mult = tf.nn.l2_normalize(text_param_mapped * att_feat_mapped, 1)
                scores = fc('fc_eltwise', eltwise_mult, output_dim=self.num_choices)

        return scores
