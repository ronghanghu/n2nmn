from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow_fold as td
from tensorflow import convert_to_tensor as to_T

from models_clevr.nmn3_netgen_att import AttentionSeq2Seq
from models_clevr.nmn3_modules import Modules
from models_clevr.nmn3_assembler import INVALID_EXPR

from util.cnn import fc_layer as fc, conv_layer as conv

class NMN3Model:
    def __init__(self, image_feat_grid, text_seq_batch, seq_length_batch,
        T_decoder, num_vocab_txt, embed_dim_txt, num_vocab_nmn,
        embed_dim_nmn, lstm_dim, num_layers, assembler,
        encoder_dropout, decoder_dropout, decoder_sampling,
        num_choices, use_gt_layout=None, gt_layout_batch=None,
        scope='neural_module_network', reuse=None):

        with tf.variable_scope(scope, reuse=reuse):
            # Part 0: Visual feature from CNN
            self.image_feat_grid = image_feat_grid

            # Part 1: Seq2seq RNN to generate module layout tokensa
            with tf.variable_scope('layout_generation'):
                att_seq2seq = AttentionSeq2Seq(text_seq_batch,
                    seq_length_batch, T_decoder, num_vocab_txt,
                    embed_dim_txt, num_vocab_nmn, embed_dim_nmn, lstm_dim,
                    num_layers, assembler, encoder_dropout, decoder_dropout,
                    decoder_sampling, use_gt_layout, gt_layout_batch)
                self.att_seq2seq = att_seq2seq
                predicted_tokens = att_seq2seq.predicted_tokens
                token_probs = att_seq2seq.token_probs
                word_vecs = att_seq2seq.word_vecs
                neg_entropy = att_seq2seq.neg_entropy
                self.atts = att_seq2seq.atts

                self.predicted_tokens = predicted_tokens
                self.token_probs = token_probs
                self.word_vecs = word_vecs
                self.neg_entropy = neg_entropy

                # log probability of each generated sequence
                self.log_seq_prob = tf.reduce_sum(tf.log(token_probs), axis=0)

            # Part 2: Neural Module Network
            with tf.variable_scope('layout_execution'):
                modules = Modules(image_feat_grid, word_vecs, num_choices)
                self.modules = modules
                # Recursion of modules
                att_shape = image_feat_grid.get_shape().as_list()[1:-1] + [1]
                # Forward declaration of module recursion
                att_expr_decl = td.ForwardDeclaration(td.PyObjectType(), td.TensorType(att_shape))
                # _Scene
                case_scene = td.Record([('time_idx', td.Scalar(dtype='int32')),
                                       ('batch_idx', td.Scalar(dtype='int32'))])
                case_scene = case_scene >> td.Function(modules.SceneModule)
                # _Find
                case_find = td.Record([('time_idx', td.Scalar(dtype='int32')),
                                       ('batch_idx', td.Scalar(dtype='int32'))])
                case_find = case_find >> td.Function(modules.FindModule)
                # _Filter
                case_filter = td.Record([('input_0', att_expr_decl()),
                                         ('time_idx', td.Scalar(dtype='int32')),
                                         ('batch_idx', td.Scalar(dtype='int32'))])
                case_filter = case_filter >> td.Function(modules.FilterModule)
                # _FindSameProperty
                case_find_same_property = td.Record([('input_0', att_expr_decl()),
                                                     ('time_idx', td.Scalar(dtype='int32')),
                                                     ('batch_idx', td.Scalar(dtype='int32'))])
                case_find_same_property = case_find_same_property >> \
                    td.Function(modules.FindSamePropertyModule)
                # _Transform
                case_transform = td.Record([('input_0', att_expr_decl()),
                                            ('time_idx', td.Scalar('int32')),
                                            ('batch_idx', td.Scalar('int32'))])
                case_transform = case_transform >> td.Function(modules.TransformModule)
                # _And
                case_and = td.Record([('input_0', att_expr_decl()),
                                      ('input_1', att_expr_decl()),
                                      ('time_idx', td.Scalar('int32')),
                                      ('batch_idx', td.Scalar('int32'))])
                case_and = case_and >> td.Function(modules.AndModule)
                # _Or
                case_or = td.Record([('input_0', att_expr_decl()),
                                     ('input_1', att_expr_decl()),
                                     ('time_idx', td.Scalar('int32')),
                                     ('batch_idx', td.Scalar('int32'))])
                case_or = case_or >> td.Function(modules.OrModule)
                # _Exist
                case_exist = td.Record([('input_0', att_expr_decl()),
                                        ('time_idx', td.Scalar('int32')),
                                        ('batch_idx', td.Scalar('int32'))])
                case_exist = case_exist >> td.Function(modules.ExistModule)
                # _Count
                case_count = td.Record([('input_0', att_expr_decl()),
                                        ('time_idx', td.Scalar('int32')),
                                        ('batch_idx', td.Scalar('int32'))])
                case_count = case_count >> td.Function(modules.CountModule)
                # _EqualNum
                case_equal_num = td.Record([('input_0', att_expr_decl()),
                                            ('input_1', att_expr_decl()),
                                            ('time_idx', td.Scalar('int32')),
                                            ('batch_idx', td.Scalar('int32'))])
                case_equal_num = case_equal_num >> td.Function(modules.EqualNumModule)
                # _MoreNum
                case_more_num = td.Record([('input_0', att_expr_decl()),
                                            ('input_1', att_expr_decl()),
                                            ('time_idx', td.Scalar('int32')),
                                            ('batch_idx', td.Scalar('int32'))])
                case_more_num = case_more_num >> td.Function(modules.MoreNumModule)
                # _LessNum
                case_less_num = td.Record([('input_0', att_expr_decl()),
                                            ('input_1', att_expr_decl()),
                                            ('time_idx', td.Scalar('int32')),
                                            ('batch_idx', td.Scalar('int32'))])
                case_less_num = case_less_num >> td.Function(modules.LessNumModule)
                # _SameProperty
                case_same_property = td.Record([('input_0', att_expr_decl()),
                                                ('input_1', att_expr_decl()),
                                                ('time_idx', td.Scalar('int32')),
                                                ('batch_idx', td.Scalar('int32'))])
                case_same_property = case_same_property >> \
                    td.Function(modules.SamePropertyModule)
                # _Describe
                case_describe = td.Record([('input_0', att_expr_decl()),
                                           ('time_idx', td.Scalar('int32')),
                                           ('batch_idx', td.Scalar('int32'))])
                case_describe = case_describe >> \
                    td.Function(modules.DescribeModule)

                recursion_cases = td.OneOf(td.GetItem('module'), {
                    '_Scene': case_scene,
                    '_Find': case_find,
                    '_Filter': case_filter,
                    '_FindSameProperty': case_find_same_property,
                    '_Transform': case_transform,
                    '_And': case_and,
                    '_Or': case_or})
                att_expr_decl.resolve_to(recursion_cases)

                # For invalid expressions, define a dummy answer
                # so that all answers have the same form
                dummy_scores = td.Void() >> td.FromTensor(np.zeros(num_choices, np.float32))
                output_scores = td.OneOf(td.GetItem('module'), {
                    '_Exist': case_exist,
                    '_Count': case_count,
                    '_EqualNum': case_equal_num,
                    '_MoreNum': case_more_num,
                    '_LessNum': case_less_num,
                    '_SameProperty': case_same_property,
                    '_Describe': case_describe,
                    INVALID_EXPR: dummy_scores})

                # compile and get the output scores
                self.compiler = td.Compiler.create(output_scores)
                self.scores = self.compiler.output_tensors[0]

            # Regularization: Entropy + L2
            self.entropy_reg = tf.reduce_mean(neg_entropy)
            module_weights = [v for v in tf.trainable_variables()
                              if (scope in v.op.name and
                                  v.op.name.endswith('weights'))]
            self.l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in module_weights])
