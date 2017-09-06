from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow_fold as td
from tensorflow import convert_to_tensor as to_T

from models_vqa.nmn3_netgen_att import AttentionSeq2Seq
from models_vqa.nmn3_modules import Modules
from models_vqa.nmn3_assembler import INVALID_EXPR
from models_vqa.question_prior_net import question_prior_net

from util.cnn import fc_layer as fc, conv_layer as conv

class NMN3Model:
    def __init__(self, image_feat_grid, text_seq_batch, seq_length_batch,
        T_decoder, num_vocab_txt, embed_dim_txt, num_vocab_nmn,
        embed_dim_nmn, lstm_dim, num_layers, assembler,
        encoder_dropout, decoder_dropout, decoder_sampling,
        num_choices, use_qpn, qpn_dropout, reduce_visfeat_dim=False, new_visfeat_dim=256,
        use_gt_layout=None, gt_layout_batch=None,
        scope='neural_module_network', reuse=None):

        with tf.variable_scope(scope, reuse=reuse):
            # Part 0: Visual feature from CNN
            self.reduce_visfeat_dim = reduce_visfeat_dim
            if reduce_visfeat_dim:
                # use an extrac linear 1x1 conv layer (without ReLU)
                # to reduce the feature dimension
                with tf.variable_scope('reduce_visfeat_dim'):
                    image_feat_grid = conv('conv_reduce_visfeat_dim',
                        image_feat_grid, kernel_size=1, stride=1,
                        output_dim=new_visfeat_dim)
                print('visual feature dimension reduced to %d' % new_visfeat_dim)
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
                modules = Modules(image_feat_grid, word_vecs, None, num_choices)
                self.modules = modules
                # Recursion of modules
                att_shape = image_feat_grid.get_shape().as_list()[1:-1] + [1]
                # Forward declaration of module recursion
                att_expr_decl = td.ForwardDeclaration(td.PyObjectType(), td.TensorType(att_shape))
                # _Find
                case_find = td.Record([('time_idx', td.Scalar(dtype='int32')),
                                       ('batch_idx', td.Scalar(dtype='int32'))])
                case_find = case_find >> td.Function(modules.FindModule)
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
                # _Describe
                case_describe = td.Record([('input_0', att_expr_decl()),
                                           ('time_idx', td.Scalar('int32')),
                                           ('batch_idx', td.Scalar('int32'))])
                case_describe = case_describe >> \
                    td.Function(modules.DescribeModule)

                recursion_cases = td.OneOf(td.GetItem('module'), {
                    '_Find': case_find,
                    '_Transform': case_transform,
                    '_And': case_and})
                att_expr_decl.resolve_to(recursion_cases)

                # For invalid expressions, define a dummy answer
                # so that all answers have the same form
                dummy_scores = td.Void() >> td.FromTensor(np.zeros(num_choices, np.float32))
                output_scores = td.OneOf(td.GetItem('module'), {
                    '_Describe': case_describe,
                    INVALID_EXPR: dummy_scores})

                # compile and get the output scores
                self.compiler = td.Compiler.create(output_scores)
                self.scores_nmn = self.compiler.output_tensors[0]

            # Add a question prior network if specified
            self.use_qpn = use_qpn
            self.qpn_dropout = qpn_dropout
            if use_qpn:
                self.scores_qpn = question_prior_net(att_seq2seq.encoder_states,
                                                     num_choices, qpn_dropout)
                self.scores = self.scores_nmn + self.scores_qpn
            else:
                self.scores = self.scores_nmn

            # Regularization: Entropy + L2
            self.entropy_reg = tf.reduce_mean(neg_entropy)
            module_weights = [v for v in tf.trainable_variables()
                              if (scope in v.op.name and
                                  v.op.name.endswith('weights'))]
            self.l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in module_weights])
