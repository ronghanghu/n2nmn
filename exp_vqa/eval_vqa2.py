from __future__ import absolute_import, division, print_function

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', required=True)
parser.add_argument('--snapshot_name', required=True)
parser.add_argument('--test_split', required=True)
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()

gpu_id = args.gpu_id  # set GPU id to use
import os; os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import numpy as np
import tensorflow as tf
# Start the session BEFORE importing tensorflow_fold
# to avoid taking up all GPU memory
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True),
    allow_soft_placement=False, log_device_placement=False))
import json

from models_vqa.nmn3_assembler import Assembler
from models_vqa.nmn3_model import NMN3Model
from util.vqa_train.data_reader import DataReader

# Module parameters
H_feat = 14
W_feat = 14
D_feat = 2048
embed_dim_txt = 300
embed_dim_nmn = 300
lstm_dim = 1000
num_layers = 2
T_encoder = 26
T_decoder = 13
N = 50
use_qpn = True
reduce_visfeat_dim = False

exp_name = args.exp_name
snapshot_name = args.snapshot_name
tst_image_set = args.test_split
snapshot_file = './exp_vqa/tfmodel/%s/%s' % (exp_name, snapshot_name)

# Data files
vocab_question_file = './exp_vqa/data/vocabulary_vqa.txt'
vocab_layout_file = './exp_vqa/data/vocabulary_layout.txt'
vocab_answer_file = './exp_vqa/data/answers_vqa.txt'

imdb_file_tst = './exp_vqa/data/imdb_vqa_v2/imdb_%s.npy' % tst_image_set

save_file = './exp_vqa/results/%s/%s.%s.txt' % (exp_name, snapshot_name, tst_image_set)
os.makedirs(os.path.dirname(save_file), exist_ok=True)
eval_output_name = 'vqa_OpenEnded_mscoco_%s_%s_%s_results.json' % (tst_image_set, exp_name, snapshot_name)
eval_output_file = './exp_vqa/eval_outputs/%s/%s' % (exp_name, eval_output_name)
os.makedirs(os.path.dirname(eval_output_file), exist_ok=True)

assembler = Assembler(vocab_layout_file)

data_reader_tst = DataReader(imdb_file_tst, shuffle=False, one_pass=True,
                             batch_size=N,
                             T_encoder=T_encoder,
                             T_decoder=T_decoder,
                             assembler=assembler,
                             vocab_question_file=vocab_question_file,
                             vocab_answer_file=vocab_answer_file)

num_vocab_txt = data_reader_tst.batch_loader.vocab_dict.num_vocab
num_vocab_nmn = len(assembler.module_names)
num_choices = data_reader_tst.batch_loader.answer_dict.num_vocab

# Network inputs
input_seq_batch = tf.placeholder(tf.int32, [None, None])
seq_length_batch = tf.placeholder(tf.int32, [None])
image_feat_batch = tf.placeholder(tf.float32, [None, H_feat, W_feat, D_feat])
expr_validity_batch = tf.placeholder(tf.bool, [None])

# The model for testing
nmn3_model_tst = NMN3Model(
    image_feat_batch, input_seq_batch,
    seq_length_batch, T_decoder=T_decoder,
    num_vocab_txt=num_vocab_txt, embed_dim_txt=embed_dim_txt,
    num_vocab_nmn=num_vocab_nmn, embed_dim_nmn=embed_dim_nmn,
    lstm_dim=lstm_dim, num_layers=num_layers,
    assembler=assembler,
    encoder_dropout=False,
    decoder_dropout=False,
    decoder_sampling=False,
    num_choices=num_choices,
    use_qpn=use_qpn, qpn_dropout=False, reduce_visfeat_dim=reduce_visfeat_dim)

snapshot_saver = tf.train.Saver(max_to_keep=None)  # keep all snapshots
snapshot_saver.restore(sess, snapshot_file)

def run_test(dataset_tst, save_file, eval_output_file):
    if dataset_tst is None:
        return
    print('Running test...')
    layout_correct = 0
    layout_valid = 0
    num_questions = 0
    answer_word_list = dataset_tst.batch_loader.answer_dict.word_list
    # the first word should be <unk> in answer list
    assert(answer_word_list[0] == '<unk>')
    output_qids_answers = []
    for n_q, batch in enumerate(dataset_tst.batches()):
        # set up input and output tensors
        h = sess.partial_run_setup(
            [nmn3_model_tst.predicted_tokens, nmn3_model_tst.scores],
            [input_seq_batch, seq_length_batch, image_feat_batch,
             nmn3_model_tst.compiler.loom_input_tensor, expr_validity_batch])

        # Part 0 & 1: Run Convnet and generate module layout
        tokens = sess.partial_run(h, nmn3_model_tst.predicted_tokens,
            feed_dict={input_seq_batch: batch['input_seq_batch'],
                       seq_length_batch: batch['seq_length_batch'],
                       image_feat_batch: batch['image_feat_batch']})

        # compute the accuracy of the predicted layout
        if dataset_tst.batch_loader.load_gt_layout:
            gt_tokens = batch['gt_layout_batch']
            layout_correct += np.sum(
                np.all(np.logical_or(tokens == gt_tokens,
                                     gt_tokens == assembler.EOS_idx),
                       axis=0))

        # Assemble the layout tokens into network structure
        expr_list, expr_validity_array = assembler.assemble(tokens)
        layout_valid += np.sum(expr_validity_array)
        # Build TensorFlow Fold input for NMN
        expr_feed = nmn3_model_tst.compiler.build_feed_dict(expr_list)
        expr_feed[expr_validity_batch] = expr_validity_array

        # Part 2: Run NMN and learning steps
        scores_val = sess.partial_run(h, nmn3_model_tst.scores, feed_dict=expr_feed)
        scores_val[:, 0] = -1e10  # remove <unk> answer

        # compute accuracy
        predictions = np.argmax(scores_val, axis=1)
        if dataset_tst.batch_loader.load_answer:
            labels = batch['answer_label_batch']
        num_questions += len(expr_validity_array)

        qid_list = batch['qid_list']
        output_qids_answers += [{'question_id': int(qid), 'answer': answer_word_list[p]}
                                for qid, p in zip(qid_list, predictions)]

    layout_accuracy = layout_correct / num_questions
    layout_validity = layout_valid / num_questions
    print('On split: %s' % tst_image_set)
    print('\tlayout accuracy = %f (%d / %d)' %
          (layout_accuracy, layout_correct, num_questions))
    print('\tlayout validity = %f (%d / %d)' %
          (layout_validity, layout_valid, num_questions))
    # write the results to file
    with open(save_file, 'w') as f:
        print('On split: %s' % tst_image_set, file=f)
        print('\tlayout accuracy = %f (%d / %d)' %
              (layout_accuracy, layout_correct, num_questions), file=f)
        print('\tlayout validity = %f (%d / %d)' %
              (layout_validity, layout_valid, num_questions), file=f)
    with open(eval_output_file, 'w') as f:
        json.dump(output_qids_answers, f, separators=(',\n', ':\n'))
        print('prediction file written to', eval_output_file)

run_test(data_reader_tst, save_file, eval_output_file)
