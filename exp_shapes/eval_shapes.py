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

from models_shapes.nmn3_assembler import Assembler
from models_shapes.nmn3_model import NMN3ModelAtt

# Module parameters
H_im = 30
W_im = 30
num_choices = 2
embed_dim_txt = 300
embed_dim_nmn = 300
lstm_dim = 256
num_layers = 2
encoder_dropout = False
decoder_dropout = False
decoder_sampling = False
T_encoder = 15
T_decoder = 11
N = 256

exp_name = args.exp_name
snapshot_name = args.snapshot_name
snapshot_file = './exp_shapes/tfmodel/%s/%s' % (exp_name, snapshot_name)

# Data files
vocab_shape_file = './exp_shapes/data/vocabulary_shape.txt'
vocab_layout_file = './exp_shapes/data/vocabulary_layout.txt'
image_sets = args.test_split.split(':')
training_text_files = './exp_shapes/shapes_dataset/%s.query_str.txt'
training_image_files = './exp_shapes/shapes_dataset/%s.input.npy'
training_label_files = './exp_shapes/shapes_dataset/%s.output'
training_gt_layout_file = './exp_shapes/data/%s.query_layout_symbols.json'
image_mean_file = './exp_shapes/data/image_mean.npy'

save_dir = './exp_shapes/results/%s/%s.%s' % (exp_name, snapshot_name, '_'.join(image_sets))
save_file = save_dir + '.txt'
os.makedirs(save_dir, exist_ok=True)


# Load vocabulary
with open(vocab_shape_file) as f:
    vocab_shape_list = [s.strip() for s in f.readlines()]
vocab_shape_dict = {vocab_shape_list[n]:n for n in range(len(vocab_shape_list))}
num_vocab_txt = len(vocab_shape_list)

assembler = Assembler(vocab_layout_file)
num_vocab_nmn = len(assembler.module_names)

# Load training data
training_questions = []
training_labels = []
training_images_list = []
gt_layout_list = []

for image_set in image_sets:
    with open(training_text_files % image_set) as f:
        training_questions += [l.strip() for l in f.readlines()]
    with open(training_label_files % image_set) as f:
        training_labels += [l.strip() == 'true' for l in f.readlines()]
    training_images_list.append(np.load(training_image_files % image_set))
    with open(training_gt_layout_file % image_set) as f:
            gt_layout_list += json.load(f)

num_questions = len(training_questions)
training_images = np.concatenate(training_images_list)

# Shuffle the training data
# fix random seed for data repeatibility
np.random.seed(3)
shuffle_inds = np.random.permutation(num_questions)

training_questions = [training_questions[idx] for idx in shuffle_inds]
training_labels = [training_labels[idx] for idx in shuffle_inds]
training_images = training_images[shuffle_inds]
gt_layout_list = [gt_layout_list[idx] for idx in shuffle_inds]

# number of training batches
num_batches = np.ceil(num_questions / N)

# Turn the questions into vocabulary indices
text_seq_array = np.zeros((T_encoder, num_questions), np.int32)
seq_length_array = np.zeros(num_questions, np.int32)
gt_layout_array = np.zeros((T_decoder, num_questions), np.int32)
for n_q in range(num_questions):
    tokens = training_questions[n_q].split()
    seq_length_array[n_q] = len(tokens)
    for t in range(len(tokens)):
        text_seq_array[t, n_q] = vocab_shape_dict[tokens[t]]
    gt_layout_array[:, n_q] = assembler.module_list2tokens(
        gt_layout_list[n_q], T_decoder)

image_mean = np.load(image_mean_file)
image_array = (training_images - image_mean).astype(np.float32)
vqa_label_array = np.array(training_labels, np.int32)

# Network inputs
text_seq_batch = tf.placeholder(tf.int32, [None, None])
seq_length_batch = tf.placeholder(tf.int32, [None])
image_batch = tf.placeholder(tf.float32, [None, H_im, W_im, 3])
expr_validity_batch = tf.placeholder(tf.bool, [None])

# The model
nmn3_model = NMN3ModelAtt(image_batch, text_seq_batch,
    seq_length_batch, T_decoder=T_decoder,
    num_vocab_txt=num_vocab_txt, embed_dim_txt=embed_dim_txt,
    num_vocab_nmn=num_vocab_nmn, embed_dim_nmn=embed_dim_nmn,
    lstm_dim=lstm_dim,
    num_layers=num_layers, EOS_idx=assembler.EOS_idx,
    encoder_dropout=encoder_dropout,
    decoder_dropout=decoder_dropout,
    decoder_sampling=decoder_sampling,
    num_choices=num_choices)

compiler = nmn3_model.compiler
scores = nmn3_model.scores

snapshot_saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
snapshot_saver.restore(sess, snapshot_file)

answer_correct = 0
layout_correct = 0
layout_valid = 0
for n_iter in range(int(num_batches)):
    n_begin = int((n_iter % num_batches)*N)
    n_end = int(min(n_begin+N, num_questions))

    # set up input and output tensors
    h = sess.partial_run_setup(
        [nmn3_model.predicted_tokens, scores],
        [text_seq_batch, seq_length_batch, image_batch,
         compiler.loom_input_tensor, expr_validity_batch])

    # Part 0 & 1: Run Convnet and generate module layout
    tokens = sess.partial_run(h, nmn3_model.predicted_tokens,
        feed_dict={text_seq_batch: text_seq_array[:, n_begin:n_end],
                   seq_length_batch: seq_length_array[n_begin:n_end],
                   image_batch: image_array[n_begin:n_end]})

    # compute the accuracy of the predicted layout
    gt_tokens = gt_layout_array[:, n_begin:n_end]
    layout_correct += np.sum(np.all(np.logical_or(tokens == gt_tokens,
                                                  gt_tokens == assembler.EOS_idx),
                                    axis=0))

    # Assemble the layout tokens into network structure
    expr_list, expr_validity_array = assembler.assemble(tokens)
    layout_valid += np.sum(expr_validity_array)
    labels = vqa_label_array[n_begin:n_end]
    # Build TensorFlow Fold input for NMN
    expr_feed = compiler.build_feed_dict(expr_list)
    expr_feed[expr_validity_batch] = expr_validity_array

    # Part 2: Run NMN and learning steps
    scores_val = sess.partial_run(h, scores, feed_dict=expr_feed)

    # compute accuracy
    predictions = np.argmax(scores_val, axis=1)
    answer_correct += np.sum(np.logical_and(expr_validity_array,
                                            predictions == labels))

answer_accuracy = answer_correct / num_questions
layout_accuracy = layout_correct / num_questions
layout_validity = layout_valid / num_questions
print("answer accuracy =", answer_accuracy, "on", '_'.join(image_sets))
print("layout accuracy =", layout_accuracy, "on", '_'.join(image_sets))
print("layout validity =", layout_validity, "on", '_'.join(image_sets))
with open(save_file, 'w') as f:
    print("answer accuracy =", answer_accuracy, "on", '_'.join(image_sets), file=f)
    print("layout accuracy =", layout_accuracy, "on", '_'.join(image_sets), file=f)
    print("layout validity =", layout_validity, "on", '_'.join(image_sets), file=f)
