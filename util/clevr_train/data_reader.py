from __future__ import absolute_import, division, print_function

import os
import sys
import threading
import queue
import numpy as np

from util import text_processing

class BatchLoaderClevr:
    def __init__(self, imdb, data_params):
        self.imdb = imdb
        self.data_params = data_params

        self.vocab_dict = text_processing.VocabDict(data_params['vocab_question_file'])
        self.T_encoder = data_params['T_encoder']

        # peek one example to see whether answer and gt_layout are in the data
        self.load_answer = ('answer' in self.imdb[0]) and (self.imdb[0]['answer'] is not None)
        self.load_gt_layout = ('gt_layout_tokens' in self.imdb[0]) and (self.imdb[0]['gt_layout_tokens'] is not None)
        if 'load_gt_layout' in data_params:
            self.load_gt_layout = data_params['load_gt_layout']
        # the answer dict is always loaded, regardless of self.load_answer
        self.answer_dict = text_processing.VocabDict(data_params['vocab_answer_file'])
        if not self.load_answer:
            print('imdb does not contain answers')
        if self.load_gt_layout:
            self.T_decoder = data_params['T_decoder']
            self.assembler = data_params['assembler']
            self.prune_filter_module = (data_params['prune_filter_module']
                                        if 'prune_filter_module' in data_params
                                        else False)
        else:
            print('imdb does not contain ground-truth layout')

        # load one feature map to peek its size
        feats = np.load(self.imdb[0]['feature_path'])
        self.feat_H, self.feat_W, self.feat_D = feats.shape[1:]

    def load_one_batch(self, sample_ids):
        actual_batch_size = len(sample_ids)
        input_seq_batch = np.zeros((self.T_encoder, actual_batch_size), np.int32)
        seq_length_batch = np.zeros(actual_batch_size, np.int32)
        image_feat_batch = np.zeros((actual_batch_size, self.feat_H, self.feat_W, self.feat_D), np.float32)
        image_path_list = [None]*actual_batch_size
        if self.load_answer:
            answer_label_batch = np.zeros(actual_batch_size, np.int32)
        if self.load_gt_layout:
            gt_layout_batch = np.zeros((self.T_decoder, actual_batch_size), np.int32)

        for n in range(len(sample_ids)):
            iminfo = self.imdb[sample_ids[n]]
            question_inds = [self.vocab_dict.word2idx(w) for w in iminfo['question_tokens']]
            seq_length = len(question_inds)
            input_seq_batch[:seq_length, n] = question_inds
            seq_length_batch[n] = seq_length
            image_feat_batch[n:n+1] = np.load(iminfo['feature_path'])
            image_path_list[n] = iminfo['image_path']
            if self.load_answer:
                answer_idx = self.answer_dict.word2idx(iminfo['answer'])
                answer_label_batch[n] = answer_idx
            if self.load_gt_layout:
                gt_layout_tokens = iminfo['gt_layout_tokens']
                if self.prune_filter_module:
                    # remove duplicated consequtive modules (only keeping one _Filter)
                    for n_t in range(len(gt_layout_tokens)-1, 0, -1):
                        if (gt_layout_tokens[n_t-1] in {'_Filter', '_Find'}
                            and gt_layout_tokens[n_t] == '_Filter'):
                            gt_layout_tokens[n_t] = None
                    gt_layout_tokens = [t for t in gt_layout_tokens if t]
                gt_layout_batch[:, n] = self.assembler.module_list2tokens(
                    gt_layout_tokens, self.T_decoder)
        batch = dict(input_seq_batch=input_seq_batch,
                     seq_length_batch=seq_length_batch,
                     image_feat_batch=image_feat_batch,
                     image_path_list=image_path_list)
        if self.load_answer:
            batch['answer_label_batch'] = answer_label_batch
        if self.load_gt_layout:
            batch['gt_layout_batch'] = gt_layout_batch
        return batch

class DataReader:
    def __init__(self, imdb_file, shuffle=True, one_pass=False, prefetch_num=8, **kwargs):
        print('Loading imdb from file...', end=''); sys.stdout.flush()
        if imdb_file.endswith('.npy'):
            imdb = np.load(imdb_file)
        else:
            raise TypeError('unknown imdb format.')
        print('Done')
        self.imdb = imdb
        self.shuffle = shuffle
        self.one_pass = one_pass
        self.prefetch_num = prefetch_num
        self.data_params = kwargs

        # Clevr data loader
        self.batch_loader = BatchLoaderClevr(self.imdb, self.data_params)

        # Start prefetching thread
        self.prefetch_queue = queue.Queue(maxsize=self.prefetch_num)
        self.prefetch_thread = threading.Thread(target=_run_prefetch,
            args=(self.prefetch_queue, self.batch_loader, self.imdb,
                  self.shuffle, self.one_pass, self.data_params))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def batches(self):
        while True:
            # Get a batch from the prefetching queue
            if self.prefetch_queue.empty():
                print('data reader: waiting for data loading (IO is slow)...')
            batch = self.prefetch_queue.get(block=True)
            if batch is None:
                assert(self.one_pass)
                print('data reader: one pass finished')
                raise StopIteration()
            yield batch

def _run_prefetch(prefetch_queue, batch_loader, imdb, shuffle, one_pass, data_params):
    num_samples = len(imdb)
    batch_size = data_params['batch_size']

    n_sample = 0
    fetch_order = np.arange(num_samples)
    while True:
        # Shuffle the sample order for every epoch
        if n_sample == 0 and shuffle:
            fetch_order = np.random.permutation(num_samples)

        # Load batch from file
        # note that len(sample_ids) <= batch_size, not necessarily equal
        sample_ids = fetch_order[n_sample:n_sample+batch_size]
        batch = batch_loader.load_one_batch(sample_ids)
        prefetch_queue.put(batch, block=True)

        n_sample += len(sample_ids)
        if n_sample >= num_samples:
            # Put in a None batch to indicate a whole pass is over
            if one_pass:
                prefetch_queue.put(None, block=True)
            n_sample = 0
