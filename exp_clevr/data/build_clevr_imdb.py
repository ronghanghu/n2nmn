import numpy as np
import json
import os

import sys
sys.path.append('../../')
from util import text_processing

question_file = './CLEVR_%s_questions_gt_layout.json'
image_dir = '../clevr-dataset/images/%s/'
feature_dir = './vgg_pool5/%s/'

def build_imdb(image_set):
    print('building imdb %s' % image_set)
    with open(question_file % image_set) as f:
        questions = json.load(f)
    abs_image_dir = os.path.abspath(image_dir % image_set)
    abs_feature_dir = os.path.abspath(feature_dir % image_set)
    imdb = [None]*len(questions)
    for n_q, q in enumerate(questions):
        if (n_q+1) % 10000 == 0:
            print('processing %d / %d' % (n_q+1, len(questions)))
        image_name = q['image_filename'].split('.')[0]
        image_path = os.path.join(abs_image_dir, q['image_filename'])
        feature_path = os.path.join(abs_feature_dir, image_name + '.npy')
        question_str = q['question']
        question_tokens = text_processing.tokenize(question_str)
        gt_layout_tokens = None
        if 'gt_layout' in q:
            gt_layout_tokens = q['gt_layout']
        answer = None
        if 'answer' in q:
            answer = q['answer']

        iminfo = dict(image_name=image_name,
                      image_path=image_path,
                      feature_path=feature_path,
                      question_str=question_str,
                      question_tokens=question_tokens,
                      gt_layout_tokens=gt_layout_tokens,
                      answer=answer)
        imdb[n_q] = iminfo
    return imdb

imdb_trn = build_imdb('train')
imdb_val = build_imdb('val')
imdb_tst = build_imdb('test')

os.makedirs('./imdb', exist_ok=True)
np.save('./imdb/imdb_trn.npy', np.array(imdb_trn))
np.save('./imdb/imdb_val.npy', np.array(imdb_val))
np.save('./imdb/imdb_tst.npy', np.array(imdb_tst))
