import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()

gpu_id = args.gpu_id  # set GPU id to use
import os; os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
import sys
sys.path.append('../../')
from glob import glob

import skimage.io
import skimage.color
import numpy as np
import tensorflow as tf

from models_clevr import vgg_net

vgg_net_model = '../tfmodel/vgg_net/vgg_net.tfmodel'
image_basedir = '../clevr-dataset/images/'
save_basedir = './vgg_pool5/'

H = 320
W = 480
image_batch = tf.placeholder(tf.float32, [1, H, W, 3])
pool5 = vgg_net.vgg_pool5(image_batch, 'vgg_net')
saver = tf.train.Saver()

sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
saver.restore(sess, vgg_net_model)

def extract_image_pool5(impath):
    im = skimage.io.imread(impath)[..., :3]
    im_val = (im[np.newaxis, ...]-vgg_net.channel_mean)
    pool5_val = pool5.eval({image_batch: im_val}, sess)
    return pool5_val

def extract_dataset_pool5(image_dir, save_dir, ext_filter='*.png'):
    image_list = glob(image_dir + '/' + ext_filter)
    os.makedirs(save_dir, exist_ok=True)

    for n_im, impath in enumerate(image_list):
        if (n_im+1) % 100 == 0:
            print('processing %d / %d' % (n_im+1, len(image_list)))
        image_name = os.path.basename(impath).split('.')[0]
        save_path = os.path.join(save_dir, image_name + '.npy')
        if not os.path.exists(save_path):
            pool5_val = extract_image_pool5(impath)
            np.save(save_path, pool5_val)

for image_set in ['train', 'val', 'test']:
    print('Extracting image set ' + image_set)
    extract_dataset_pool5(os.path.join(image_basedir, image_set),
                          os.path.join(save_basedir, image_set))
    print('Done.')
