# Learning to Reason: End-to-End Module Networks for Visual Question Answering

This repository contains the code for the following paper:

* R. Hu, J. Andreas, M. Rohrbach, T. Darrell, K. Saenko, *Learning to Reason: End-to-End Module Networks for Visual Question Answering*. in ICCV, 2017. ([PDF](https://arxiv.org/pdf/1704.05526.pdf))
```
@inproceedings{hu2017learning,
  title={Learning to Reason: End-to-End Module Networks for Visual Question Answering},
  author={Hu, Ronghang and Andreas, Jacob and Rohrbach, Marcus and Darrell, Trevor and Saenko, Kate},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year={2017}
}
```

Project Page: http://ronghanghu.com/n2nmn

## Installation

1. Install Python 3 (Anaconda recommended: https://www.continuum.io/downloads).
2. Install TensorFlow v1.0.0 (Note: newer or older versions of TensorFlow may fail to work due to incompatibility with TensorFlow Fold):  
`pip install tensorflow-gpu==1.0.0`  
3. Install [TensorFlow Fold](https://github.com/tensorflow/fold) (which is needed to run dynamic graph):  
`pip install https://storage.googleapis.com/tensorflow_fold/tensorflow_fold-0.0.1-py3-none-linux_x86_64.whl`
4. Download this repository or clone with Git, and then enter the root directory of the repository:  
`git clone https://github.com/ronghanghu/n2nmn.git && cd n2nmn`

## Train and evaluate on the CLEVR dataset

### Download and preprocess the data

1. Download the CLEVR dataset from http://cs.stanford.edu/people/jcjohns/clevr/, and symbol link it to `exp_clevr/clevr-dataset`. After this step, the file structure should look like
```
exp_clevr/clevr-dataset/
  images/
    train/
      CLEVR_train_000000.png
      ...
    val/
    test/
  questions/
    CLEVR_train_questions.json
    CLEVR_val_questions.json
    CLEVR_test_questions.json
  ...
```

2. Extract visual features from the images and store them on the disk. In our experiments, we keep the original 480 x 320 image size in CLEVR, and use the *pool5* layer output of shape the (1, 10, 15, 512) from VGG-16 network (feature stored as numpy array in HxWxC format). Then, construct the "expert layout" from ground-truth functional programs, and build image collections (imdb) for clevr. These procedures can be down as follows.
```
./exp_clevr/tfmodel/vgg_net/download_vgg_net.sh  # VGG-16 converted to TF

cd ./exp_clevr/data/
python extract_visual_features_vgg_pool5.py  # feature extraction
python get_ground_truth_layout.py  # construct expert policy
python build_clevr_imdb.py  # build image collections
cd ../../
```
The saved features will take up approximately **29GB disk space** (for all images in CLEVR train, val and test).

### Training

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  

1. Train with ground-truth layout (cloning expert + policy search after cloning)  
    * Step a (cloning expert):  
    `python exp_clevr/train_clevr_gt_layout.py`  
    * Step b (policy search after cloning):  
    `python exp_clevr/train_clevr_rl_gt_layout.py`  
    which is by default initialized from `exp_clevr/tfmodel/clevr_gt_layout/00050000` (the 50000-iteration snapshot in Step a). If you want to initialize from another snapshot, use the `--pretrained_model` flag to specify the snapshot path.

2. Train without ground-truth layout (policy search from scratch)  
`python exp_clevr/train_clevr_scratch.py`  

Note:
* By default, the above scripts use GPU 0. To train on a different GPU, set the `--gpu_id` flag. During training, the script will write TensorBoard events to `exp_clevr/tb/` and save the snapshots under `exp_clevr/tfmodel/`.
* Pre-trained models (TensorFlow snapshots) on CLEVR dataset can be downloaded from:  
    - clevr_gt_layout (cloning expert): https://people.eecs.berkeley.edu/~ronghang/projects/n2nmn/models/clevr_gt_layout/  
    - clevr_rl_gt_layout (policy search after cloning): https://people.eecs.berkeley.edu/~ronghang/projects/n2nmn/models/clevr_rl_gt_layout/  
    - clevr_scratch (policy search from scratch): https://people.eecs.berkeley.edu/~ronghang/projects/n2nmn/models/clevr_scratch/  
The downloaded snapshots should be placed under `exp_clevr/tfmodel/clevr_gt_layout`, `exp_clevr/tfmodel/clevr_rl_gt_layout` and `exp_clevr/tfmodel/clevr_scratch` respectively. You may evaluate their performance using the test code below.

### Test

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  

1. Evaluate *clevr_gt_layout* (cloning expert):  
`python exp_clevr/eval_clevr.py --exp_name clevr_gt_layout --snapshot_name 00050000 --test_split val`  
Expected accuracy: 78.9% (on val split).

2. Evaluate *clevr_rl_gt_layout* (policy search after cloning):  
`python exp_clevr/eval_clevr.py --exp_name clevr_rl_gt_layout --snapshot_name 00050000 --test_split val`  
Expected accuracy: 83.6% (on val split).

3. Evaluate *clevr_scratch* (policy search from scratch):  
`python exp_clevr/eval_clevr.py --exp_name train_clevr_scratch --snapshot_name 00100000 --test_split val`  
Expected accuracy: 69.1% (on val split).

Note:
* The above evaluation scripts will print out the accuracy (only for val split) and also save it under `exp_clevr/results/`. It will also save a prediction output file under `exp_clevr/eval_outputs/`.  
* By default, the above scripts use GPU 0, and evaluate on the *validation* split of CLEVR. To evaluate on a different GPU, set the `--gpu_id` flag.  
* To evaluate on the *test* split, use `--test_split tst` instead. As there is no ground-truth answers for *test* split in the downloaded CLEVR data, the evaluation script above will print out zero accuracy on the *test* split. You may email the prediction outputs in `exp_clevr/eval_outputs/` to the CLEVR dataset authors for the *test* split accuracy.

## Train and evaluate on the VQA dataset

### Download and preprocess the data

1. Download the VQA dataset annotations from http://www.visualqa.org/download.html, and symbol link it to `exp_vqa/vqa-dataset`. After this step, the file structure should look like
```
exp_vqa/vqa-dataset/
  Questions/
    OpenEnded_mscoco_train2014_questions.json
    OpenEnded_mscoco_val2014_questions.json
    OpenEnded_mscoco_test-dev2015_questions.json
    OpenEnded_mscoco_test2015_questions.json
  Annotations/
    mscoco_train2014_annotations.json
    mscoco_val2014_annotations.json
```

2. Download the COCO images from http://mscoco.org/, extract features from the images, and store them under `exp_vqa/data/resnet_res5c/`. In our experiments, we resize all the COCO images to 448 x 448, and use the *res5c* layer output of shape (1, 14, 14, 2048) from the [ResNet-152](https://github.com/KaimingHe/deep-residual-networks) network pretrained on ImageNET classification (feature stored as numpy array in HxWxC format). **In our experiments, we use the same ResNet-152 res5c features as in [MCB](https://github.com/akirafukui/vqa-mcb), except that the extracted features are stored in NHWC format (instead of NCHW format used in MCB).** 

The saved features will take up approximately **307GB disk space** (for all images in COCO train2014, val2014 and test2015). After feature extraction, the file structure for the features should look like
```
exp_vqa/data/resnet_res5c/
  train2014/
    COCO_train2014_000000000009.npy
    ...
  val2014/
    COCO_val2014_000000000042.npy
    ...
  test2015/
    COCO_test2015_000000000001.npy
    ...
```
where each of the `*.npy` file contains COCO image feature extracted from the *res5c* layer of the [ResNet-152](https://github.com/KaimingHe/deep-residual-networks) network, which is a numpy array of shape (1, 14, 14, 2048) and float32 type, stored in HxWxC format.  

3. Build image collections (imdb) for VQA:  
```
cd ./exp_vqa/data/
python build_vqa_imdb.py
cd ../../
```

Note: this repository already contains the parsing results from Stanford Parser for the VQA questions under `exp_vqa/data/parse/new_parse` (parsed using [this script](https://gist.github.com/ronghanghu/67aeb391f4839611d119c73eba53bc5f)), with the converted ground-truth (expert) layouts under `exp_vqa/data/gt_layout_*_new_parse.npy` (converted using notebook `exp_vqa/data/convert_new_parse_to_gt_layout.ipynb`).

### Training

Train with ground-truth layout:

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  
1. Step a (cloning expert):  
`python exp_vqa/train_vqa_gt_layout.py`  
2. Step b (policy search after cloning):  
`python exp_vqa/train_vqa_rl_gt_layout.py`

Note:
* By default, the above scripts use GPU 0, and train on the union of *train2014* and *val2014* splits. To train on a different GPU, set the `--gpu_id` flag. During training, the script will write TensorBoard events to `exp_vqa/tb/` and save the snapshots under `exp_vqa/tfmodel/`.
* Pre-trained models (TensorFlow snapshots) on VQA dataset can be downloaded from:  
    - vqa_gt_layout (cloning expert): https://people.eecs.berkeley.edu/~ronghang/projects/n2nmn/models/vqa_gt_layout/
    - vqa_rl_gt_layout (policy search after cloning): https://people.eecs.berkeley.edu/~ronghang/projects/n2nmn/models/vqa_rl_gt_layout/
The downloaded snapshots should be placed under `exp_vqa/tfmodel/vqa_gt_layout` and `exp_vqa/tfmodel/vqa_rl_gt_layout`. You may evaluate their performance using the test code below.

### Test

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  

1. Evaluate on *vqa_gt_layout* (cloning expert):  
    - (on test-dev2015 split):  
    `python exp_vqa/eval_vqa.py --exp_name vqa_gt_layout --snapshot_name 00040000 --test_split test-dev2015`
    - (on test2015 split):  
    `python exp_vqa/eval_vqa.py --exp_name vqa_gt_layout --snapshot_name 00040000 --test_split test2015`

2. Evaluate on *vqa_rl_gt_layout* (policy search after cloning):  
    - (on test-dev2015 split):  
    `python exp_vqa/eval_vqa.py --exp_name vqa_rl_gt_layout --snapshot_name 00040000 --test_split test-dev2015`
    - (on test2015 split):  
    `python exp_vqa/eval_vqa.py --exp_name vqa_rl_gt_layout --snapshot_name 00040000 --test_split test2015`

Note: the above evaluation scripts will not print out the accuracy, but will write the prediction outputs to `exp_vqa/eval_outputs/`, which can be uploaded to the evaluation sever (http://www.visualqa.org/roe.html) for evaluation. The expected accuacy of *vqa_rl_gt_layout* on test-dev2015 split is 64.9%.

## Train and evaluate on the VQAv2 dataset

### Download and preprocess the data

1. Download the VQAv2 dataset annotations from http://www.visualqa.org/download.html, and symbol link it to `exp_vqa/vqa-dataset`. After this step, the file structure should look like
```
exp_vqa/vqa-dataset/
  Questions/
    v2_OpenEnded_mscoco_train2014_questions.json
    v2_OpenEnded_mscoco_val2014_questions.json
    v2_OpenEnded_mscoco_test-dev2015_questions.jso
    v2_OpenEnded_mscoco_test2015_questions.json
  Annotations/
    v2_mscoco_train2014_annotations.json
    v2_mscoco_val2014_annotations.json
    v2_mscoco_train2014_complementary_pairs.json
    v2_mscoco_val2014_complementary_pairs.json
```

2. Download the COCO images from http://mscoco.org/, extract features from the images, and store them under `exp_vqa/data/resnet_res5c/`. In our experiments, we resize all the COCO images to 448 x 448, and use the *res5c* layer output of shape (1, 14, 14, 2048) from the [ResNet-152](https://github.com/KaimingHe/deep-residual-networks) network pretrained on ImageNET classification (feature stored as numpy array in HxWxC format). **In our experiments, we use the same ResNet-152 res5c features as in [MCB](https://github.com/akirafukui/vqa-mcb), except that the extracted features are stored in NHWC format (instead of NCHW format used in MCB).** 

The saved features will take up approximately **307GB disk space** (for all images in COCO train2014, val2014 and test2015). After feature extraction, the file structure for the features should look like
```
exp_vqa/data/resnet_res5c/
  train2014/
    COCO_train2014_000000000009.npy
    ...
  val2014/
    COCO_val2014_000000000042.npy
    ...
  test2015/
    COCO_test2015_000000000001.npy
    ...
```
where each of the `*.npy` file contains COCO image feature extracted from the *res5c* layer of the [ResNet-152](https://github.com/KaimingHe/deep-residual-networks) network, which is a numpy array of shape (1, 14, 14, 2048) and float32 type, stored in HxWxC format.  

3. Build image collections (imdb) for VQAv2:  
```
cd ./exp_vqa/data/
python build_vqa_v2_imdb.py
cd ../../
```

Note: this repository already contains the parsing results from Stanford Parser for the VQAv2 questions under `exp_vqa/data/parse/new_parse_vqa_v2` (parsed using [this script](https://gist.github.com/ronghanghu/67aeb391f4839611d119c73eba53bc5f)), with the converted ground-truth (expert) layouts under `exp_vqa/data/v2_gt_layout_*_new_parse.npy`.

### Training

Train with ground-truth layout:

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  
1. Step a (cloning expert):  
`python exp_vqa/train_vqa2_gt_layout.py`  
2. Step b (policy search after cloning):  
`python exp_vqa/train_vqa2_rl_gt_layout.py`

Note:
* By default, the above scripts use GPU 0, and train on the union of *train2014* and *val2014* splits. To train on a different GPU, set the `--gpu_id` flag. During training, the script will write TensorBoard events to `exp_vqa/tb/` and save the snapshots under `exp_vqa/tfmodel/`.
* Pre-trained models (TensorFlow snapshots) on VQAv2 dataset can be downloaded from:  
    - vqa2_gt_layout (cloning expert): https://people.eecs.berkeley.edu/~ronghang/projects/n2nmn/models/vqa2_gt_layout/
    - vqa2_rl_gt_layout (policy search after cloning): https://people.eecs.berkeley.edu/~ronghang/projects/n2nmn/models/vqa2_rl_gt_layout/
The downloaded snapshots should be placed under `exp_vqa/tfmodel/vqa2_gt_layout` and `exp_vqa/tfmodel/vqa2_rl_gt_layout`. You may evaluate their performance using the test code below.

### Test

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  

1. Evaluate on *vqa2_gt_layout* (cloning expert):  
    - (on test-dev2015 split):  
    `python exp_vqa/eval_vqa2.py --exp_name vqa2_gt_layout --snapshot_name 00080000 --test_split test-dev2015`
    - (on test2015 split):  
    `python exp_vqa/eval_vqa2.py --exp_name vqa2_gt_layout --snapshot_name 00080000 --test_split test2015`

2. Evaluate on *vqa2_rl_gt_layout* (policy search after cloning):  
    - (on test-dev2015 split):  
    `python exp_vqa/eval_vqa2.py --exp_name vqa2_rl_gt_layout --snapshot_name 00080000 --test_split test-dev2015`
    - (on test2015 split):  
    `python exp_vqa/eval_vqa2.py --exp_name vqa2_rl_gt_layout --snapshot_name 00080000 --test_split test2015`

Note: the above evaluation scripts will not print out the accuracy, but will write the prediction outputs to `exp_vqa/eval_outputs/`, which can be uploaded to the evaluation sever (http://www.visualqa.org/roe.html) for evaluation. The expected accuacy of *vqa2_rl_gt_layout* on test-dev2015 split is 63.3%.

## Train and evaluate on the SHAPES dataset

A copy of the SHAPES dataset is contained in this repository under `exp_shapes/shapes_dataset`. The ground-truth module layouts (expert layouts) we use in our experiments are also provided under `exp_shapes/data/*_symbols.json`. The script to obtain the expert layouts from the annotations is in `exp_shapes/data/get_ground_truth_layout.ipynb`.

### Training

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  

1. Train with ground-truth layout (behavioral cloning from expert):  
`python exp_shapes/train_shapes_gt_layout.py`  

2. Train without ground-truth layout (policy search from scratch):  
`python exp_shapes/train_shapes_scratch.py`  

Note: by default, the above scripts use GPU 0. To train on a different GPU, set the `--gpu_id` flag. During training, the script will write TensorBoard events to `exp_shapes/tb/` and save the snapshots under `exp_shapes/tfmodel/`.

### Test

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  

1. Evaluate *shapes_gt_layout* (behavioral cloning from expert):  
`python exp_shapes/eval_shapes.py --exp_name shapes_gt_layout --snapshot_name 00040000 --test_split test`  

2. Evaluate *shapes_scratch* (policy search from scratch):  
`python exp_shapes/eval_shapes.py --exp_name shapes_scratch --snapshot_name 00400000 --test_split test`  

Note: the above evaluation scripts will print out the accuracy and also save it under `exp_shapes/results/`. By default, the above scripts use GPU 0, and evaluate on the *test* split of SHAPES. To evaluate on a different GPU, set the `--gpu_id` flag. To evaluate on the *validation* split, use `--test_split val` instead.
