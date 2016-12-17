# -*- coding: UTF-8 -*-
"""
First Steps Towards Camera Model Identification with Convolutional Neural Networks
@author: Luca Bondi (luca.bondi@polimi.it)
"""

import os

dresden_csv = 'dresden.csv'
dresden_images_root = 'dresden'
images_db_path = 'dresden.npy'

patch_dim = 64
patch_stride = 64
patch_num =32

gpu_id = 0
seed = 197

patches_root = 'patches'
patches_db_path = os.path.join(patches_root,'patch.npy')

split_path = 'split.npy'

caffe_root = 'caffe'
def caffe_txt_path_generator(set_label):
    return os.path.join(caffe_root,set_label + '.txt')
def caffe_lmdb_path_generator(set_label):
    return os.path.join(caffe_root,set_label + '_db')
caffe_mean_path = os.path.join(caffe_root,'mean.binaryproto')
caffe_snapshot_folder = 'snapshot'
caffe_state_file = 'state.npy'
caffe_best_model_path = os.path.join(caffe_root,'best.caffemodel')

scores_path = 'scores.npy'
ppi_pdf_path = 'ppi.pdf'

