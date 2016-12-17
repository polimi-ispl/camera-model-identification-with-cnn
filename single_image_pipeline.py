# -*- coding: UTF-8 -*-
"""
First Steps Towards Camera Model Identification with Convolutional Neural Networks
@author: Luca Bondi (luca.bondi@polimi.it)
"""
import os
os.environ['GLOG_minloglevel'] = '2'
import argparse
from caffe_wrapper import extract_features_scores
from patch_extractor import patch_extractor, mid_intensity_high_texture
import cv2
import numpy as np

def main():

    from params import caffe_root,caffe_mean_path,caffe_best_model_path, patch_num, patch_dim, patch_stride, caffe_best_model_path, caffe_txt_path_generator

    parser = argparse.ArgumentParser()
    parser.add_argument('imgpath',help='Image path')
    args = parser.parse_args()

    file_path = args.imgpath
    if not os.path.exists(file_path):
        raise ValueError('File not found: {:}'.format(file_path))

    labels_file = caffe_txt_path_generator('labels')
    with open(labels_file,'r') as f:
        labels = f.readlines()
    labels = [i[:-1] for i in labels]

    print('Extracting patches...')
    img = cv2.imread(file_path)
    patches = patch_extractor(img,patch_dim,stride=patch_stride,num=patch_num,function=mid_intensity_high_texture)

    print('Extracting scores from patches...')
    cnn_model_path = os.path.join(caffe_root,'deploy.prototxt')
    _,scores = extract_features_scores(cnn_model_path, caffe_best_model_path, caffe_mean_path,patches)

    print('Averaging patches scores...')
    img_score = scores.mean(axis=0)
    prediction_class = np.argmax(img_score)
    prediction_label = labels[prediction_class]

    print('Predicted class: {:}'.format(prediction_label))


if __name__=='__main__':
    main()