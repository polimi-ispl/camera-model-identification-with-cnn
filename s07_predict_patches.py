# -*- coding: UTF-8 -*-
"""
First Steps Towards Camera Model Identification with Convolutional Neural Networks
@author: Luca Bondi (luca.bondi@polimi.it)
"""

import os
os.environ['GLOG_minloglevel'] = '2'
import numpy as np
import glob
from caffe_wrapper import extract_features_scores_files

def main():
    # Parameters
    from params import patches_db_path, patches_root, scores_path, caffe_best_model_path, caffe_mean_path, caffe_root

    batch_size = 1024

    # Load patches dataset
    patch_dataset = np.load(patches_db_path).item()

    cnn_model_path = os.path.join(caffe_root,'deploy.prototxt')

    args = (cnn_model_path, caffe_best_model_path, caffe_mean_path, batch_size,
            patches_root, patch_dataset['path'])

    _,scores = extract_features_scores_files(*args)

    print('Saving patch scores to: {:}'.format(scores_path))
    np.save(scores_path,{'score':scores})


    print('Completed.')


if __name__ == '__main__':
    main()


