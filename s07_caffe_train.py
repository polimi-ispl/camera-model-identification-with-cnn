# -*- coding: UTF-8 -*-
"""
First Steps Towards Camera Model Identification with Convolutional Neural Networks
@author: Luca Bondi (luca.bondi@polimi.it)
"""

def main():
    from params import caffe_state_file, caffe_snapshot_folder, caffe_root, caffe_best_model_path
    from caffe_wrapper import train_net

    plot = False
    resume = True

    train_net(caffe_root, caffe_snapshot_folder, caffe_state_file, best_model_path=caffe_best_model_path,plot_enable=plot, resume=resume)

    print('Completed.')

if __name__ == '__main__':
    main()
