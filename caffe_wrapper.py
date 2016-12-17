# -*- coding: UTF-8 -*-
"""
First Steps Towards Camera Model Identification with Convolutional Neural Networks
@author: Luca Bondi (luca.bondi@polimi.it)
"""

import scipy.ndimage
from tqdm import trange
import subprocess
import numpy as np
import os
import glob
import shutil
import caffe
from fractions import gcd

## Functions --
def read_solver_params(solver_path):
    params = {}
    with open(solver_path, 'r') as f:
        for line in f:
            valid = line.split('#', 1)
            valid = valid[0]
            if len(valid):
                key, value = valid.split(':', 1)
                value = value.strip()
                params[key] = value
    return params


def plot_init(title):
    gnuplot = subprocess.Popen(["gnuplot"], stdin=subprocess.PIPE)
    # gnuplot.stdin.write("set term dumb\n")
    gnuplot.stdin.write("set title \"" + title + "\" noenhanced \n")
    return gnuplot


def plot_iter(gnuplot, train_loss, val_loss, val_acc):
    gnuplot.stdin.write( \
        "plot '-' using 1:2 title 'Train Loss' with linespoints," +
        "'-' using 1:2 title 'Val loss' with linespoints \n")
    for x, y in train_loss:
        gnuplot.stdin.write("%f %f\n" % (x, y))
    gnuplot.stdin.write("e\n")
    for x, y in val_loss:
        gnuplot.stdin.write("%f %f\n" % (x, y))
    gnuplot.stdin.write("e\n")
    gnuplot.stdin.flush()


def val_loss_acc(solver, test_iter):
    loss = 0
    acc = 0
    for test_it in range(test_iter):
        solver.test_nets[0].forward()
        loss += solver.test_nets[0].blobs['loss'].data
        acc += solver.test_nets[0].blobs['accuracy'].data
    loss /= test_iter
    acc /= test_iter
    return loss, acc


def print_iter(dst_root, train_loss, val_loss, val_acc):
    train_loss_arr = np.array(train_loss)
    val_loss_arr = np.array(val_loss)
    val_acc_arr = np.array(val_acc)

    last_it = train_loss_arr[-1, 0]
    last_train_loss = train_loss_arr[-1, 1]
    last_val_loss = val_loss_arr[-1, 1]
    last_val_acc = val_acc_arr[-1, 1]

    print(
    '[' + dst_root + '] It:{:06.0f} TrainLoss:{:1.4f} ValLoss:{:1.4f} ValAcc:{:1.2f}'.format(last_it, last_train_loss,
                                                                                             last_val_loss,
                                                                                             last_val_acc))

def smallest_val_loss_iter(val_loss):
    val_loss_arr = np.array(val_loss)
    min_val_loss_idx = np.argmin(val_loss_arr[:, 1])
    min_val_loss_it = val_loss_arr[min_val_loss_idx, 0]
    return int(min_val_loss_it)

def print_smallest_val_loss(dst_root, val_loss, val_acc):
    val_loss_arr = np.array(val_loss)
    val_acc_arr = np.array(val_acc)

    min_val_loss_idx = np.argmin(val_loss_arr[:, 1])
    min_val_loss_it = val_loss_arr[min_val_loss_idx, 0]
    min_val_loss_val = val_loss_arr[min_val_loss_idx, 1]
    min_val_loss_acc = val_acc_arr[min_val_loss_idx, 1]

    print('[' + dst_root + '] MinValLoss - It:{:06.0f} ValLoss:{:1.4f} ValAcc:{:1.2f}'.format(min_val_loss_it,
                                                                                              min_val_loss_val,
                                                                                              min_val_loss_acc))


def train_net_call(args):
    train_net(*args)


def train_net(dst_root, snapshot_folder='snapshot', state_file='state.npy', best_model_path=None, gpu_id=0, plot_enable=False, resume=True):

    print('\n\nTraining started: ' + dst_root)

    pwd = os.getcwd()
    os.chdir(dst_root)

    it = 0
    train_loss = []
    val_loss = []
    val_acc = []

    # Check for resume/restart
    if os.path.exists(os.path.join(snapshot_folder, state_file)):
        if resume:
            # Determine last snapshot and prepare for resume
            train_val_data = np.load(os.path.join(snapshot_folder, state_file)).item()
            train_loss = train_val_data['train_loss']
            val_loss = train_val_data['val_loss']
            val_acc = train_val_data['val_acc']

            snapshots_names = glob.glob(os.path.join(snapshot_folder, 'snapshot_iter_*.solverstate'))
            snapshots_it = [int(name.rsplit('.')[0].rsplit('_')[-1]) for name in snapshots_names]

            if len(snapshots_it):

                it = np.max(snapshots_it)

                train_loss = [element for element in train_loss if element[0] < it]
                val_loss = [element for element in val_loss if element[0] < it]
                val_acc = [element for element in val_acc if element[0] < it]

                resume_solver_file = 'snapshot_iter_{:}.solverstate'.format(it)
                resume_model_file = 'snapshot_iter_{:}.caffemodel'.format(it)
                print('Resume from iteration {:}'.format(it))

            else:
                resume = False
                print('Nothing to resume. Restarting')
                shutil.rmtree(snapshot_folder)
                os.mkdir(snapshot_folder)

        else:
            # Delete previous snapshots
            print('Deleting previous snapshots')
            shutil.rmtree(snapshot_folder)
            os.mkdir(snapshot_folder)

    else:
        resume = False
        try:
            os.makedirs(snapshot_folder)
        except OSError:
            pass

    if plot_enable:
        print('Initializing plot')
        plot_env = plot_init(dst_root)

    solver_params = read_solver_params('solver.prototxt')
    max_iter = int(solver_params['max_iter'])  # maximum number of training iterations
    test_iter = int(solver_params['test_iter'])  # number of iterations necessary to test the whole validation set
    test_interval = int(solver_params['test_interval'])  # number of training iterations after which test is performed
    display = int(solver_params['display'])  # display ongoing train every display iterations

    print('Loading solver')
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    solver = caffe.get_solver('solver.prototxt')

    os.chdir(snapshot_folder)

    if resume:
        print('Resuming state')
        solver.restore(resume_solver_file)
        solver.test_nets[0].copy_from(resume_model_file)

    print('Initial net forwarding')
    solver.net.forward()

    # the main solver loop
    it_step = gcd(display, test_interval)
    print('Training')

    try:
        while it < max_iter:

            # store and eventually show train loss
            if it % display == 0:
                train_loss += [(it, float(solver.net.blobs['loss'].data))]
                if it % test_interval != 0:
                    if plot_enable:
                        plot_iter(plot_env, train_loss, val_loss, val_acc)
                    print_iter(dst_root, train_loss, val_loss, val_acc)
                    np.save(state_file,
                            {'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc});

            # store and eventually show validation loss
            if it % test_interval == 0:
                loss, acc = val_loss_acc(solver, test_iter)
                val_loss += [(it, loss)]
                val_acc += [(it, acc)]
                if plot_enable:
                    plot_iter(plot_env, train_loss, val_loss, val_acc)
                print_iter(dst_root, train_loss, val_loss, val_acc)
                np.save(state_file,
                        {'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc})

            solver.step(it_step)  # SGD by Caffe
            it += it_step


    except KeyboardInterrupt:
        print('Training interrupted')
        pass

    os.chdir(pwd)
    if best_model_path is not None:
        print('Copying smaller val loss model to: {:}'.format(best_model_path))
        smaller_val_loss_it = smallest_val_loss_iter(val_loss)
        best_model_src = os.path.join(dst_root,snapshot_folder,'*{:}.caffemodel'.format(smaller_val_loss_it))
        best_model_src = glob.glob(best_model_src)[0]
        shutil.copyfile(best_model_src,best_model_path)
    print('Training completed: ' + dst_root)
    print_smallest_val_loss(dst_root, val_loss, val_acc)


def extract_features_scores_call(args):
    extract_features_scores_files(*args)


def extract_features_scores_files(cnn_model_path, cnn_weights_path, cnn_mean_path,
                                  batch_size, patches_root, patches_path, features_path = None, gpu_id = 0):
    print('Extracting features...')
    # Network initialization
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(cnn_model_path, cnn_weights_path, caffe.TEST)

    # Set CNN batch size
    net.blobs['data'].reshape(batch_size, net.blobs['data'].shape[1], net.blobs['data'].shape[2],
                              net.blobs['data'].shape[3])
    net.reshape()

    # Load mean
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(cnn_mean_path, 'rb').read()
    blob.ParseFromString(data)
    mean = np.array(caffe.io.blobproto_to_array(blob))

    # Set data transformer
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    if net.blobs['data'].data.shape[1] == 3:
        transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', mean.squeeze(0))

    features = np.zeros((len(patches_path),net.blobs['ip1'].shape[1]),np.float32)
    scores = np.zeros((len(patches_path),net.blobs['ip2'].shape[1]),np.float32)

    for batch_start_idx in trange(0, len(patches_path), batch_size):
        imgs = np.zeros(net.blobs['data'].data.shape, np.float32)
        batch_end_idx = min(batch_start_idx + batch_size, len(patches_path))
        for dst_img_idx, img_idx in enumerate(
                range(batch_start_idx, batch_end_idx)):
            img = scipy.ndimage.imread(os.path.join(patches_root, patches_path[img_idx]))
            if img.ndim < 3:
                img.shape = img.shape + (1,)
                raise ValueError(
                    'image is single channel: ' + os.path.join(patches_root, patches_path[img_idx]))
            imgs[dst_img_idx] = transformer.preprocess('data', img)

        net.blobs['data'].data[...] = imgs
        #net.set_input_arrays(imgs, np.zeros(batch_size,np.float32))
        net.forward()

        features[batch_start_idx:batch_end_idx] = net.blobs['ip1'].data[:batch_end_idx-batch_start_idx]
        scores[batch_start_idx:batch_end_idx] = net.blobs['ip2'].data[:batch_end_idx - batch_start_idx]

    if features_path is not None:
        print('Saving features to: ' + features_path)
        np.save(features_path, {'feature': features, 'score': scores})

    return features,scores

def extract_features_scores(cnn_model_path, cnn_weights_path, cnn_mean_path,patches_list, gpu_id = 0):

    # Network initialization
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(cnn_model_path, cnn_weights_path, caffe.TEST)

    # Set CNN batch size
    net.blobs['data'].reshape(len(patches_list), net.blobs['data'].shape[1], net.blobs['data'].shape[2],
                              net.blobs['data'].shape[3])
    net.reshape()

    # Load mean
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(cnn_mean_path, 'rb').read()
    blob.ParseFromString(data)
    mean = np.array(caffe.io.blobproto_to_array(blob))

    # Set data transformer
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    if net.blobs['data'].data.shape[1] == 3:
        transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', mean.squeeze(0))

    imgs = np.zeros(net.blobs['data'].data.shape, np.float32)
    for dst_img_idx, img in enumerate(patches_list):
        imgs[dst_img_idx] = transformer.preprocess('data', img)

    net.blobs['data'].data[...] = imgs
    net.forward()

    features = np.copy(net.blobs['ip1'].data)
    scores = np.copy(net.blobs['ip2'].data)

    return features,scores