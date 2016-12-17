# -*- coding: UTF-8 -*-
"""
First Steps Towards Camera Model Identification with Convolutional Neural Networks
@author: Luca Bondi (luca.bondi@polimi.it)
"""
import numpy as np
import lmdb
import caffe
import cv2
import os
from multiprocessing import Pool

def create_lmdb(args):
    lmdb_path = args['lmdb_path']
    txt_path  = args['txt_path']
    patch_dim = args['patch_dim']
    patches_root = args['patch_root']
    mean_path = args.pop('mean_path')



    with open(txt_path, 'r') as f:
        f_lines = f.readlines()

    mean_accumulator = np.zeros((patch_dim,patch_dim,3))
    num_patches = len(f_lines)

    if not os.path.exists(lmdb_path):
        print('Creating ' + lmdb_path + '...')
        map_size = num_patches * (patch_dim ** 2 * 3) * 2
        env = lmdb.open(lmdb_path, map_size=map_size)
        with env.begin(write=True) as txn:

            for idx,row in enumerate(f_lines):
                filename, label = row.rsplit(' ', 1)
                X = cv2.imread(os.path.join(patches_root,filename))
                mean_accumulator += X/255.
                X = np.swapaxes(X, 0, 2)
                X = np.swapaxes(X, 1, 2)
                X = X[::-1, :, :]
                datum = caffe.io.array_to_datum(X, int(label))
                str_id = '{:08}'.format(idx)

                # The encode is only essential in Python 3
                txn.put(str_id.encode('ascii'), datum.SerializeToString())

    elif mean_path is not None and not os.path.exists(mean_path):
        print('Computing the mean...')
        for idx, row in enumerate(f_lines):
            filename, label = row.rsplit(' ', 1)
            X = cv2.imread(os.path.join(patches_root, filename))
            mean_accumulator += X / 255.

    if mean_path is not None and not os.path.exists(mean_path):
        print('Creating ' + mean_path + '...')
        mean = mean_accumulator/num_patches*255
        mean = np.swapaxes(mean, 0, 2)
        mean = np.swapaxes(mean, 1, 2)
        mean = mean[::-1, :, :]
        mean.shape = (1,)+mean.shape

        mean_blob = caffe.io.array_to_blobproto(mean)

        with open(mean_path,'w') as f:
            f.write(mean_blob.SerializeToString())

def main():
    # Parameters
    from params import caffe_txt_path_generator,caffe_lmdb_path_generator,patch_dim,patches_root,caffe_mean_path
    num_processes = 2

    # Generate list of jobs
    jobs_args = []

    for set_label in ['train','val']:
        txt_path = caffe_txt_path_generator(set_label)
        lmdb_path = caffe_lmdb_path_generator(set_label)

        mean_path = caffe_mean_path if set_label=='train' else None
        jobs_args += [{'lmdb_path':lmdb_path,'txt_path':txt_path,'patch_dim':patch_dim,
                       'patch_root':patches_root,'mean_path':mean_path}]


    # Parallelize jobs
    pool = Pool(processes=num_processes)
    pool.map(create_lmdb,jobs_args)
    pool.close()
    pool.join()

    print('Completed.')

if __name__=='__main__':
    main()