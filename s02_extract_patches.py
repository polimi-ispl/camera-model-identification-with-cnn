# -*- coding: UTF-8 -*-
"""
First Steps Towards Camera Model Identification with Convolutional Neural Networks
@author: Luca Bondi (luca.bondi@polimi.it)
"""

import numpy as np
import os
import cv2
from tqdm import tqdm
from multiprocessing import Pool
from patch_extractor import patch_extractor,mid_intensity_high_texture

# Functions ---
def extract_and_save(args):
    """
    Extract patches for a single image
    :param args: {'img_path','img_brand','img_model','patch_dim','patch_num'}
    :return: patches file path relative to patches_root
    """

    output_rel_paths = [ os.path.join(args['img_brand'],args['img_model'],os.path.splitext(os.path.split(args['img_path'])[-1])[0],
                                      os.path.splitext(os.path.split(args['img_path'])[-1])[0]+'_'+'{:04}'.format(patch_idx) + '.png')\
            for patch_idx in range(args['patch_num'])]

    # Check if image loading is necessary
    read_img = False
    for out_path in output_rel_paths:
        out_fullpath = os.path.join(args['patch_root'], out_path)
        if not os.path.exists(out_fullpath):
            read_img = True
            break

    if read_img:
        img = cv2.imread(os.path.join(args['img_root'],args['img_path']))
        if img is None or not isinstance(img, np.ndarray):
            print('Unable to read the image: {:}'.format(args['img_path']))
        patches = patch_extractor(img,args['patch_dim'],stride=args['patch_stride'],function=mid_intensity_high_texture,num=args['patch_num'])

        for out_path,patch in zip(output_rel_paths,patches):
            out_fullpath = os.path.join(args['patch_root'],out_path)
            out_fulldir = os.path.split(out_fullpath)[0]
            if not os.path.exists(out_fulldir):
                os.makedirs(out_fulldir)
            if not os.path.exists(out_fullpath):
                cv2.imwrite(out_fullpath,patch)

    return output_rel_paths

def main():
    # Parameters
    num_processes = 12

    from params import images_db_path,patch_dim,patch_stride,patches_root,\
        patch_num,dresden_images_root,patches_db_path

    # Load dataset
    images_db = np.load(images_db_path).item()

    print('Collecting image data...')
    imgs_list = []
    for img_brand,img_model,img_path in \
            tqdm(zip(images_db['brand'], images_db['model'], images_db['path'])):
        imgs_list += [{'img_path':img_path,
                       'img_brand':img_brand,
                       'img_model':img_model,
                       'patch_dim':patch_dim,
                       'patch_stride': patch_stride,
                       'patch_num':patch_num,
                       'patch_root': patches_root,
                       'img_root': dresden_images_root
                       }]

    print('Extracting patches...')
    pool = Pool(processes=num_processes)
    patches_paths = pool.map(extract_and_save,imgs_list)

    # Create patches dataset
    print('Creating patches dataset...')
    patch_dataset = dict()
    patch_dataset['path'] = []
    patch_dataset['shot'] = []

    for patch_rel_paths,img_shot in tqdm(zip(patches_paths,images_db['shot'])):
        for patch_rel_path in patch_rel_paths:
            patch_dataset['path'] += [patch_rel_path]
            patch_dataset['shot'] += [img_shot]

    patch_dataset['path'] = np.asarray(patch_dataset['path']).flatten()
    patch_dataset['shot'] = np.asarray(patch_dataset['shot']).flatten()

    print('Saving patches dataset to: {:}'.format(patches_db_path))
    np.save(patches_db_path, patch_dataset)

    print('Completed.')

if __name__ == '__main__':
    main()