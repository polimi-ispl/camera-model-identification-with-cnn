# -*- coding: UTF-8 -*-
"""
First Steps Towards Camera Model Identification with Convolutional Neural Networks
@author: Luca Bondi (luca.bondi@polimi.it)
"""

import os
import csv
import urllib
import cv2
from tqdm import tqdm
import numpy as np

def main():

    from params import dresden_images_root,dresden_csv, images_db_path

    brand_list = []
    model_list = []
    brand_model_list = []
    instance_list = []
    shot_list = []
    path_list = []
    position_num_list = []
    position_name_list = []
    motive_num_list = []
    motive_name_list = []

    # Create output folder if needed
    if not os.path.exists(dresden_images_root):
        os.makedirs(dresden_images_root)

    # Load input CSV
    with open(dresden_csv, 'r') as f_csv:
        reader = csv.reader(f_csv)

        csv_headers = reader.next()
        csv_rows = [csv_row for csv_row in reader]

        count = 0
        for csv_row in tqdm(csv_rows):
            filename, \
            brand, model, instance, shot, \
            position_num, position_name, motive_num, motive_name, \
            url = csv_row

            file_path = os.path.join(dresden_images_root,filename)

            try:
                if not os.path.exists(file_path):
                    print('Downloading {:}'.format(filename))
                    urllib.urlretrieve(url, file_path)

                #print('Loading {:}'.format(filename))

                # Load the image and check its dimensions
                img = cv2.imread(file_path)

                if img is None or not isinstance(img, np.ndarray):
                    print('Unable to read image: {:}'.format(filename))
                    os.unlink(file_path)

                if all(img.shape[:2]):
                    count += 1

                    brand_model = '_'.join([brand, model])
                    brand_list.append(brand)
                    model_list.append(model)
                    brand_model_list.append(brand_model)
                    instance_list.append(int(instance))
                    shot_list.append(int(shot))
                    path_list.append(filename)
                    position_num_list.append(int(position_num))
                    position_name_list.append(position_name)
                    motive_num_list.append(int(motive_num))
                    motive_name_list.append(motive_name)

                else:
                    print('Zero-sized image: {:}'.format(filename))
                    os.unlink(file_path)

            except IOError:
                print('Unable to decode: {:}'.format(filename))
                os.unlink(file_path)

            except Exception as e:
                print('Error while loading: {:}'.format(filename))
                if os.path.exists(file_path):
                    os.unlink(file_path)

    print('Number of images: {:}'.format(len(shot_list)))

    print('Saving db to: {:}'.format(images_db_path))
    images_db = {
        'brand': np.asarray(brand_list).flatten(),
        'model': np.asarray(model_list).flatten(),
        'brand_model': np.asarray(brand_model_list).flatten(),
        'instance': np.asarray(instance_list).flatten(),
        'shot': np.asarray(shot_list).flatten(),
        'path': np.asarray(path_list).flatten(),
        'position_num': np.asarray(position_num_list).flatten(),
        'position_name': np.asarray(position_name_list).flatten(),
        'motive_num': np.asarray(motive_num_list).flatten(),
        'motive_name': np.asarray(motive_name_list).flatten(),
    }
    np.save(images_db_path, images_db)

    print('Completed.')

if __name__ == '__main__':
    main()
