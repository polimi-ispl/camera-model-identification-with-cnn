# -*- coding: UTF-8 -*-
"""
First Steps Towards Camera Model Identification with Convolutional Neural Networks
@author: Luca Bondi (luca.bondi@polimi.it)
@author: Nicolò Bonettini (nicolo.bonettini@mail.polimi.it)
"""

import random
import numpy as np
import cv2
import types


## Score functions ---

def one(img):
    """
    Fake patch scoring function. Always returns 1
    :param img:
    :return:
    """
    return 1


def mid_intensity_high_texture(img):
    """

    :param img: 2D or 3D ndarray. Values are expected in [0,1] if img is float, in [0,255] if img is uint8
    :return score: score in [0,1]. Score tends to 1 as intensity is not saturated and high texture occurs
    """

    if img.dtype == np.uint8:
        img = img / 255.

    mean_std_weight = .7

    num_ch = 1 if img.ndim == 2 else img.shape[-1]
    img_flat = img.reshape(-1, num_ch)
    ch_mean = img_flat.mean(axis=0)
    ch_std = img_flat.std(axis=0)

    ch_mean_score = -4 * ch_mean ** 2 + 4 * ch_mean
    ch_std_score = 1 - np.exp(-2 * np.log(10) * ch_std)

    ch_mean_score_aggr = ch_mean_score.mean()
    ch_std_score_aggr = ch_std_score.mean()

    score = mean_std_weight * ch_mean_score_aggr + (1 - mean_std_weight) * ch_std_score_aggr
    return score


def patch_extractor(img, dim, **kwargs):
    """
    Patch extractor.
    Args:
    :param img (numpy.ndarray): the image to process. dtype must be either uint8 or float

    :param dim (tuple | int): the dimensions of the patches (rows,cols).

    Named args:
    :param offset (tuple | int): the offsets of each axis starting from top left corner (rows,cols).

    :param stride (tuple | int): the stride of each axis starting from top left corner (rows,cols).

    :param rand (bool): rand patches. Must not be set together with function_handler

    :param function_handler (function_handler): patch quality function_handler handler. Must not be set together with rand

    :param threshold (float):  minimum quality threshold

    :param num (int): maximum number of patches

    :return list: list of numpy.ndarray of the same type as the input img
    """

    # Arguments parser ---
    if not isinstance(img, np.ndarray):
        raise ValueError('img must be of type: ' + str(np.ndarray))

    if not img.dtype == np.uint8 and not img.dtype == np.float32 and not img.dtype == np.float32:
        raise ValueError('img must have dtype: [' + '|'.join([str(np.uint8), str(np.float32), str(np.float64)]) + ']')

    if isinstance(dim, int):
        dim = (dim, dim)
    if not isinstance(dim, tuple):
        raise ValueError('dim must be of type: [' + '|'.join([str(int), str(tuple)]) + ']')

    if kwargs.has_key('offset'):
        offset = kwargs.pop('offset')
        if isinstance(offset, int):
            offset = (offset, offset)
        if not isinstance(offset, tuple):
            raise ValueError('offset must be of type: [' + '|'.join([str(int), str(tuple)]) + ']')
    else:
        offset = (0, 0)

    if kwargs.has_key('stride'):
        stride = kwargs.pop('stride')
        if isinstance(stride, int):
            stride = (stride, stride)
        if not isinstance(stride, tuple):
            raise ValueError('stride must be of type: [' + '|'.join([str(int), str(tuple)]) + ']')
    else:
        stride = dim

    if kwargs.has_key('rand'):
        rand = kwargs.pop('rand')
        if not isinstance(rand, bool):
            raise ValueError('rand must be of type: ' + str(bool))
    else:
        rand = False

    if kwargs.has_key('function'):
        function_handler = kwargs.pop('function')
        if not callable(function_handler):
            raise ValueError('function must be a function handler')
    else:
        function_handler = None

    if kwargs.has_key('threshold'):
        threshold = kwargs.pop('threshold')
        if not isinstance(threshold, float):
            raise ValueError('threshold must be of type [' + '|'.join([str(np.float32), str(np.float64)]) + ']')
    else:
        threshold = 0

    if kwargs.has_key('num'):
        num = kwargs.pop('num')
        if type(num) != int:
            raise ValueError('num must be of type: ' + str(int))
    else:
        num = None

    if rand and function_handler is not None:
        raise ValueError('rand and function cannot be both set at the same time')

    if len(kwargs.keys()):
        for key in kwargs:
            raise('Unrecognized parameter: {:}'.format(key))

    if function_handler is None:
        function_handler = one

    # Offset ---
    img = img[offset[0]:, offset[1]:]

    # Vertical padding ---
    if img.shape[0] < dim[0] or img.shape[1] < dim[1]:
        pad_amount_0 = np.max((np.ceil((dim[0] - img.shape[0]) / 2.), 0)).astype(np.int)
        pad_amount_1 = np.max((np.ceil((dim[1] - img.shape[1]) / 2.), 0)).astype(np.int)
        if img.ndim == 3:
            img = np.pad(img, [(pad_amount_0,), (pad_amount_1,), (0,)], 'constant')
        else:
            img = np.pad(img, [(pad_amount_0,), (pad_amount_1,)], 'constant')

    # Patch list ---
    patch_list = []
    for start_col in np.arange(start=0, stop=img.shape[1] - dim[1] + 1, step=stride[1]):
        for start_row in np.arange(start=0, stop=img.shape[0] - dim[0] + 1, step=stride[0]):
            patch = img[start_row:start_row + dim[0], start_col:start_col + dim[1], ...]
            patch_list += [patch]

    # Evaluate patches or rand sort ---
    if rand:
        random.shuffle(patch_list)
    else:
        patch_scores = np.asarray(map(function_handler, patch_list))
        patch_array = np.asarray(patch_list)
        sort_idxs = np.argsort(patch_scores)[::-1]
        patch_scores = patch_scores[sort_idxs]
        patch_array = patch_array[sort_idxs]
        patch_array = patch_array[patch_scores >= threshold]
        patch_list = [patch for patch in patch_array]

    if num is not None:
        patch_list = patch_list[:num]

    return patch_list


def patch_extractor_one_arg(args):
    """
    Patch extractor.
    args keys:

    :param img_path (string): path of the image to process.

    :param dim (tuple | int): the dimensions of the patches (rows,cols).

    :param offset (tuple | int): the offsets of each axis starting from top left corner (rows,cols).

    :param stride (tuple | int): the stride of each axis starting from top left corner (rows,cols).

    :param rand (bool): rand patches. Must not be set together with function

    :param function (function): patch quality function handler. Must not be set together with rand

    :param threshold (float):  minimum quality threshold

    :param num (int): maximum number of patches

    :return list: list of numpy.ndarray of the same type as the input img
    """

    img_path = args.pop('img_path')
    dim = args.pop('dim')

    img = cv2.imread(img_path, 1)

    return patch_extractor(img,
                           dim,
                           **args)


if __name__ == "__main__":
    img = np.random.randint(256, size=(64, 32, 3)).astype(np.uint8)
    img_path = 'patch_extractor_test.png'
    cv2.imwrite(img_path, img)

    args = {'img_path': img_path,
            'dim': (8, 6),
            'offset': 5,
            'stride': 8,
            'rand': False,
            'function': mid_intensity_high_texture,
            'threshold': .5,
            'num': 12,
            }

    patches = patch_extractor_one_arg(args)

    print('Extracted {} patches'.format(len(patches)))