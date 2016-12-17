# -*- coding: UTF-8 -*-
"""
First Steps Towards Camera Model Identification with Convolutional Neural Networks
@author: Luca Bondi (luca.bondi@polimi.it)
"""

import numpy as np
import matplotlib
matplotlib.use('PDF')
from matplotlib import rc
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)
import matplotlib.pyplot as plt
import scipy.io.matlab as mat
from tqdm import tqdm

def main():

    # Parameters
    label_fontsize = 20
    legend_fontsize = 20
    linewidth = 3
    tick_fontsize = 18

    from params import  split_path,scores_path, ppi_pdf_path,patch_num,patches_db_path

    images_db = np.load(split_path).item()
    patches_db = np.load(patches_db_path).item()
    patch_scores = np.load(scores_path).item()['score']

    num_classes = patch_scores.shape[1]

    ppi_acc = {}
    for set_id,set in zip([1,2,3],['train', 'val', 'test']):
        print('Set: {:}'.format(set))

        set_db_mask = images_db['split'] == set_id
        set_db = {}
        for key in images_db:
            set_db[key] = images_db[key][set_db_mask]
        num_imgs = len(set_db['shot'])
        gt = set_db['class']

        shot_scores = np.zeros((num_imgs, patch_num, num_classes))
        for shot_idx, shot in enumerate(tqdm(set_db['shot'])):
            shot_scores[shot_idx] = patch_scores[patches_db['shot'] == shot]

        ppi_acc_set = np.zeros(patch_num)
        for ppi_idx,ppi_num in enumerate(range(1,patch_num+1)):
            shot_scores_ppi = shot_scores[:,:ppi_num,:].mean(axis=1)
            pred = np.argmax(shot_scores_ppi,axis=1)
            acc = np.sum(gt == pred)/float(num_imgs)
            ppi_acc_set[ppi_idx] = acc
        ppi_acc[set] = ppi_acc_set

    fig = plt.figure()
    fig.canvas.set_window_title('PPI vs ACC')
    ax = plt.subplot(1,1,1)
    line = {}
    for set in ['train','val','test']:
        line['set'] = plt.plot(np.asarray(range(1,patch_num+1)), ppi_acc[set]*100,
                                   label='$\mathcal{M}_{Conv4}$ [' + set + '] (64x64 patches)',linewidth=linewidth)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(tick_fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(tick_fontsize)
    xlabel = plt.xlabel('# patches per image')
    xlabel.set_fontsize(label_fontsize)
    plt.grid(True)
    ylabel = plt.ylabel('Accuracy (%)')
    ylabel.set_fontsize(label_fontsize)
    plt.ylim([80,100])
    plt.xlim([1,patch_num])
    plt.legend(loc='lower right',fontsize=legend_fontsize)
    plt.draw()
    print('Saving plot to: {:}'.format(ppi_pdf_path))
    plt.savefig(ppi_pdf_path)



if __name__ == '__main__':
    main()
