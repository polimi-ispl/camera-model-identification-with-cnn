import os
import numpy as np
from multiprocessing import Pool

def create_txt(args):
    np.random.seed(args['seed'])
    images_db = args['images_db']
    patch_dataset = args['patch_dataset']
    caffe_txt_path = args['txt_path']

    print('Creating ' + caffe_txt_path + '...')

    patch_list = []
    split_set_mask = images_db['split'] == args['set_id']
    shot_split_set = images_db['shot'][split_set_mask]
    class_split_set = images_db['class'][split_set_mask]
    for shot, shot_class in zip(shot_split_set, class_split_set):
        patch_path_shot = patch_dataset['path'][patch_dataset['shot'] == shot]
        for patch_path in patch_path_shot:
            patch_list += [(patch_path.tostring(), '{}'.format(shot_class))]

    # Shuffling
    rand_idxs = np.random.permutation(np.arange(len(patch_list)))

    # Save to file
    caffe_txt_folder = os.path.split(caffe_txt_path)[0]
    if not os.path.exists(caffe_txt_folder):
        os.makedirs(caffe_txt_folder)
    with open(caffe_txt_path, 'w') as f:
        for idx in rand_idxs:
            f.write(' '.join(patch_list[idx]) + '\n')

def main():

    # Parameters
    num_processes = 2
    from params import patches_db_path,patch_num,patch_dim,patch_stride,split_path,\
        seed,caffe_txt_path_generator

    # Load images
    images_db = np.load(split_path).item()

    # Load patches
    patch_dataset = np.load(patches_db_path).item()

    # Generate labels
    labels = []
    for class_id in np.sort(np.unique(images_db['class'])):
        class_label = images_db['brand_model'][images_db['class'] == class_id][0]
        labels += [class_label + '\n']

    # Create labels file
    with open(caffe_txt_path_generator('labels'), 'w') as f:
        f.writelines(labels)

    # Generate list of jobs
    jobs_args = []

    for set_id, set_label in zip([1,2],['train','val']):
        caffe_txt_path = caffe_txt_path_generator(set_label)
        jobs_args += [{'seed':seed,
                      'txt_path':caffe_txt_path,
                       'images_db': images_db,
                       'patch_dataset': patch_dataset,
                       'set_id': set_id,
                       }]

    # Parallelize jobs
    pool = Pool(processes=num_processes)
    pool.map(create_txt,jobs_args)
    pool.close()
    pool.join()

    print('Completed.')

if __name__=='__main__':
    main()
