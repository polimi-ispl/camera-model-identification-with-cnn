# -*- coding: UTF-8 -*-
"""
First Steps Towards Camera Model Identification with Convolutional Neural Networks
@author: Luca Bondi (luca.bondi@polimi.it)

@brief: PoliMi split for Dresden Dataset tailored at Camera Model Identification
@description:
    Considered only the models having more than one instance: 
    'Canon_Ixus70'
    'Casio_EX-Z150'
    'FujiFilm_FinePixJ50'
    'Kodak_M1063'
    'Nikon_CoolPixS710'
    'Nikon_D200'
    'Nikon_D70'
    'Nikon_D70s'
    'Olympus_mju-1050SW'
    'Panasonic_DMC-FZ50'
    'Pentax_OptioA40'
    'Praktica_DCZ5.9'
    'Ricoh_GX100'
    'Rollei_RCP-7325XS'
    'Samsung_L74wide'
    'Samsung_NV15'
    'Sony_DSC-H50'
    'Sony_DSC-T77'
    'Sony_DSC-W170'
    
    Fair splitting policy: 
    Training set: N scenes, instances 1+ for all models.
    Validation set: M different scenes, instance 1+ for all models.
    Test set: remaining 10 scenes, 0 instances
"""


import numpy as np
import os
import argparse

def main():
    # Parameters ---
    from params import images_db_path,split_path,seed
    
    train_scenes = 61
    test_scenes = 10
    test_instance = 0

    # Load db
    images_db = np.load(images_db_path).item()
    num_shots = len(images_db['path'])
    
    # Assign unique id to scenes
    images_db['scene'] = np.asarray([ position_num*10+motive_num for position_num,motive_num in zip(images_db['position_num'],images_db['motive_num'])])

    # Filter out models with a single instance ---
    multiple_instance_models = np.unique(images_db['model'][images_db['instance']>0])
    multiple_instance_idxs = np.asarray(
        [i for i in range(num_shots) if images_db['model'][i] in multiple_instance_models]).astype(np.uint32)

    for key in images_db.keys():
        images_db[key] = images_db[key][multiple_instance_idxs]
    num_shots = len(images_db['path'])

    # Merge Nikon_D70(s)
    db_brand_model = images_db['brand_model']
    db_brand_model[db_brand_model=='Nikon_D70s'] = 'Nikon_D70'

    # Determine numeric labels for classes
    brand_models = np.sort(np.unique(db_brand_model))
    brand_models_class = np.arange(len(brand_models))
    
    # Assign numeric class to each shot ---
    images_db['class'] = -1*np.ones(num_shots,np.int32)
    for brand_model, brand_model_class in zip(brand_models,brand_models_class):
        idxs = db_brand_model == brand_model
        images_db['class'][idxs] = brand_model_class

    ## Determine mask for train_val and test instances ---
    train_val_mask_instances = images_db['instance'] != test_instance
    test_mask_instances = images_db['instance'] == test_instance
    assert(np.sum(np.logical_and(train_val_mask_instances,test_mask_instances)) == 0)


    ## Generate splits ---
    print('Generating random split')
    np.random.seed(seed)

    scene_num_unique = np.unique(images_db['scene'])

    '''
    1 - train
    2 - val
    3 - test
    '''
    images_db['split'] = np.zeros(num_shots,np.uint8)

    generate = True
    while(generate):

        random_scene_ids = np.random.permutation(scene_num_unique)
        test_scene_ids = random_scene_ids[-test_scenes:]

        for test_motive_id in test_scene_ids:
            images_db['split'][np.logical_and(images_db['scene'] == test_motive_id, images_db['instance'] == test_instance)] = 3

        train_motives_ids = random_scene_ids[:train_scenes]
        for train_motive_id in train_motives_ids:
            images_db['split'][np.logical_and(images_db['scene'] == train_motive_id, images_db['instance'] != test_instance)] = 1

        val_motives_ids = random_scene_ids[train_scenes:-test_scenes]
        for val_motive_id in val_motives_ids:
            images_db['split'][np.logical_and(images_db['scene'] == val_motive_id, images_db['instance'] != test_instance)] = 2

        # Check that each class is represented in every split
        generate = False
        for split_id in [1, 2, 3]:
            split_mask = images_db['split'] == split_id
            for shot_class in brand_models_class:
                generate = np.sum(np.logical_and(split_mask, images_db['class'] == shot_class)) == 0
                if generate:
                    print('Some models are not represented. Regenerating the split')
                    break
            if generate:
                break

    assert((np.unique(images_db['split']) == np.asarray([0,1,2,3])).all())
    print('Train shots: {:}'.format(np.sum(images_db['split'] == 1).astype(np.int)))
    print('Val shots: {:}'.format(np.sum(images_db['split'] == 2).astype(np.int)))
    print('Test shots: {:}'.format(np.sum(images_db['split'] == 3).astype(np.int)))
    print('Unassigned shots: {:}'.format(np.sum(images_db['split'] == 0).astype(np.int)))

    # Saving ---
    print('Saving to: ' + split_path)
    np.save(split_path, images_db)

    print('Completed')

if __name__ == '__main__':
    main()
