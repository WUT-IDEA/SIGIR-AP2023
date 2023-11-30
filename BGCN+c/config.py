#!/usr/bin/env python3
# -*- coding: utf-8 -*-

CONFIG = {
    'name': '',
    'path': './data',
    'log': './log',
    'visual': './visual',
    'gpu_id': "0,1,2,3",
    'note': 'some_note',
    'model': 'BGCN',
    'dataset_name': 'MealRec-77',
    'task': 'tune',
    'eval_task': 'test',

    ## optimal hyperparameters 
    'lrs': [3e-4],
    'message_dropouts': [0],
    'node_dropouts': [0],
    'decays': [1e-7],

    ## hard negative sample and further train
    'sample': 'simple',
    #  'sample': 'hard',
    'hard_window': [0.7, 1.0], # top 30%
    'hard_prob': [0.4, 0.4], # probability 0.8
    'conti_train': 'model_file_from_simple_sample.pth',

    ## other settings
    'epochs': 200,
    'early': 50,
    'log_interval': 50,
    'test_interval': 1,
    'retry': 1,

    ## test path
    'test':['model_path_from_hard_sample']
}

print(CONFIG)

