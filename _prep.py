#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 21:24:42 2023

@author: talha
"""

#Libraries 
import torch
import random
import numpy as np
import os


#paths on server
# path = "/data/mnk7465/atomgen_1/"
# device_ = "cuda:0" # "cuda:0" "cuda:1" "cpu"

#paths on local
path = "/Users/talha/Desktop/atomgen_1/"
device_ = "cpu" # "cuda:0" "cuda:1" "cpu"

path_dataset = path + "data/light/" #light 
path_outputs = path + "outputs/"
path_graphics = path_outputs + "plots/"
path_training_results = path_outputs + "trainining_results"
path_trained_models =  path_outputs + "models"


# Dataset Information

datasets = {'train':{   'name': 'train',
                        'path': path_dataset + 'train.csv',
                        'prop': 'energy_per_atom',
                        'niggli': True,
                        'primitive': False,
                        'graph_method': 'crystalnn',
                        'preprocess_workers': 4,
                        'lattice_scale_method': 'scale_length'},
            
            'val':  {   'name': 'val',
                        'path': path_dataset + 'val.csv',
                        'prop': 'energy_per_atom',
                        'niggli': True,
                        'primitive': False,
                        'graph_method': 'crystalnn',
                        'preprocess_workers': 4,
                        'lattice_scale_method': 'scale_length'},
            
            'test': {   'name': 'test',
                        'path': path_dataset + 'test.csv',
                        'prop': 'energy_per_atom',
                        'niggli': True,
                        'primitive': False,
                        'graph_method': 'crystalnn',
                        'preprocess_workers': 4,
                        'lattice_scale_method': 'scale_length'}
                    } 

num_workers = {'train': 8, 'val': 8, 'test': 8}
batch_size = {'train': 32, 'val': 32, 'test': 32}

def get_initials():
    
    return (path_trained_models, path_training_results, path_graphics, 
                datasets, device_, num_workers,batch_size)


def set_seed(seed = 42):
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 

def save_pytorch_model(model, model_name, saved_path):
    
    # Check the planned path whether it is exist 
    isExist = os.path.exists(saved_path)
    if not isExist:
        print("Path you wished the model to be saved is not valid...")
        return False
    
    # model.state_dict():
        
    # save it
    path = os.path.join(saved_path, model_name+".ptk")   
    torch.save(model.state_dict(), path)
    
def load_pytorch_model(path, raw_model):   
    """
    take the transform .cuda() and .cpu() into consideration.
    """
    import os 
    # Check the file whether it is exist
    isExist = os.path.isfile(path)
    if not isExist:
        print("model couldnt found...")
        return False
    

    if torch.cuda.is_available():
        raw_model.load_state_dict(torch.load(path))
    else:
        raw_model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

    return raw_model
