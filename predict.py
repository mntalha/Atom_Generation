#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 20:51:16 2023

@author: talha
"""

# batch_size = 32
# epoch_number = 50
# learning_rate = 3e-4
# weight_decay = 3e-6
# model_name = None
# device_ = "cuda:0" # "cuda:0" "cuda:1" "cpu"


# #paths on server
# path = "/data/mnk7465/ebsd/"
# #path_dataset = path + "dataset/original_data_0/" #concat
# path_dataset = path + "dataset/concat_data_0/" #
# path_outputs = path + "outputs/"
# path_graphics = path_outputs + "plots/"
# path_training_results = path_outputs + "trainining_results"
# path_trained_models =  path_outputs + "models"

#Libraries 
import torch.optim as optim
import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt


# Dataloader
from dataloader import CrystDataModule


device_ = "cpu" # "cuda:0" "cuda:1" "cpu"

#
if device_ != None:
    device = torch.device(device_)
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
