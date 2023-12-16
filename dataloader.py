#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 21:06:31 2023

@author: talha
"""
from torch_geometric.loader import DataLoader
import pdb
from typing import Optional, Sequence
from dataset import CrystDataset
from utils.data_utils import get_scaler_from_data_list
import torch
from pathlib import Path
import numpy as np
import random
import os
import torch
def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)

class CrystDataModule():
    def __init__(
        self,
        datasets: dict,
        num_workers: int,
        batch_size: dict,
        scaler_path="./built/"
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.scaler_path = scaler_path

        self.train_dataset = None
        self.val_datasets = None
        self.test_datasets = None


    def prepare_data(self) -> None:
        
        self.train_dataset = CrystDataset(
                name=self.datasets["train"]["name"],
                path=self.datasets["train"]["path"],
                prop=self.datasets["train"]["prop"],
                niggli=self.datasets["train"]["niggli"],
                primitive=self.datasets["train"]["primitive"],
                graph_method=self.datasets["train"]["graph_method"],
                preprocess_workers=self.num_workers["train"],
                run_process=True,
                lattice_scale_method=self.datasets["train"]["lattice_scale_method"]
        )
        self.get_scaler(self.train_dataset)

        self.val_dataset = CrystDataset(
                name=self.datasets["val"]["name"],
                path=self.datasets["val"]["path"],
                prop=self.datasets["val"]["prop"],
                niggli=self.datasets["val"]["niggli"],
                primitive=self.datasets["val"]["primitive"],
                graph_method=self.datasets["val"]["graph_method"],
                preprocess_workers=self.num_workers["val"],
                run_process=True,
                lattice_scale_method=self.datasets["val"]["lattice_scale_method"]
        )

        self.test_dataset = CrystDataset(
                name=self.datasets["test"]["name"],
                path=self.datasets["test"]["path"],
                prop=self.datasets["test"]["prop"],
                niggli=self.datasets["test"]["niggli"],
                primitive=self.datasets["test"]["primitive"],
                graph_method=self.datasets["test"]["graph_method"],
                preprocess_workers=self.num_workers["val"],
                run_process=True,
                lattice_scale_method=self.datasets["test"]["lattice_scale_method"]
        )
    

    def get_scaler(self, train_dataset: CrystDataset) -> None:

        if self.scaler_path is None:
            self.scaler = None
            print("No scaler path provided. Skipping scaler loading.")
            return

        # load lattice scaler if exists
        lattice_scaler_path = os.path.join(self.scaler_path, 'lattice_scaler.pt')
        if os.path.exists(lattice_scaler_path):
            self.lattice_scaler = torch.load(lattice_scaler_path)
        else:
            self.lattice_scaler = get_scaler_from_data_list( 
                                        train_dataset.cached_data, 
                                        'scaled_lattice')
            torch.save(self.lattice_scaler, lattice_scaler_path)

        # load property scaler if exists
        scaler_path = os.path.join(self.scaler_path, 'prop_scaler.pt')
        if os.path.exists(scaler_path):
            self.scaler = torch.load(scaler_path)
        else:
            self.scaler = get_scaler_from_data_list(train_dataset.cached_data,
                                                    train_dataset.prop)
            torch.save(self.scaler, scaler_path)
            
    def setup(self):

        # set up train, val, test datasets
        self.prepare_data()

        # set up train, val, test scalers
        self.train_dataset.lattice_scaler = self.lattice_scaler
        self.train_dataset.scaler = self.scaler
        self.val_dataset.lattice_scaler = self.lattice_scaler
        self.val_dataset.scaler = self.scaler
        self.test_dataset.lattice_scaler = self.lattice_scaler
        self.test_dataset.scaler = self.scaler
        

        # set up train, val, test dataloaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size["train"],
            shuffle=True,
            num_workers=self.num_workers["train"],
            worker_init_fn=worker_init_fn,
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size["val"],
            shuffle=False,
            num_workers=self.num_workers["val"],
            worker_init_fn=worker_init_fn,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size["test"],
            shuffle=False,
            num_workers=self.num_workers["test"],
            worker_init_fn=worker_init_fn,
        )


        return train_loader, val_loader, test_loader

def main():
    datasets = { 'train': {'name': 'train',
                            'path': './data/train.csv',
                            'prop': 'energy_per_atom',
                            'niggli': True,
                            'primitive': False,
                            'graph_method': 'crystalnn',
                            'preprocess_workers': 4,
                            'lattice_scale_method': 'scale_length'},
                   'val':  {'name': 'val',
                            'path': './data/val.csv',
                            'prop': 'energy_per_atom',
                            'niggli': True,
                            'primitive': False,
                            'graph_method': 'crystalnn',
                            'preprocess_workers': 4,
                            'lattice_scale_method': 'scale_length'},
                    'test': {'name': 'test',
                            'path': './data/test.csv',
                            'prop': 'energy_per_atom',
                            'niggli': True,
                            'primitive': False,
                            'graph_method': 'crystalnn',
                            'preprocess_workers': 4,
                            'lattice_scale_method': 'scale_length'}
                    }   
    num_workers = {'train': 8, 'val': 8, 'test': 8}
    batch_size = {'train': 5, 'val': 5, 'test': 5}
    data_module = CrystDataModule(datasets, num_workers, batch_size)
    train_loader, val_loader, test_loader = data_module.setup()                            
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = main()

    #test dataloaders
    for batch in train_loader:
        print(batch.y)
    for batch in val_loader:
        print(batch.y)
    for batch in test_loader:
        print(batch.y)
        
