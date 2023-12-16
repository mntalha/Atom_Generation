#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 21:06:31 2023

@author: talha
"""

from torch.utils.data import Dataset
import pandas as pd
from torch_geometric.data import Data
import torch
from torch_geometric.data import Batch
import numpy as np

from utils.data_utils import preprocess, add_scaled_lattice_prop
from utils.data_utils import get_scaler_from_data_list


#Macros
name = 'Formation energy train'
prop = 'energy_per_atom' # target
graph_method = 'crystalnn'
lattice_scale_method = 'scale_length'
niggli = True
primitive = False

class CrystDataset(Dataset):
    def __init__(self,
                name: str,
                path: str,
                prop: str,
                niggli: bool,
                primitive: bool,
                graph_method: str,
                lattice_scale_method: str,
                preprocess_workers: int = 4,
                built_path: str = "./built/",
                run_process = False,
                ):
        
        super().__init__()

        self.path = path
        self.name = name
        self.df = pd.read_csv(path)[:10]
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.preprocess_workers = preprocess_workers
        self.lattice_scaler = None
        self.scaler = None
        self.cached_data = None

        #csv file to graph data
        print(name, " being preprocessed...")
        if run_process:
            self.cached_data = preprocess(
                self.df,
                self.preprocess_workers,
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
                prop_list=[prop])
        
            #add lattice scaled property into cached data
            add_scaled_lattice_prop(self.cached_data, lattice_scale_method)

    def __len__(self) -> int:
        return len(self.cached_data)
    
    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        try:
            prop = self.scaler.transform(data_dict[self.prop])
        except:
            #print("Not defined scaler")
            prop = torch.tensor(data_dict[self.prop])
            
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']
        
        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
        )
        
        
        return data

def main():
    train_dataset = CrystDataset(name = name,
                            path = './data/train.csv',
                            prop = prop, 
                            niggli = niggli, 
                            primitive = primitive,
                            graph_method = graph_method,
                            run_process = True,
                            lattice_scale_method = lattice_scale_method)
    lattice_scaler = get_scaler_from_data_list(train_dataset.cached_data, key='scaled_lattice')
    scaler = get_scaler_from_data_list(train_dataset.cached_data, key=train_dataset.prop)

if __name__ == '__main__':
    main()  