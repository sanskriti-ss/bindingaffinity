################################################################################
# Copyright 2019-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# File utility functions
################################################################################

import os
import sys
import csv
import h5py
import torch
import numpy as np

from torch.utils.data import Dataset


# Note: if csv_path exists, the following columns should be included:
#pdb_id/compound_id, pose_id, rmsd, affinity
# if csv doesn't exist, rmsd and affinity become unknown -> only testing is available (no evaluation)

class Dataset_MLHDF(Dataset):
    def __init__(self, mlhdf_path, mlhdf_ver, csv_path="", is_crystal=False, rmsd_weight=False, rmsd_thres=2, max_atoms=2000, feat_dim=22):
        super(Dataset_MLHDF, self).__init__()
        self.mlhdf_ver = mlhdf_ver
        self.mlhdf_path = mlhdf_path
        self.csv_path = csv_path
        self.is_crystal = is_crystal
        self.rmsd_weight = rmsd_weight
        self.rmsd_thres = rmsd_thres
        self.max_atoms = max_atoms
        self.feat_dim = feat_dim

        self.mlhdf = h5py.File(self.mlhdf_path, 'r')
        self.data_info_list = []

        if self.mlhdf_ver == 1: # for fusion model
            try:
                with open(self.csv_path, 'r') as fp:
                    csv_reader = csv.reader(fp, delimiter=',')
                    next(csv_reader)
                    for row in csv_reader:
                        # Your CSV format: ligand_id, file_prefix, label, train_test_split, ...
                        ligand_id = row[0]
                        file_prefix = row[1] 
                        affinity = float(row[2])  # This is the actual label (binding affinity)
                        train_test_split = row[3]
                        
                        # For crystal structures, no RMSD filtering - use all samples
                        self.data_info_list.append([ligand_id, 0, 0.0, affinity])

            except IOError:
                for comp_id in self.mlhdf.keys():
                    if self.is_crystal:
                        self.data_info_list.append([comp_id, 0, 0, 0])
                    else:
                        pose_ids = self.mlhdf[comp_id]["pybel"]["processed"]["docking"].keys()
                        for pose_id in pose_ids:
                            self.data_info_list.append([comp_id, pose_id, 0, 0])
        elif self.mlhdf_ver == 1.5: # for cfusion model
            if is_crystal:
                for pdbid in self.mlhdf["regression"].keys():
                    affinity = float(self.mlhdf["regression"][pdbid].attrs["affinity"])
                    self.data_info_list.append([pdbid, 0, 0, affinity])
            else:
                print("not supported!")

        self.labels = self._get_labels()

    def close(self):
        self.mlhdf.close()

    def _get_labels(self):
        labels = []
        for _, _, _, affinity in self.data_info_list:
            labels.append(affinity)
        return np.array(labels, dtype=np.float32)

    def __len__(self):
        count = len(self.data_info_list)
        return count

    def __getitem__(self, idx):
        pdbid, poseid, rmsd, affinity = self.data_info_list[idx]

        if self.mlhdf_ver == 1:
            if self.is_crystal:
                # Check if this is a 3D CNN HDF file (direct structure) or original structure
                # For 3D CNN HDF files, the data is stored directly as a dataset
                hdf_item = self.mlhdf[pdbid]
                if isinstance(hdf_item, h5py.Group) and "pybel" in hdf_item:
                    # Original structure - point cloud data
                    mlhdf_ds = hdf_item["pybel"]["processed"]["pdbbind"]["data"]
                    actual_data = mlhdf_ds[:]
                    data = np.zeros((self.max_atoms, self.feat_dim), dtype=np.float32)
                    data[:actual_data.shape[0],:] = actual_data
                    x = torch.tensor(data, dtype=torch.float32)
                else:
                    # 3D CNN HDF file - data is stored directly as 3D volume
                    # Shape: (48, 48, 48, 19) -> convert to (19, 48, 48, 48) for PyTorch
                    actual_data = hdf_item[:]
                    x = torch.tensor(actual_data.transpose(3, 0, 1, 2), dtype=torch.float32)  # Move channels to first dimension
            else:
                hdf_item = self.mlhdf[pdbid]
                if isinstance(hdf_item, h5py.Group) and "pybel" in hdf_item:
                    # Original structure - point cloud data
                    mlhdf_ds = hdf_item["pybel"]["processed"]["docking"][poseid]
                    actual_data = mlhdf_ds["data"][:]
                    data = np.zeros((self.max_atoms, self.feat_dim), dtype=np.float32)
                    data[:actual_data.shape[0],:] = actual_data
                    x = torch.tensor(data, dtype=torch.float32)
                else:
                    # 3D CNN HDF file - data is stored directly as 3D volume
                    actual_data = hdf_item[:]
                    x = torch.tensor(actual_data.transpose(3, 0, 1, 2), dtype=torch.float32)  # Move channels to first dimension
        elif self.mlhdf_ver == 1.5:
            if self.is_crystal:
                mlhdf_ds = self.mlhdf["regression"][pdbid]["pybel"]["processed"]
                actual_data = mlhdf_ds["pdbbind_sgcnn"]["data0"][:]
                data = np.zeros((self.max_atoms, self.feat_dim), dtype=np.float32)
                data[:actual_data.shape[0],:] = actual_data
                x = torch.tensor(data, dtype=torch.float32)

        y = torch.tensor(np.expand_dims(affinity, axis=0), dtype=torch.float32)

        if self.rmsd_weight == True:
            data_w = 0.5 + self.rmsd_thres - rmsd
            w = torch.tensor(np.expand_dims(data_w, axis=0), dtype=torch.float32)
            return x, y, w
        else:
            return x, y