################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Spatial Graph Convolutional Network data loading utilities
################################################################################


import torch
import os.path as osp

from glob import glob
import multiprocessing as mp
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import DataLoader as GeometricDataLoader
from tqdm import tqdm
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch

import h5py

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


class PDBBindDataset(Dataset):
    def __init__(
        self,
        data_file,
        dataset_name,
        feature_type,
        preprocessing_type,
        use_docking=False,
        output_info=False,
        cache_data=True,
        h5_file_driver=None
    ):
        super(PDBBindDataset, self).__init__()
        self.dataset_name = dataset_name
        self.data_file = data_file
        self.feature_type = feature_type
        self.preprocessing_type = preprocessing_type
        self.use_docking = use_docking
        self.output_info = output_info
        self.cache_data = cache_data
        self.data_dict = {}
        self.data_list = []
        self.h5_file_driver = h5_file_driver

        if self.use_docking:
            with h5py.File(data_file, "r") as f:
                for name in list(f):
                    if self.feature_type in list(f[name]):
                        # Try to get affinity from top level first, then from pdbbind level
                        affinity = self._get_affinity(f, name)
                        if affinity is not None:
                            if self.preprocessing_type in f[name][self.feature_type]:
                                if self.dataset_name in list(f[name][self.feature_type][self.preprocessing_type]):
                                    for pose in f[name][self.feature_type][self.preprocessing_type][self.dataset_name]:
                                        self.data_list.append((name, pose, affinity))
        else:
            with h5py.File(data_file, "r", driver=self.h5_file_driver) as f:
                for name in list(f):
                    if self.feature_type in list(f[name]):
                        # Try to get affinity from top level first, then from pdbbind level  
                        affinity = self._get_affinity(f, name)
                        if affinity is not None:
                            self.data_list.append((name, 0, affinity))

    def _get_affinity(self, f, name):
        """
        Get affinity from either top level or pdbbind sublevel
        """
        # First try top level (where your extraction script stores it)
        if "affinity" in f[name].attrs:
            return np.asarray(f[name].attrs["affinity"]).reshape(1, -1)
        
        # Then try pdbbind sublevel
        pdbbind_path = f"{name}/{self.feature_type}/{self.preprocessing_type}/{self.dataset_name}"
        if pdbbind_path in f and "affinity" in f[pdbbind_path].attrs:
            return np.asarray(f[pdbbind_path].attrs["affinity"]).reshape(1, -1)
        
        print(f"Warning: No affinity found for {name}")
        return None

    def _get_vdw_radii(self, f, pdbid, pose=None):
        """
        Get VDW radii from the appropriate location
        """
        if self.use_docking and pose is not None:
            vdw_path = f"{pdbid}/{self.feature_type}/{self.preprocessing_type}/{self.dataset_name}"
            if "van_der_waals" in f[vdw_path][pose].attrs:
                return f[vdw_path][pose].attrs["van_der_waals"].reshape(-1, 1)
        else:
            vdw_path = f"{pdbid}/{self.feature_type}/{self.preprocessing_type}/{self.dataset_name}"
            if "van_der_waals" in f[vdw_path].attrs:
                return f[vdw_path].attrs["van_der_waals"].reshape(-1, 1)
        
        print(f"Warning: No VDW radii found for {pdbid}")
        return None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        if self.cache_data and item in self.data_dict.keys():
            return self.data_dict[item]

        pdbid, pose, affinity = self.data_list[item]

        with h5py.File(self.data_file, "r") as f:
            data_path = f"{pdbid}/{self.feature_type}/{self.preprocessing_type}/{self.dataset_name}"
            
            if not self.dataset_name in f[f"{pdbid}/{self.feature_type}/{self.preprocessing_type}"].keys():
                print(f"Dataset {self.dataset_name} not found for {pdbid}")
                return None

            if self.use_docking:
                data = f[data_path][pose]["data"]
                vdw_radii = self._get_vdw_radii(f, pdbid, pose)
            else:
                data = f[data_path]["data"]
                vdw_radii = self._get_vdw_radii(f, pdbid)

            if vdw_radii is None:
                print(f"No VDW radii found for {pdbid}, skipping")
                return None

            if self.feature_type == "pybel":
                coords = data[:, 0:3]
                node_feats = np.concatenate([vdw_radii, data[:, 3:]], axis=1)
            else:
                raise NotImplementedError

        # Create graph data
        dists = pairwise_distances(coords, metric="euclidean")
        edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dists).float())
        x = torch.from_numpy(node_feats).float()
        y = torch.FloatTensor(affinity).view(-1, 1)
        
        data = Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr.view(-1, 1), 
            y=y
        )

        if self.cache_data:
            result = (pdbid, pose, data) if self.output_info else data
            self.data_dict[item] = result
            return result
        else:
            return (pdbid, pose, data) if self.output_info else data


def worker_init_fn(worker_id):
    np.random.seed(int(0))


def test():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--docking-2007", default=False, action="store_true")
    parser.add_argument("--exp-2007", default=False, action="store_true")
    parser.add_argument("--exp-2016", default=False, action="store_true")
    args = parser.parse_args()

    from torch.utils.data import ConcatDataset

    docking_core_2007_dataset = PDBBindDataset(
        data_file="/p/lscratchh/jones289/data/pdbbind_2007_with_docking/core.hdf",
        feature_type="pybel",
        preprocessing_type="processed",
        dataset_name="docking",
        use_docking=True,
    )
    docking_refined_2007_dataset = PDBBindDataset(
        data_file="/p/lscratchh/jones289/data/pdbbind_2007_with_docking/refined.hdf",
        feature_type="pybel",
        preprocessing_type="processed",
        dataset_name="docking",
        use_docking=True,
    )

    exp_core_2007_dataset = PDBBindDataset(
        data_file="/p/lscratchh/jones289/data/pdbbind_2007_with_docking/core.hdf",
        feature_type="pybel",
        preprocessing_type="processed",
        dataset_name="pdbbind",
        use_docking=False,
    )
    exp_refined_2007_dataset = PDBBindDataset(
        data_file="/p/lscratchh/jones289/data/pdbbind_2007_with_docking/refined.hdf",
        feature_type="pybel",
        preprocessing_type="processed",
        dataset_name="pdbbind",
        use_docking=False,
    )

    exp_core_2016_dataset = PDBBindDataset(
        data_file="/p/lscratchh/jones289/data/pdbbind_2016_pybel_processed/core.hdf",
        feature_type="pybel",
        preprocessing_type="processed",
        dataset_name="pdbbind",
        use_docking=False,
    )
    exp_refined_2016_dataset = PDBBindDataset(
        data_file="/p/lscratchh/jones289/data/pdbbind_2016_pybel_processed/refined.hdf",
        feature_type="pybel",
        preprocessing_type="processed",
        dataset_name="pdbbind",
        use_docking=False,
    )
    exp_general_2016_dataset = PDBBindDataset(
        data_file="/p/lscratchh/jones289/data/pdbbind_2016_pybel_processed/general.hdf",
        feature_type="pybel",
        preprocessing_type="processed",
        dataset_name="pdbbind",
        use_docking=False,
    )

    if args.exp_2016:
        dataset_2016 = ConcatDataset(
            [exp_core_2016_dataset, exp_refined_2016_dataset, exp_general_2016_dataset]
        )
        dataloader_2016 = GeometricDataLoader(
            dataset_2016,
            batch_size=4,
            shuffle=True,
            num_workers=4,
            worker_init_fn=worker_init_fn,
        )
        print("{} experimental complexes in 2016 dataset".format(len(dataset_2016)))

        for batch in tqdm(dataloader_2016, desc="2016 experimental data"):
            pass

    if args.exp_2007:
        dataset_2007 = ConcatDataset([exp_core_2007_dataset, exp_refined_2007_dataset])
        dataloader_2007 = GeometricDataLoader(
            dataset_2007,
            batch_size=4,
            shuffle=True,
            num_workers=4,
            worker_init_fn=worker_init_fn,
        )
        print("{} experimental complexes in 2007 dataset".format(len(dataset_2007)))

        for batch in tqdm(dataloader_2007, desc="2007 experimental data"):
            pass

    if args.docking_2007:
        docking_2007_dataset = ConcatDataset(
            [docking_core_2007_dataset, docking_refined_2007_dataset]
        )
        docking_2007_dataloader = GeometricDataLoader(
            docking_2007_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=4,
            worker_init_fn=worker_init_fn,
        )
        print("{} docking complexes in 2007 dataset".format(len(docking_2007_dataset)))

        for batch in tqdm(docking_2007_dataloader, desc="2007 docking data"):
            pass


if __name__ == "__main__":

    test()
