################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Generate 3D representation in hdf5 format from input hdf5 (e.g., data/core_test.hdf) for 3D-CNN model training
################################################################################


import os
import sys

import argparse
import csv
import h5py
import numpy as np
import torch


from torch.utils.data import Dataset, DataLoader, Subset

from data_reader import Dataset_MLHDF
from img_util import GaussianFilter, Voxelizer3D




if __name__ != "__main__":
    print("This script is not intended to be imported.")
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--device-name", default="cuda:0", help="use cpu or cuda:0, cuda:1 ...")
# parser.add_argument("--mlhdf-fn", default="pdbbind_2020_refined_train.hdf", help="imput hdf path")
parser.add_argument("--mlhdf-fn", default="data\pdbbind_2020_refined_val.hdf", help="imput hdf path")

parser.add_argument("--csv-fn", default="", help="input csv file")
parser.add_argument("--rmsd-weight", action='store_false', default=0, help="whether rmsd-based weighted loss is used or not")
parser.add_argument("--rmsd-threshold", type=float, default=2, help="rmsd cut-off threshold in case of docking data and/or --rmsd-weight is true")
parser.add_argument("--verbose", type=int, default=0, help="print all input/output shapes or not")


parser.add_argument("--output-file", default="data/pdbbind_2020_refined_val_voxelised.hdf", help="output hdf path")
# parser.add_argument("--output-file", default="pdbbind_2020_refined_train_voxelised.hdf", help="output HDF filename")
args = parser.parse_args()

# set CUDA for PyTorch
use_cuda = torch.cuda.is_available() and args.device_name != "cpu"
cuda_count = torch.cuda.device_count()
if use_cuda:
    device = torch.device(args.device_name)
    torch.cuda.set_device(int(args.device_name.split(':')[1]))
else:
    device = torch.device("cpu")
print(f"Use cuda: {use_cuda}, count: {cuda_count}, device: {device}") 

def copy_good_protein(src_file, dest_file, protein, data):

    with h5py.File(src_file, 'r') as src:
        with h5py.File(dest_file, 'a') as dest:
            group = dest.create_group(protein)
            
            # Copy attributes
            for attr in src[protein].attrs:
                print(f"Copy attr: {attr}")
                group.attrs[attr] = src[protein].attrs[attr]

            # Create the necessary hierarchy
            pybel_group = group.create_group('pybel')
            processed_group = pybel_group.create_group('processed')
            pdbbind_group = processed_group.create_group('pdbbind')

            # Copy the group/dataset from the source to destination file
            pdbbind_group.create_dataset('data', data=data, dtype='float32')

            # Copy van_der_waals attribute if it exists
            if 'van_der_waals' in src[f'{protein}/pybel/processed/pdbbind'].attrs:
                print(f"Copy van_der_waals...")
                pdbbind_group.attrs['van_der_waals'] = src[f'{protein}/pybel/processed/pdbbind'].attrs['van_der_waals']



# define voxelizer, gaussian_filter
voxelizer = Voxelizer3D(use_cuda=use_cuda, verbose=args.verbose)
gaussian_filter = GaussianFilter(dim=3, channels=19, kernel_size=11, sigma=1, use_cuda=use_cuda)


import torch
import numpy as np
with h5py.File(args.mlhdf_fn, 'r') as f_mlhdf:
    for protein in f_mlhdf.keys():
        print(protein)

        data_cpu = torch.Tensor(f_mlhdf[protein]['pybel/processed/pdbbind/data'])
        data = data_cpu.to(device)

        vol_batch = torch.zeros((19,48,48,48)).float().to(device)

        xyz, feat = data[:,:3], data[:,3:]
        vol_batch[:,:,:,:] = voxelizer(xyz, feat)
        vol_batch = gaussian_filter(vol_batch)


        copy_good_protein(args.mlhdf_fn, args.output_file, protein, vol_batch.cpu().numpy())
        



