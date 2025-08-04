################################################################################
# Copyright 2019-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Quantum-Enhanced 3D CNN Training Script
# Fusion models for Atomic and molecular STructures (FAST)
################################################################################

import os
import sys
sys.stdout.flush()
sys.path.insert(0, "../common")
import argparse
import random
import numpy as np
import torch
import torch.nn as nn

from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Adam, RMSprop, lr_scheduler
from torch.utils.data import Dataset, DataLoader, Subset

# Import both classic and quantum models
from model import Model_3DCNN, strip_prefix_if_present
from quantum_enhanced_model import QuantumEnhanced3DCNN, create_quantum_model

from data_reader import Dataset_MLHDF
from img_util import GaussianFilter, Voxelizer3D

from tqdm import tqdm

from file_util import *

import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# program arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device-name", default="cuda:0", help="use cpu or cuda:0, cuda:1 ...")
parser.add_argument("--data-dir", default="data", help="dataset directory")
parser.add_argument("--dataset-type", type=float, default=1, help="ml-hdf version, (1: for fusion, 1.5: for cfusion 2: ml-hdf v2)")
parser.add_argument("--mlhdf-fn", default="pdbbind_2020_refined_train_voxelised.hdf", help="training ml-hdf path")
parser.add_argument("--csv-fn", default="", help="training csv file name")
parser.add_argument("--vmlhdf-fn", default="pdbbind_2020_refined_val_voxelised.hdf", help="validation ml-hdf path")
parser.add_argument("--vcsv-fn", default="", help="validation csv file name")
parser.add_argument("--model-path", default="", help="model checkpoint file path")
parser.add_argument("--complex-type", type=int, default=1, help="1: crystal, 2: docking")
parser.add_argument("--rmsd-weight", action='store_false', default=0, help="whether rmsd-based weighted loss is used or not")
parser.add_argument("--rmsd-threshold", type=float, default=2, help="rmsd cut-off threshold in case of docking data and/or --rmsd-weight is true")
parser.add_argument("--epoch-count", type=int, default=50, help="number of training epochs")
parser.add_argument("--batch-size", type=int, default=16, help="mini-batch size")
parser.add_argument("--learning-rate", type=float, default=4e-3, help="initial learning rate")
parser.add_argument("--decay-rate", type=float, default=0.95, help="learning rate decay")
parser.add_argument("--decay-iter", type=int, default=1, help="learning rate decay")
parser.add_argument("--checkpoint-dir", default="checkpoint_quantum/", help="checkpoint save directory")
parser.add_argument("--checkpoint-iter", type=int, default=1, help="number of epochs per checkpoint")
parser.add_argument("--verbose", type=int, default=0, help="print all input/output shapes or not")
parser.add_argument("--num-workers", type=int, default=0, help="number of workers for dataloader")
parser.add_argument("--multi-gpus", default=True, action="store_true", help="whether to use multi-gpus")

# Quantum-specific arguments
parser.add_argument("--use-quantum", action="store_true", default=False, help="use quantum-enhanced model")
parser.add_argument("--quantum-features", action="store_true", default=True, help="enable quantum feature layers")
parser.add_argument("--quantum-attention", action="store_true", default=True, help="enable quantum attention mechanism")
parser.add_argument("--quantum-qubits", type=int, default=6, help="number of qubits for quantum layers")
parser.add_argument("--quantum-layers", type=int, default=3, help="number of quantum circuit layers")

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
print(f"Quantum mode: {args.use_quantum}")

print(f"Train args: {args}")

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, y_pred, y_true, weight):
        return (weight * (y_pred - y_true) ** 2).mean()


def worker_init_fn(worker_id):
    np.random.seed(int(0))


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized tensors.
    Pads tensors to the maximum size in the batch.
    """
    if args.rmsd_weight:
        pdb_ids, x_tensors, y_tensors, w_tensors = zip(*batch)
        
        # Check if data is already voxelized (5D: batch, channels, depth, height, width)
        if len(x_tensors[0].shape) == 4:  # Already voxelized: [19, 48, 48, 48]
            x_batch = torch.stack(x_tensors, 0)
        else:  # Raw atomic data: [N_atoms, features]
            # Find the maximum number of atoms in this batch
            max_atoms = max(x.size(0) for x in x_tensors)
            batch_size = len(x_tensors)
            feat_dim = x_tensors[0].size(1)
            
            # Create padded tensor for x
            x_batch = torch.zeros(batch_size, max_atoms, feat_dim, dtype=torch.float32)
            for i, x in enumerate(x_tensors):
                x_batch[i, :x.size(0), :] = x
        
        # Stack y and w tensors (these should have the same size)
        y_batch = torch.stack(y_tensors, 0)
        w_batch = torch.stack(w_tensors, 0)
        
        return list(pdb_ids), x_batch, y_batch, w_batch
    else:
        pdb_ids, x_tensors, y_tensors = zip(*batch)
        
        # Check if data is already voxelized (5D: batch, channels, depth, height, width)
        if len(x_tensors[0].shape) == 4:  # Already voxelized: [19, 48, 48, 48]
            x_batch = torch.stack(x_tensors, 0)
        else:  # Raw atomic data: [N_atoms, features]
            # Find the maximum number of atoms in this batch
            max_atoms = max(x.size(0) for x in x_tensors)
            batch_size = len(x_tensors)
            feat_dim = x_tensors[0].size(1)
            
            # Create padded tensor for x
            x_batch = torch.zeros(batch_size, max_atoms, feat_dim, dtype=torch.float32)
            for i, x in enumerate(x_tensors):
                x_batch[i, :x.size(0), :] = x
        
        # Stack y tensors
        y_batch = torch.stack(y_tensors, 0)
        
        return list(pdb_ids), x_batch, y_batch


def check_voxelized(x_shape):
    # ANSI escape code for yellow foreground
    YELLOW = '\033[93m'
    # ANSI escape code to reset to default color
    RESET = '\033[0m'

    expected_shape = [19, 48, 48, 48]
    actual_shape = list(x_shape[1:])

    is_voxelized = len(x_shape) == 5 and actual_shape == expected_shape
    if args.verbose:
        print(f"IS VOXELIZED (Train) {is_voxelized}: {len(x_shape) == 5} and {actual_shape} and {actual_shape == expected_shape}")

    if not is_voxelized:
        print(f"{YELLOW} WARNING: (Train) - Voxelizer is not applied to the dataset, training time might increase significantly.!{RESET}")

    return is_voxelized

def train():
    import time
    start_time = time.time()

    # load dataset
    if args.complex_type == 1:
        is_crystal = True
    else:
        is_crystal = False
    train_dataset = Dataset_MLHDF(os.path.join(args.data_dir, args.mlhdf_fn), args.dataset_type, os.path.join(args.data_dir, args.csv_fn), is_crystal=is_crystal, rmsd_weight=args.rmsd_weight, rmsd_thres=args.rmsd_threshold, verbose=args.verbose)

    # if validation set is available
    val_dataset = None
    if len(args.vmlhdf_fn) > 0:
        val_dataset = Dataset_MLHDF(os.path.join(args.data_dir, args.vmlhdf_fn), args.dataset_type, os.path.join(args.data_dir, args.vcsv_fn), is_crystal=is_crystal, rmsd_weight=args.rmsd_weight, rmsd_thres=args.rmsd_threshold, verbose=args.verbose)

    # check multi-gpus
    num_workers = args.num_workers # 0 earlier, pickling issue with h5py

    # initialize data loader
    batch_count = len(train_dataset) // args.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=None, pin_memory=True, collate_fn=custom_collate_fn)
    
    # if validation set is available
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=None, pin_memory=True, collate_fn=custom_collate_fn)

    # define voxelizer, gaussian_filter
    voxelizer = Voxelizer3D(use_cuda=use_cuda, verbose=args.verbose)
    gaussian_filter = GaussianFilter(dim=3, channels=19, kernel_size=11, sigma=1, use_cuda=use_cuda)

    # define model - use quantum-enhanced model if requested
    if args.use_quantum:
        print("="*50)
        print("USING QUANTUM-ENHANCED 3D CNN MODEL")
        print("="*50)
        model = QuantumEnhanced3DCNN(
            feat_dim=19,
            output_dim=1,
            num_filters=[64, 128, 256],
            use_cuda=use_cuda,
            verbose=args.verbose,
            quantum_features=args.quantum_features,
            quantum_attention=args.quantum_attention
        )
    else:
        print("="*50)
        print("USING CLASSICAL 3D CNN MODEL")
        print("="*50)
        model = Model_3DCNN(use_cuda=use_cuda, verbose=args.verbose)
    
    if args.multi_gpus and cuda_count > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    if isinstance(model, (DistributedDataParallel, DataParallel)):
        model_to_save = model.module
    else:
        model_to_save = model

    # set loss, optimizer, decay, other parameters
    if args.rmsd_weight == True:
        loss_fn = WeightedMSELoss().float()
    else:
        loss_fn = nn.MSELoss().float()
    
    # Adjust learning rate for quantum models (they might need different optimization)
    if args.use_quantum:
        learning_rate = args.learning_rate * 0.5  # Start with lower LR for quantum layers
        print(f"Adjusted learning rate for quantum model: {learning_rate}")
    else:
        learning_rate = args.learning_rate
    
    optimizer = RMSprop(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_iter, gamma=args.decay_rate)

    # load model
    epoch_start = 0
    step = 0
    
    best_checkpoint_dict = None
    best_checkpoint_epoch = 0
    best_checkpoint_r2 = -9e9

    print(f"Starting training with {'Quantum-Enhanced' if args.use_quantum else 'Classical'} model...")

    for epoch_ind in range(epoch_start, args.epoch_count):
        vol_batch = torch.zeros((args.batch_size,19,48,48,48)).float().to(device)
        model.train()

        # Assuming `device` is your CUDA device
        losses = []
        r2_scores = []

        for batch_ind, batch in enumerate(train_dataloader):
            # transfer to GPU
            if args.rmsd_weight == True:
                pdb_id_batch, x_batch_cpu, y_batch_cpu, w_batch_cpu = batch
            else:
                pdb_id_batch, x_batch_cpu, y_batch_cpu = batch
            x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)

            if args.verbose:
                print(f"TRAIN:x_batch shape: {x_batch.shape}, y_batch shape: {y_batch.shape}")
            
            if check_voxelized(x_batch.shape):
                vol_batch = x_batch
            else:
                # voxelize into 3d volume
                bsize = x_batch.shape[0]
                for i in range(x_batch.shape[0]):
                    xyz, feat = x_batch[i,:,:3], x_batch[i,:,3:]
                    vol_batch[i,:,:,:,:] = voxelizer(xyz, feat)
                vol_batch = gaussian_filter(vol_batch)
            
            # forward training
            ypred_batch, _ = model(vol_batch[:x_batch.shape[0]])

            # compute loss
            if args.rmsd_weight == True:
                loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.float(), w_batch_cpu.float())
            else:
                loss = loss_fn(ypred_batch.float(), y_batch.float())

            losses.append(loss)

            # Compute R² on GPU
            ss_res = torch.sum((y_batch - ypred_batch) ** 2)
            ss_tot = torch.sum((y_batch - torch.mean(y_batch)) ** 2)
            r2_score = 1 - ss_res / ss_tot

            r2_scores.append(r2_score)
            
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.verbose:
                free_mem, total_mem = torch.cuda.mem_get_info()
                print("="*20, " Mem stats: step", step, " ", "="*20)
                print(f"Available GPU memory: {free_mem / 1e9:.2f} GB")
                print(f"Total GPU memory:     {total_mem / 1e9:.2f} GB")
                print("-"*55)
                print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                print(f"Max memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
                print(f"Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

            torch.cuda.empty_cache()
            import gc
            gc.collect()

            if args.verbose:
                model_type = "Quantum" if args.use_quantum else "Classical"
                print(f"{"="*20}, {model_type} epoch={epoch_ind}, step={step}, {"="*20}") 

        # update lr after epoch
        scheduler.step()

        if (epoch_ind+1) % args.checkpoint_iter == 0:
            loss_mean = torch.mean(torch.stack(losses)).item()
            r2_mean =torch.mean(torch.stack(r2_scores)).item()

            train_metrics = {
                "loss": loss_mean,
                "r2": r2_mean,
                "model_type": "quantum" if args.use_quantum else "classical"
            }
            
            model_type = "QUANTUM" if args.use_quantum else "CLASSICAL"
            tqdm.write(
                "[{}/{}] {} TRAINING: \tloss:{:0.4f}\n R2: {}"
                .format(
                    epoch_ind + 1,
                    args.epoch_count,
                    model_type,
                    loss_mean,
                    r2_mean,
                )
            )
            
            checkpoint_dict = checkpoint_model(model, val_dataloader,
                args.checkpoint_dir
                    + "/model-epoch-{}.pth".format(epoch_ind),
                optimizer, train_metrics)
            
            if checkpoint_dict["validate_dict"]["r2"] > best_checkpoint_r2:
                best_checkpoint_epoch = epoch_ind
                best_checkpoint_r2 = checkpoint_dict["validate_dict"]["r2"]
                best_checkpoint_dict = checkpoint_dict

        
    if best_checkpoint_dict is not None:
        model_type = "quantum" if args.use_quantum else "classical"
        print("best {} checkpoint epoch: %d, r2: %.4f" % (model_type, best_checkpoint_epoch, best_checkpoint_r2))
        torch.save(best_checkpoint_dict, args.checkpoint_dir + f"/best_{model_type}_checkpoint.pth")

    # close dataset
    train_dataset.close()
    if val_dataset:
        val_dataset.close()

    end_time = time.time()
    time_delta_seconds = end_time - start_time
    hours = time_delta_seconds // 3600
    minutes = (time_delta_seconds % 3600) // 60
    seconds = time_delta_seconds % 60

    model_type = "QUANTUM" if args.use_quantum else "CLASSICAL"
    print(f"{model_type} TRAINING took {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds.")
    print(f"TRAIN ARGS: {args}")


def validate(model, val_dataloader, epoch_ind):
    with torch.no_grad():
        model.eval()

        val_batch_count = len(val_dataloader.dataset) // args.batch_size
        losses = []
        r2_scores = []

        for batch_ind, batch in enumerate(val_dataloader):
            # transfer to GPU
            if args.rmsd_weight == True:
                pdb_id_batch, x_batch_cpu, y_batch_cpu, w_batch_cpu = batch
            else:
                pdb_id_batch, x_batch_cpu, y_batch_cpu = batch
            
            x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)

            if args.verbose:
                print(f"VAL: x_batch shape: {x_batch.shape}, y_batch shape: {y_batch.shape}")

            voxelizer = Voxelizer3D(use_cuda=use_cuda, verbose=args.verbose)
            gaussian_filter = GaussianFilter(dim=3, channels=19, kernel_size=11, sigma=1, use_cuda=use_cuda)
            
            vol_batch = torch.zeros((args.batch_size,19,48,48,48)).float().to(device)
            
            if check_voxelized(x_batch.shape):
                vol_batch = x_batch
            else:
                # voxelize into 3d volume
                for i in range(x_batch.shape[0]):
                    xyz, feat = x_batch[i,:,:3], x_batch[i,:,3:]
                    vol_batch[i,:,:,:,:] = voxelizer(xyz, feat)
                vol_batch = gaussian_filter(vol_batch)

            # forward training
            bsize = x_batch.shape[0]
            ypred_batch, _ = model(vol_batch[:x_batch.shape[0]])

            if args.rmsd_weight == True:
                loss_fn = WeightedMSELoss().float()
                loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.float(), w_batch_cpu.float())
            else:
                loss_fn = nn.MSELoss().float()
                loss = loss_fn(ypred_batch.float(), y_batch.float())

            losses.append(loss)

            ss_res = torch.sum((y_batch - ypred_batch) ** 2)
            ss_tot = torch.sum((y_batch - torch.mean(y_batch)) ** 2)
            r2_score = 1 - ss_res / ss_tot

            r2_scores.append(r2_score)

            if args.verbose:
                free_mem, total_mem = torch.cuda.mem_get_info()
                print("="*20, " Mem stats: batch_ind", batch_ind, " ", "="*20)
                print(f"Available GPU memory: {free_mem / 1e9:.2f} GB")
                print(f"Total GPU memory:     {total_mem / 1e9:.2f} GB")
                print("-"*55)
                print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                print(f"Max memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
                print(f"Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
                print("="*55)
            
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        loss_mean = torch.mean(torch.stack(losses)).item()
        r2_mean = torch.mean(torch.stack(r2_scores)).item()

        val_metrics = {
            "loss": loss_mean,
            "r2": r2_mean,
            "model_type": "quantum" if args.use_quantum else "classical"
        }
        
        model_type = "QUANTUM" if args.use_quantum else "CLASSICAL"        
        tqdm.write(
            "[{}/{}] {} VALIDATION: \tloss:{:0.4f}\n R2: {}"
            .format(
                epoch_ind+1, args.epoch_count,
                model_type,
                loss_mean,
                r2_mean,
            )
        )

    model.train() # restore to train mode
    return val_metrics

        
def checkpoint_model(model, dataloader, checkpoint_path, optimizer, train_dict):
    import os
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path))
    
    import re
    # Extract the filename from the path
    filename = os.path.basename(checkpoint_path)
    if match := re.match(r"model-epoch-(\d+)\.pth", filename):
        epoch_ind = int(match.group(1))

    validate_dict = validate(model, dataloader, epoch_ind)
    model.train()

    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "train_dict": train_dict,
        "validate_dict": validate_dict,
    }

    torch.save(checkpoint_dict, checkpoint_path)

    # return the computed metrics so it can be used to update the training loop
    return checkpoint_dict


def main():
    train()

if __name__ == "__main__":
    main()
