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

from model import Model_3DCNN, strip_prefix_if_present
from data_reader import Dataset_MLHDF
from img_util import GaussianFilter, Voxelizer3D

from tqdm import tqdm

from file_util import *

import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr



# seed all random number generators and set cudnn settings for deterministic: https://github.com/rusty1s/pytorch_geometric/issues/217
#random.seed(0)
#np.random.seed(0)
#torch.manual_seed(0)
#torch.cuda.manual_seed(0)
#torch.cuda.manual_seed_all(0)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False  # NOTE: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
#os.environ["PYTHONHASHSEED"] = "0"



# program arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device-name", default="cuda:0", help="use cpu or cuda:0, cuda:1 ...")
parser.add_argument("--data-dir", default="data", help="dataset directory")
parser.add_argument("--dataset-type", type=float, default=1, help="ml-hdf version, (1: for fusion, 1.5: for cfusion 2: ml-hdf v2)")
parser.add_argument("--mlhdf-fn", default="pdbbind2021_demo_train.hdf", help="training ml-hdf path")
parser.add_argument("--csv-fn", default="", help="training csv file name")
parser.add_argument("--vmlhdf-fn", default="pdbbind2021_demo_val.hdf", help="validation ml-hdf path")
parser.add_argument("--vcsv-fn", default="", help="validation csv file name")
parser.add_argument("--model-path", default="data/pdbbind2021_a1_demo_model_20250716.pth", help="model checkpoint file path")
parser.add_argument("--complex-type", type=int, default=1, help="1: crystal, 2: docking")
parser.add_argument("--rmsd-weight", action='store_false', default=0, help="whether rmsd-based weighted loss is used or not")
parser.add_argument("--rmsd-threshold", type=float, default=2, help="rmsd cut-off threshold in case of docking data and/or --rmsd-weight is true")
parser.add_argument("--epoch-count", type=int, default=50, help="number of training epochs")
parser.add_argument("--batch-size", type=int, default=50, help="mini-batch size")
parser.add_argument("--learning-rate", type=float, default=0.0007, help="initial learning rate")
parser.add_argument("--decay-rate", type=float, default=0.95, help="learning rate decay")
parser.add_argument("--decay-iter", type=int, default=100, help="learning rate decay")
parser.add_argument("--checkpoint-dir", default="checkpoint/", help="checkpoint save directory")
parser.add_argument("--checkpoint-iter", type=int, default=10000, help="number of epochs per checkpoint")
parser.add_argument("--verbose", type=int, default=0, help="print all input/output shapes or not")
parser.add_argument("--multi-gpus", default=False, action="store_true", help="whether to use multi-gpus")
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



class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, y_pred, y_true, weight):
        return (weight * (y_pred - y_true) ** 2).mean()


def worker_init_fn(worker_id):
    np.random.seed(int(0))

def train():

    # load dataset
    if args.complex_type == 1:
        is_crystal = True
    else:
        is_crystal = False
    dataset = Dataset_MLHDF(os.path.join(args.data_dir, args.mlhdf_fn), args.dataset_type, os.path.join(args.data_dir, args.csv_fn), is_crystal=is_crystal, rmsd_weight=args.rmsd_weight, rmsd_thres=args.rmsd_threshold)

    # if validation set is available
    val_dataset = None
    if len(args.vmlhdf_fn) > 0:
        val_dataset = Dataset_MLHDF(os.path.join(args.data_dir, args.vmlhdf_fn), args.dataset_type, os.path.join(args.data_dir, args.vcsv_fn), is_crystal=is_crystal, rmsd_weight=args.rmsd_weight, rmsd_thres=args.rmsd_threshold)

    # check multi-gpus
    num_workers = 0
    if args.multi_gpus and cuda_count > 1:
        num_workers = cuda_count

    # initialize data loader
    batch_count = len(dataset) // args.batch_size
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=None)
    
    # if validation set is available
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=None)

    # define voxelizer, gaussian_filter
    voxelizer = Voxelizer3D(use_cuda=use_cuda, verbose=args.verbose)
    gaussian_filter = GaussianFilter(dim=3, channels=19, kernel_size=11, sigma=1, use_cuda=use_cuda)

    # define model
    model = Model_3DCNN(use_cuda=use_cuda, verbose=args.verbose)
    #if use_cuda:
    #	model = model.cuda()
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
    #optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    optimizer = RMSprop(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_iter, gamma=args.decay_rate)

    # load model
    epoch_start = 0
    if valid_file(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        #checkpoint = torch.load(args.model_path)
        model_state_dict = checkpoint.pop("model_state_dict")
        strip_prefix_if_present(model_state_dict, "module.")
        model_to_save.load_state_dict(model_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_start = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print("checkpoint loaded: %s" % args.model_path)

    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    output_dir = os.path.dirname(args.model_path)

    step = 0
    
    best_checkpoint_dict = None
    best_checkpoint_epoch = 0
    best_checkpoint_r2 = -9e9

    for epoch_ind in range(epoch_start, args.epoch_count):
        vol_batch = torch.zeros((args.batch_size,19,48,48,48)).float().to(device)
        losses = []
        model.train()

        

        y_true_arr = np.zeros((len(dataset),), dtype=np.float32)
        y_pred_arr = np.zeros((len(dataset),), dtype=np.float32)

        for batch_ind, batch in enumerate(dataloader):
            # transfer to GPU
            if args.rmsd_weight == True:
                pdb_id_batch, x_batch_cpu, y_batch_cpu, w_batch_cpu = batch
            else:
                pdb_id_batch, x_batch_cpu, y_batch_cpu = batch
            x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
            print(f"TRAIN:x_batch shape: {x_batch.shape}, y_batch shape: {y_batch.shape}")
            
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
                loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.float())
                
            print("[%d/%d-%d/%d] training, loss: %.3f, lr: %.7f" % (epoch_ind+1, args.epoch_count, batch_ind+1, batch_count, loss.cpu().data.item(), optimizer.param_groups[0]['lr']))
            
            ytrue = y_batch.detach().cpu().float().numpy()[:,0]
            ypred = ypred_batch.detach().cpu().float().numpy()[:,0]
            y_true_arr[batch_ind*args.batch_size : batch_ind*args.batch_size+bsize] = ytrue
            y_pred_arr[batch_ind*args.batch_size : batch_ind*args.batch_size+bsize] = ypred
            
            step += 1

            losses.append(loss.cpu().data.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        print("[%d/%d] training, epoch loss: %.3f" % (epoch_ind+1, args.epoch_count, np.mean(losses)))

        if (epoch_ind+1) % args.checkpoint_iter == 0:
            train_metrics = compute_metrics(y_true_arr, y_pred_arr, float(loss))
        
            tqdm.write(
                "After Training: \tloss:{:0.4f}\n Metrics: {}"
                .format(
                    loss.cpu().data.numpy(), train_metrics
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
        print("best checkpoint epoch: %d, r2: %.4f" % (best_checkpoint_epoch, best_checkpoint_r2))
        torch.save(best_checkpoint_dict, args.checkpoint_dir + "/best_checkpoint.pth")

    # close dataset
    dataset.close()
    val_dataset.close()


def compute_metrics(ytrue_arr, ypred_arr, loss):
    print("Compute metrics shape debug: true/pred", ytrue_arr.shape, "/", ypred_arr.shape)
    rmse = math.sqrt(mean_squared_error(ytrue_arr, ypred_arr))
    mae = mean_absolute_error(ytrue_arr, ypred_arr)
    r2 = r2_score(ytrue_arr, ypred_arr)
    
    try:
        pearson, ppval = pearsonr(ytrue_arr, ypred_arr)
    except:
        pearson, ppval = float('nan'), float('nan')

    try:
        spearman, spval = spearmanr(ytrue_arr, ypred_arr)
    except:
        spearman, spval = float('nan'), float('nan')

    return {
        "loss": float(loss),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson": float(pearson),
        "spearman": float(spearman),
        "mae": float(mae),
        "label_mean": float(np.mean(ytrue_arr)),
        "label_stdev": float(np.std(ytrue_arr)),
        "pred_mean": float(np.mean(ypred_arr)),
        "pred_stdev": float(np.std(ypred_arr)),
    }

def validate(model, val_dataloader, epoch_ind):

    model.eval()

    y_true_arr = []
    y_pred_arr = []
    pdbid_list = []
    pose_list = []

    val_batch_count = len(val_dataloader.dataset) // args.batch_size

    y_true_arr = np.zeros((len(val_dataloader.dataset),), dtype=np.float32)
    y_pred_arr = np.zeros((len(val_dataloader.dataset),), dtype=np.float32)


    for batch_ind,batch in enumerate(val_dataloader):
        # transfer to GPU
        if args.rmsd_weight == True:
            pdb_id_batch, x_batch_cpu, y_batch_cpu, w_batch_cpu = batch
        else:
            pdb_id_batch, x_batch_cpu, y_batch_cpu = batch
        
        x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
        print(f"VAL: x_batch shape: {x_batch.shape}, y_batch shape: {y_batch.shape}")

        voxelizer = Voxelizer3D(use_cuda=use_cuda, verbose=args.verbose)
        gaussian_filter = GaussianFilter(dim=3, channels=19, kernel_size=11, sigma=1, use_cuda=use_cuda)
        
        vol_batch = torch.zeros((args.batch_size,19,48,48,48)).float().to(device)
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
            # loss = float('nan') # TODO: fix WeightMSELoss
        else:
            loss_fn = nn.MSELoss().float()
            loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.float())

        print("[%d/%d-%d/%d] validation, loss: %.3f" % (epoch_ind+1, args.epoch_count, batch_ind+1, val_batch_count, loss.cpu().data.item()))
        
        ytrue = y_batch.detach().cpu().float().numpy()[:,0]
        ypred = ypred_batch.detach().cpu().float().numpy()[:,0]

        y_true_arr[batch_ind*args.batch_size:batch_ind*args.batch_size+bsize] = ytrue
        y_pred_arr[batch_ind*args.batch_size:batch_ind*args.batch_size+bsize] = ypred




    print(f"Len y_true_arr: {len(y_true_arr)}")
    print(f"Len y_pred_arr: {len(y_pred_arr)}")

    
    val_metrics = compute_metrics(y_true_arr, y_pred_arr, float(loss))
            
    tqdm.write(
        "[{}/{}-{}/{}] Validation: \tloss:{:0.4f}\n Metrics: {}"
        .format(
            epoch_ind+1, args.epoch_count, batch_ind+1, val_batch_count,
            loss.cpu().data.numpy(),
            val_metrics
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
