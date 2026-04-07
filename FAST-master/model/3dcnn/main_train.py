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
from dataclasses import dataclass
import random
import numpy as np
import torch
import torch.nn as nn

from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Adam, AdamW, RMSprop, lr_scheduler
from torch.utils.data import Dataset, DataLoader, Subset

from model import Model_3DCNN, strip_prefix_if_present
from data_reader import Dataset_MLHDF
from img_util import GaussianFilter, Voxelizer3D, voxelize_batch

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

# Resolved at import time so paths are stable regardless of working directory
_DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data"))

@dataclass
class TrainArgs:
    device_name: str = "cuda:0"
    data_dir: str = _DATA_DIR
    dataset_type: float = 2  # 2=pre-voxelized (48,48,48,19); 1=atom-level pafnucy format
    mlhdf_fn: str = "3dcnn_train.hdf"  # pre-voxelized split of step5a.hdf
    csv_fn: str = ""                   # no CSV needed; Dataset_MLHDF reads all keys from HDF
    vmlhdf_fn: str = "3dcnn_val.hdf"
    vcsv_fn: str = ""
    model_path: str =  "checkpoints/best_model.pth" # os.path.join(_DATA_DIR, "checkpoints/best_model.pth")
    complex_type: int = 1
    rmsd_weight: bool = False
    rmsd_threshold: float = 2.0
    epoch_count: int = 50
    batch_size: int = 50
    learning_rate: float = 0.0007
    weight_decay: float = 1e-2  # AdamW weight decay; important regularizer on small datasets
    # decay_rate: float = 0.95  # StepLR
    # decay_iter: int = 100     # StepLR
    cosine_T_max: int = 50      # CosineAnnealingLR: steps to eta_min; set to epoch_count if stepping per epoch
    cosine_eta_min: float = 5e-7  # CosineAnnealingLR: minimum LR at trough
    checkpoint_dir: str = "checkpoints/3dcnn" # os.path.join(_DATA_DIR, "checkpoints/3dcnn")
    checkpoint_iter: int = 10000
    verbose: int = 0
    multi_gpus: bool = False
    train_from_scratch: bool = False


def get_args() -> TrainArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-name", default="cuda:0")
    parser.add_argument("--data-dir", default=_DATA_DIR)
    parser.add_argument("--dataset-type", type=float, default=2)
    parser.add_argument("--mlhdf-fn", default="3dcnn_train.hdf")
    parser.add_argument("--csv-fn", default="")
    parser.add_argument("--vmlhdf-fn", default="3dcnn_val.hdf")
    parser.add_argument("--vcsv-fn", default="")
    parser.add_argument("--model-path", default=os.path.join(_DATA_DIR, "checkpoints/best_model.pth"))
    parser.add_argument("--complex-type", type=int, default=1)
    parser.add_argument("--rmsd-weight", action='store_false', default=0)
    parser.add_argument("--rmsd-threshold", type=float, default=2)
    parser.add_argument("--epoch-count", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    # parser.add_argument("--decay-rate", type=float, default=0.95)
    # parser.add_argument("--decay-iter", type=int, default=100)
    parser.add_argument("--cosine-T-max", type=int, default=50)
    parser.add_argument("--cosine-eta-min", type=float, default=5e-7)
    parser.add_argument("--checkpoint-dir", default=os.path.join(_DATA_DIR, "checkpoints/3dcnn"))
    parser.add_argument("--checkpoint-iter", type=int, default=10000)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--multi-gpus", default=False, action="store_true")
    parser.add_argument("--train-from-scratch", default=False, action="store_true")
    ns = parser.parse_args()
    return TrainArgs(
        device_name=ns.device_name,
        data_dir=ns.data_dir,
        dataset_type=ns.dataset_type,
        mlhdf_fn=ns.mlhdf_fn,
        csv_fn=ns.csv_fn,
        vmlhdf_fn=ns.vmlhdf_fn,
        vcsv_fn=ns.vcsv_fn,
        model_path=ns.model_path,
        complex_type=ns.complex_type,
        rmsd_weight=bool(ns.rmsd_weight),
        rmsd_threshold=ns.rmsd_threshold,
        epoch_count=ns.epoch_count,
        batch_size=ns.batch_size,
        learning_rate=ns.learning_rate,
        weight_decay=ns.weight_decay,
        # decay_rate=ns.decay_rate,
        # decay_iter=ns.decay_iter,
        cosine_T_max=ns.cosine_T_max,
        cosine_eta_min=ns.cosine_eta_min,
        checkpoint_dir=ns.checkpoint_dir,
        checkpoint_iter=ns.checkpoint_iter,
        verbose=ns.verbose,
        multi_gpus=ns.multi_gpus,
        train_from_scratch=ns.train_from_scratch,
    )


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, y_pred, y_true, weight):
        return (weight * (y_pred - y_true) ** 2).mean()


def worker_init_fn(worker_id):
    np.random.seed(int(0))

def train(args: TrainArgs):

    # set CUDA for PyTorch
    use_cuda = torch.cuda.is_available() and args.device_name != "cpu"
    cuda_count = torch.cuda.device_count()
    if use_cuda:
        device = torch.device(args.device_name)
        torch.cuda.set_device(int(args.device_name.split(':')[1]))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    print(f"Use cuda: {use_cuda}, count: {cuda_count}, device: {device}")

    # load dataset
    is_crystal = args.complex_type == 1
    dataset = Dataset_MLHDF(os.path.join(args.data_dir, args.mlhdf_fn), args.dataset_type, os.path.join(args.data_dir, args.csv_fn), is_crystal=is_crystal, rmsd_weight=args.rmsd_weight, rmsd_thres=int(args.rmsd_threshold))

    # if validation set is available
    val_dataset = None
    if len(args.vmlhdf_fn) > 0:
        val_dataset = Dataset_MLHDF(os.path.join(args.data_dir, args.vmlhdf_fn), args.dataset_type, os.path.join(args.data_dir, args.vcsv_fn), is_crystal=is_crystal, rmsd_weight=args.rmsd_weight, rmsd_thres=int(args.rmsd_threshold))

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
    scaler = torch.amp.GradScaler('cuda')

    if isinstance(model, (DistributedDataParallel, DataParallel)):
        model_to_save = model.module
    else:
        model_to_save = model

    # set loss, optimizer, decay, other parameters
    if args.rmsd_weight == True:
        loss_fn = WeightedMSELoss().float()
    else:
        loss_fn = nn.MSELoss().float()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = RMSprop(model.parameters(), lr=args.learning_rate)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_iter, gamma=args.decay_rate)
    
    warmup_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters= args.epoch_count // 10)
    cos_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.cosine_T_max, args.cosine_eta_min)
    
    scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cos_scheduler], milestones=[10])
    

    # load model
    epoch_start = 0
    if args.train_from_scratch:
        print("train_from_scratch enabled: skipping checkpoint load")
    elif valid_file(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        #checkpoint = torch.load(args.model_path)
        model_state_dict = checkpoint.pop("model_state_dict")
        strip_prefix_if_present(model_state_dict, "module.")
        model_to_save.load_state_dict(model_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        import re as _re
        _m = _re.match(r"model-epoch-(\d+)\.pth", os.path.basename(args.model_path))
        epoch_start = int(_m.group(1)) + 1 if _m else checkpoint.get("epoch", 0)
        loss = checkpoint.get("loss", checkpoint.get("train_dict", {}).get("loss", 0.0))
        print("checkpoint loaded: %s" % args.model_path)

    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    output_dir = os.path.dirname(args.model_path)

    step = 0

    best_checkpoint_dict = None
    best_checkpoint_epoch = 0
    best_checkpoint_r2 = -9e9

    for epoch_ind in range(epoch_start, args.epoch_count):
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

            bsize = x_batch.shape[0]
            with torch.autocast(device_type='cuda'):
                if args.dataset_type == 2:
                    # already voxelized: x_batch is (B, 19, 48, 48, 48)
                    vol_batch = x_batch
                else:
                    vol_batch = voxelize_batch(x_batch[:, :, :3], x_batch[:, :, 3:])
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

            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        scheduler.step() # lr updates once per epoch rather than batch

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
                optimizer, train_metrics, args, device, use_cuda)

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

    return best_checkpoint_dict


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

def validate(model, val_dataloader, epoch_ind, args: TrainArgs, device, use_cuda):

    model.eval()

    val_batch_count = len(val_dataloader.dataset) // args.batch_size

    y_true_arr = np.zeros((len(val_dataloader.dataset),), dtype=np.float32)
    y_pred_arr = np.zeros((len(val_dataloader.dataset),), dtype=np.float32)

    val_gaussian = GaussianFilter(dim=3, channels=19, kernel_size=11, sigma=1, use_cuda=use_cuda)
    for batch_ind, batch in enumerate(val_dataloader):
        # transfer to GPU
        if args.rmsd_weight == True:
            pdb_id_batch, x_batch_cpu, y_batch_cpu, w_batch_cpu = batch
        else:
            pdb_id_batch, x_batch_cpu, y_batch_cpu = batch

        x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)

        with torch.autocast(device_type='cuda'):
            bsize = x_batch.shape[0]
            if args.dataset_type == 2:
                vol_batch = x_batch
            else:
                vol_batch = voxelize_batch(x_batch[:, :, :3], x_batch[:, :, 3:])
                vol_batch = val_gaussian(vol_batch)

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



def checkpoint_model(model, dataloader, checkpoint_path, optimizer, train_dict, args: TrainArgs, device, use_cuda):
    import re
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path))

    filename = os.path.basename(checkpoint_path)
    if match := re.match(r"model-epoch-(\d+)\.pth", filename):
        epoch_ind = int(match.group(1))

    validate_dict = validate(model, dataloader, epoch_ind, args, device, use_cuda)
    model.train()

    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args) if hasattr(args, '__dict__') else args.__dict__,
        "train_dict": train_dict,
        "validate_dict": validate_dict,
    }

    torch.save(checkpoint_dict, checkpoint_path)

    # return the computed metrics so it can be used to update the training loop
    return checkpoint_dict


def main():
    args = get_args()
    train(args)

if __name__ == "__main__":
    main()
