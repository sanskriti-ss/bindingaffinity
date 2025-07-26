################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Spatial Graph Convolutional Network training script
################################################################################


import os
import itertools
from glob import glob
import multiprocessing as mp
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import init
from torch.optim import Adam, lr_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch_geometric.data import DataLoader as GeometricDataLoader
from torch_geometric.nn import DataParallel as GeometricDataParallel
from torch_geometric.data import DataListLoader
from data_utils import PDBBindDataset
from model import PotentialNetParallel, GraphThreshold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, SubsetRandomSampler


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--device-name", default="cuda:0", help="use cpu or cuda:0, cuda:1 ...")

parser.add_argument(
    "--checkpoint", type=bool, default=True, help="boolean flag for checkpoints"
)
parser.add_argument(
    "--checkpoint-dir", default=os.getcwd(), help="path to store model checkpoints"
)
parser.add_argument(
    "--checkpoint-iter", default=10000, type=int, help="number of epochs per checkpoint"
)
parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
parser.add_argument(
    "--num-workers", default=1, type=int, help="number of workers for dataloader"
)
parser.add_argument(
    "--batch-size", default=32, type=int, help="batch size to use for training"
)
parser.add_argument(
    "--lr", default=1e-3, type=float, help="learning rate to use for training"
)
parser.add_argument(
    "--preprocessing-type",
    type=str,
    choices=["raw", "processed"],
    default="processed",
    help="idicate raw pdb or (chimera) processed",
)
parser.add_argument(
    "--feature-type",
    type=str,
    choices=["pybel", "rdkit"],
    default="pybel",
    help="indicate pybel (openbabel) or rdkit features",
)
parser.add_argument(
    "--dataset-name", type=str, default="pdbbind", help="indicate dataset name"
)  # NOTE: this should probably just consist of a set of choices




parser.add_argument("--covalent-gather-width", type=int, default=16)
parser.add_argument("--non-covalent-gather-width", type=int, default=12)
parser.add_argument("--covalent-k", type=int, default=2)
parser.add_argument("--non-covalent-k", type=int, default=2)
parser.add_argument("--covalent-threshold", type=float, default=1.5)
parser.add_argument("--non-covalent-threshold", type=float, default=4.5)
parser.add_argument("--train-data", type=str, default="data/train.hdf", nargs="+")
parser.add_argument("--val-data", type=str, default="data/val.hdf", nargs="+")
parser.add_argument("--use-docking", default=False, action="store_true")
args = parser.parse_args()

# seed all random number generators and set cudnn settings for deterministic: https://github.com/rusty1s/pytorch_geometric/issues/217
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # NOTE: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
os.environ["PYTHONHASHSEED"] = "0"

# set CUDA for PyTorch
use_cuda = torch.cuda.is_available() and args.device_name != "cpu"
cuda_count = torch.cuda.device_count()
if use_cuda:
    device = torch.device(args.device_name)
    torch.cuda.set_device(int(args.device_name.split(':')[1]))
else:
    device = torch.device("cpu")
print(f"Use cuda: {use_cuda}, count: {cuda_count}, device: {device}") 

def worker_init_fn(worker_id):
    np.random.seed(int(0))


def collate_fn_none_filter(batch):
    return [x for x in batch if x is not None]


def train():

    # set the input channel dims based on featurization type
    if args.feature_type == "pybel":
        feature_size = 20
    else:
        feature_size = 75

    print("found {} datasets in input train-data".format(len(args.train_data)))
    train_dataset_list = []
    val_dataset_list = []

    for data in args.train_data:
        train_dataset_list.append(
            PDBBindDataset(
                data_file=data,
                dataset_name=args.dataset_name,
                feature_type=args.feature_type,
                preprocessing_type=args.preprocessing_type,
                output_info=True,
                use_docking=args.use_docking,
            )
        )

    for data in args.val_data:
        val_dataset_list.append(
            PDBBindDataset(
                data_file=data,
                dataset_name=args.dataset_name,
                feature_type=args.feature_type,
                preprocessing_type=args.preprocessing_type,
                output_info=True,
                use_docking=args.use_docking,
            )
        )

    train_dataset = ConcatDataset(train_dataset_list)
    val_dataset = ConcatDataset(val_dataset_list)

    train_dataloader = DataListLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )  # just to keep batch sizes even, since shuffling is used

    val_dataloader = DataListLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )

    tqdm.write("{} complexes in training dataset".format(len(train_dataset)))
    tqdm.write("{} complexes in validation dataset".format(len(val_dataset)))

    potential_net_parallel = PotentialNetParallel(
            in_channels=feature_size,
            out_channels=1,
            covalent_gather_width=args.covalent_gather_width,
            non_covalent_gather_width=args.non_covalent_gather_width,
            covalent_k=args.covalent_k,
            non_covalent_k=args.non_covalent_k,
            covalent_neighbor_threshold=args.covalent_threshold,
            non_covalent_neighbor_threshold=args.non_covalent_threshold,
            always_return_hidden_feature=True,
        ).float()


    if torch.cuda.device_count() > 1:
        model = GeometricDataParallel(
            potential_net_parallel, device_ids=list(range(torch.cuda.device_count()))
            ).float()
        # model.to(0)
    else:
        model = potential_net_parallel.to('cuda' if torch.cuda.is_available() else 'cpu')

    model.train()
    tqdm.write(str(model))
    tqdm.write(
        "{} trainable parameters.".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    tqdm.write(
        "{} total parameters.".format(sum(p.numel() for p in model.parameters()))
    )

    loss_fn = nn.MSELoss().float()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_checkpoint_dict = None
    best_checkpoint_epoch = 0
    best_checkpoint_step = 0
    best_checkpoint_r2 = -9e9
    step = 0

    # if not os.path.exists(os.path.dirname('fc-layer/')):
    #     os.makedirs(os.path.dirname('fc-layer/'))


    for epoch in range(args.epochs):
        losses = []
        r2_scores = []
        # print(f"MANVI: Epoch {epoch}")
        # print(f"Length of train_dataloader: {len(train_dataloader)}")

        for batch in tqdm(train_dataloader):
            batch = [x for x in batch if x is not None]
            if len(batch) < 1:
                print("TRAIN:empty batch, skipping to next batch")
                continue

            data_cpu = [x[2] for x in batch]
            y_batch_cpu = torch.cat([x[2].y for x in batch])

            data = data_cpu
            # data = data_cpu.to(device)
            y_batch = y_batch_cpu.to(device)


            from torch_geometric.data import Batch
            data = Batch.from_data_list(data)
            # data = 
            yo_ = model(data)

            if len(yo_) > 1:
                import numpy as np
                ypred_batch, avg_covalent_x, avg_non_covalent_x, pool_x, fc0_x, fc1_x = yo_
                # np.save(f'fc-layer/fc-0-epoch-{epoch}-step-{step}.npy', fc0_x.cpu().detach().numpy())
                # np.save(f'fc-layer/fc-1-epoch-{epoch}-step-{step}.npy', fc1_x.cpu().detach().numpy())
            else:
                ypred_batch, *_ = yo_

            


            loss = loss_fn(y_batch.float(), ypred_batch.float())
            losses.append(loss)

            # Compute R² on GPU
            ss_res = torch.sum((y_batch - ypred_batch) ** 2)
            ss_tot = torch.sum((y_batch - torch.mean(y_batch)) ** 2)
            r2_score = 1 - ss_res / ss_tot
            

            r2_scores.append(r2_score)



            # Handle both wrapped and unwrapped models
            base_model = model.module if hasattr(model, 'module') else model

            # threshold = base_model.covalent_neighbor_threshold.t.cpu().data.item()



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

        if args.checkpoint:
            loss_mean = torch.mean(torch.stack(losses)).item()
            r2_mean =torch.mean(torch.stack(r2_scores)).item()

            train_metrics = {
                "loss": loss_mean,
                "r2": r2_mean
            }

            tqdm.write(
                "[{}/{}] Training: \tloss:{:0.4f}\n R2: {}"
                .format(
                    epoch + 1,
                    args.epochs,
                    loss_mean,
                    r2_mean,
                )
            )


            checkpoint_dict = checkpoint_model(
                model,
                optimizer,
                val_dataloader,
                epoch,
                step,
                args.checkpoint_dir + "/model-epoch-{}.pth".format(epoch),
            )
            if checkpoint_dict["validate_dict"]["r2"] > best_checkpoint_r2:
                best_checkpoint_step = step
                best_checkpoint_epoch = epoch
                best_checkpoint_r2 = checkpoint_dict["validate_dict"]["r2"]
                best_checkpoint_dict = checkpoint_dict

    if args.checkpoint:
        # Save the best checkpoint model
        torch.save(best_checkpoint_dict, args.checkpoint_dir + "/best_checkpoint.pth")
    
    print(
        "best training checkpoint epoch {}/step {} with r2: {}".format(
            best_checkpoint_epoch, best_checkpoint_step, best_checkpoint_r2
        )
    )


def validate(model, val_dataloader, epoch_ind):

    model.eval()

    loss_fn = nn.MSELoss().float()


    # Assuming `device` is your CUDA device
    losses = []
    r2_scores = []

    for batch in tqdm(val_dataloader):
        data = [x[2] for x in batch if x is not None]

        device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'

        batch_data = Batch.from_data_list(data).to(device)
        with torch.no_grad():
            ypred_batch, *_ = model(batch_data)

        y_batch_cpu = torch.cat([x[2].y for x in batch])
        y_batch = y_batch_cpu.to(device)

        loss = loss_fn(ypred_batch.float(), y_batch.float())
        losses.append(loss)

        ss_res = torch.sum((y_batch - ypred_batch) ** 2)
        ss_tot = torch.sum((y_batch - torch.mean(y_batch)) ** 2)
        r2_score = 1 - ss_res / ss_tot
        

        r2_scores.append(r2_score)




    loss_mean = torch.mean(torch.stack(losses)).item()
    r2_mean =torch.mean(torch.stack(r2_scores)).item()

    val_metrics = {
        "loss": loss_mean,
        "r2": r2_mean
    }
            
    tqdm.write(
        "[{}/{}] Validation: \tloss:{:0.4f}\n R2: {}"
        .format(
            epoch_ind+1, args.epochs,
            loss_mean,
            r2_mean,
        )
    )

    model.train()
    
    return val_metrics


def checkpoint_model(model, optimizer,  dataloader, epoch, step, checkpoint_path):
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path))
    validate_dict = validate(model, dataloader, epoch)
    model.train() # set model back to train mode

    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "step": step,
        "epoch": epoch,
        "validate_dict": validate_dict,
    }

    torch.save(checkpoint_dict, checkpoint_path)

    # return the computed metrics so it can be used to update the training loop
    return checkpoint_dict


def main():
    train()


if __name__ == "__main__":
    main()
