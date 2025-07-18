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
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint", type=bool, default=False, help="boolean flag for checkpoints"
)
parser.add_argument(
    "--checkpoint-dir", default=os.getcwd(), help="path to store model checkpoints"
)
parser.add_argument(
    "--checkpoint-iter", default=10, type=int, help="number of epochs per checkpoint"
)
parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
parser.add_argument(
    "--num-workers", default=24, type=int, help="number of workers for dataloader"
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
    help="idicate raw pdb or (chimera) processed",
    required=True,
)
parser.add_argument(
    "--feature-type",
    type=str,
    choices=["pybel", "rdkit"],
    help="indicate pybel (openbabel) or rdkit features",
    required=True,
)
parser.add_argument(
    "--dataset-name", type=str, required=True
)
parser.add_argument("--covalent-gather-width", type=int, default=128)
parser.add_argument("--non-covalent-gather-width", type=int, default=128)
parser.add_argument("--covalent-k", type=int, default=1)
parser.add_argument("--non-covalent-k", type=int, default=1)
parser.add_argument("--covalent-threshold", type=float, default=1.5)
parser.add_argument("--non-covalent-threshold", type=float, default=7.5)
parser.add_argument("--train-data", type=str, required=True, nargs="+")
parser.add_argument("--val-data", type=str, required=True, nargs="+")
parser.add_argument("--use-docking", default=False, action="store_true")

# Add wandb arguments
parser.add_argument("--wandb-project", default="binding-affinity", help="wandb project name")
parser.add_argument("--wandb-run-name", default="", help="wandb run name (auto-generated if empty)")
parser.add_argument("--disable-wandb", action="store_true", help="disable wandb logging")

args = parser.parse_args()

# seed all random number generators and set cudnn settings for deterministic
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = "0"

def worker_init_fn(worker_id):
    np.random.seed(int(0))

def collate_fn_none_filter(batch):
    return [x for x in batch if x is not None]

# Get wandb API key
wandb_api_key = os.environ.get("WANDB_API_KEY")

def train():
    # Initialize wandb
    if not args.disable_wandb:
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        
        # Generate run name if not provided
        run_name = args.wandb_run_name
        if not run_name:
            run_name = f"sgcnn-{args.feature_type}-lr{args.lr}-bs{args.batch_size}-{args.dataset_name}"
        
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": "SGCNN",
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "feature_type": args.feature_type,
                "preprocessing_type": args.preprocessing_type,
                "dataset_name": args.dataset_name,
                "covalent_gather_width": args.covalent_gather_width,
                "non_covalent_gather_width": args.non_covalent_gather_width,
                "covalent_k": args.covalent_k,
                "non_covalent_k": args.non_covalent_k,
                "covalent_threshold": args.covalent_threshold,
                "non_covalent_threshold": args.non_covalent_threshold,
                "use_docking": args.use_docking,
                "num_workers": args.num_workers,
                "checkpoint_iter": args.checkpoint_iter
            }
        )

    # set the input channel dims based on featurization type
    if args.feature_type == "pybel":
        feature_size = 22 + 1 # 22 features from pybel + 1 for the van der waals
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
        drop_last=True,
    )

    val_dataloader = DataListLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    tqdm.write("{} complexes in training dataset".format(len(train_dataset)))
    tqdm.write("{} complexes in validation dataset".format(len(val_dataset)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        model = GeometricDataParallel(
            PotentialNetParallel(
                in_channels=feature_size,
                out_channels=1,
                covalent_gather_width=args.covalent_gather_width,
                non_covalent_gather_width=args.non_covalent_gather_width,
                covalent_k=args.covalent_k,
                non_covalent_k=args.non_covalent_k,
                covalent_neighbor_threshold=args.covalent_threshold,
                non_covalent_neighbor_threshold=args.non_covalent_threshold,
            )
        ).float()
    else:
        model = PotentialNetParallel(
            in_channels=feature_size,
            out_channels=1,
            covalent_gather_width=args.covalent_gather_width,
            non_covalent_gather_width=args.non_covalent_gather_width,
            covalent_k=args.covalent_k,
            non_covalent_k=args.non_covalent_k,
            covalent_neighbor_threshold=args.covalent_threshold,
            non_covalent_neighbor_threshold=args.non_covalent_threshold,
        ).float()

    model.train()
    model.to(device)
    tqdm.write(str(model))
    tqdm.write(
        "{} trainable parameters.".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    tqdm.write(
        "{} total parameters.".format(sum(p.numel() for p in model.parameters()))
    )

    # Log model info to wandb
    if not args.disable_wandb:
        wandb.log({
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "train_dataset_size": len(train_dataset),
            "val_dataset_size": len(val_dataset)
        })

    criterion = nn.MSELoss().float()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_checkpoint_dict = None
    best_checkpoint_epoch = 0
    best_checkpoint_step = 0
    best_checkpoint_r2 = -9e9
    step = 0
    
    for epoch in range(args.epochs):
        losses = []
        epoch_y_true = []
        epoch_y_pred = []
        
        for batch in tqdm(train_dataloader):
            batch = [x for x in batch if x is not None]
            if len(batch) < 1:
                print("empty batch, skipping to next batch")
                continue
            optimizer.zero_grad()

            # Create a proper batch from the list of data objects
            data_list = [x[2] for x in batch]
            batched_data = Batch.from_data_list(data_list)
            
            # Move to device if using GPU
            if torch.cuda.is_available():
                batched_data = batched_data.to(device)
            
            y_ = model(batched_data)
            y = torch.cat([x[2].y for x in batch])

            # Move y to device if using GPU
            if torch.cuda.is_available():
                y = y.to(device)

            loss = criterion(y.float(), y_.float())
            losses.append(loss.cpu().data.item())
            loss.backward()

            y_true = y.cpu().data.numpy()
            y_pred = y_.cpu().data.numpy()

            # Collect for epoch-level metrics
            epoch_y_true.extend(y_true.flatten())
            epoch_y_pred.extend(y_pred.flatten())

            r2 = r2_score(y_true=y_true, y_pred=y_pred)
            mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)

            pearsonr = stats.pearsonr(y_true.reshape(-1), y_pred.reshape(-1))
            spearmanr = stats.spearmanr(y_true.reshape(-1), y_pred.reshape(-1))

            # Fix threshold access for CPU/GPU compatibility
            if torch.cuda.is_available():
                cov_thresh = model.module.covalent_neighbor_threshold.t.cpu().data.item()
                non_cov_thresh = model.module.non_covalent_neighbor_threshold.t.cpu().data.item()
            else:
                cov_thresh = model.covalent_neighbor_threshold.t.cpu().data.item()
                non_cov_thresh = model.non_covalent_neighbor_threshold.t.cpu().data.item()

            # Log batch metrics to wandb
            if not args.disable_wandb:
                wandb.log({
                    "batch_loss": loss.cpu().data.item(),
                    "batch_r2": r2,
                    "batch_mae": mae,
                    "batch_pearsonr": float(pearsonr[0]),
                    "batch_spearmanr": float(spearmanr[0]),
                    "covalent_threshold": cov_thresh,
                    "non_covalent_threshold": non_cov_thresh,
                    "step": step,
                    "epoch": epoch
                })

            tqdm.write(
                "epoch: {}\tloss:{:0.4f}\tr2: {:0.4f}\t pearsonr: {:0.4f}\tspearmanr: {:0.4f}\tmae: {:0.4f}\tpred stdev: {:0.4f}"
                "\t pred mean: {:0.4f} \tcovalent_threshold: {:0.4f} \tnon covalent threshold: {:0.4f}".format(
                    epoch,
                    loss.cpu().data.numpy(),
                    r2,
                    float(pearsonr[0]),
                    float(spearmanr[0]),
                    float(mae),
                    np.std(y_pred),
                    np.mean(y_pred),
                    cov_thresh,
                    non_cov_thresh,
                )
            )

            if args.checkpoint:
                if step % args.checkpoint_iter == 0:
                    checkpoint_dict = checkpoint_model(
                        model,
                        val_dataloader,
                        epoch,
                        step,
                        args.checkpoint_dir
                        + "/model-epoch-{}-step-{}.pth".format(epoch, step),
                    )
                    if checkpoint_dict["validate_dict"]["r2"] > best_checkpoint_r2:
                        best_checkpoint_step = step
                        best_checkpoint_epoch = epoch
                        best_checkpoint_r2 = checkpoint_dict["validate_dict"]["r2"]
                        best_checkpoint_dict = checkpoint_dict
                        
                        # Log best model update to wandb
                        if not args.disable_wandb:
                            wandb.log({
                                "best_val_r2": best_checkpoint_r2,
                                "best_model_epoch": epoch,
                                "best_model_step": step
                            })

            optimizer.step()
            step += 1

        # Calculate epoch-level training metrics
        epoch_y_true = np.array(epoch_y_true)
        epoch_y_pred = np.array(epoch_y_pred)
        train_epoch_loss = np.mean(losses)
        train_r2 = r2_score(epoch_y_true, epoch_y_pred)
        train_mae = mean_absolute_error(epoch_y_true, epoch_y_pred)
        train_rmse = np.sqrt(mean_squared_error(epoch_y_true, epoch_y_pred))
        train_pearsonr = stats.pearsonr(epoch_y_true, epoch_y_pred)
        train_spearmanr = stats.spearmanr(epoch_y_true, epoch_y_pred)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_epoch_loss": train_epoch_loss,
            "train_epoch_r2": train_r2,
            "train_epoch_mae": train_mae,
            "train_epoch_rmse": train_rmse,
            "train_epoch_pearsonr": float(train_pearsonr[0]),
            "train_epoch_spearmanr": float(train_spearmanr[0])
        }

        print(f"[{epoch+1}/{args.epochs}] training epoch - loss: {train_epoch_loss:.4f}, R²: {train_r2:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")

        # End of epoch checkpoint and validation
        if args.checkpoint:
            checkpoint_dict = checkpoint_model(
                model,
                val_dataloader,
                epoch,
                step,
                args.checkpoint_dir + "/model-epoch-{}-step-{}.pth".format(epoch, step),
            )
            
            val_metrics = checkpoint_dict["validate_dict"]
            epoch_metrics.update({
                "val_epoch_loss": val_metrics["mse"],  # MSE is the loss
                "val_epoch_r2": val_metrics["r2"],
                "val_epoch_mae": val_metrics["mae"],
                "val_epoch_rmse": np.sqrt(val_metrics["mse"]),
                "val_epoch_pearsonr": float(val_metrics["pearsonr"][0]),
                "val_epoch_spearmanr": float(val_metrics["spearmanr"][0])
            })
            
            if checkpoint_dict["validate_dict"]["r2"] > best_checkpoint_r2:
                best_checkpoint_step = step
                best_checkpoint_epoch = epoch
                best_checkpoint_r2 = checkpoint_dict["validate_dict"]["r2"]
                best_checkpoint_dict = checkpoint_dict
                
                # Log best model update to wandb
                if not args.disable_wandb:
                    wandb.log({
                        "best_val_r2": best_checkpoint_r2,
                        "best_model_epoch": epoch,
                        "best_model_step": step
                    })

        # Log epoch metrics to wandb
        if not args.disable_wandb:
            wandb.log(epoch_metrics)

    if args.checkpoint:
        # Save final model
        checkpoint_dict = checkpoint_model(
            model,
            val_dataloader,
            epoch,
            step,
            args.checkpoint_dir + "/model-epoch-{}-step-{}.pth".format(epoch, step),
        )

        if checkpoint_dict["validate_dict"]["r2"] > best_checkpoint_r2:
            best_checkpoint_step = step
            best_checkpoint_epoch = epoch
            best_checkpoint_r2 = checkpoint_dict["validate_dict"]["r2"]
            best_checkpoint_dict = checkpoint_dict

        # Save best checkpoint
        torch.save(best_checkpoint_dict, args.checkpoint_dir + "/best_checkpoint.pth")
        
        # Log final summary to wandb
        if not args.disable_wandb:
            wandb.log({
                "final_best_val_r2": best_checkpoint_r2,
                "final_best_epoch": best_checkpoint_epoch,
                "final_best_step": best_checkpoint_step,
                "training_completed": True
            })

    print(
        "best training checkpoint epoch {}/step {} with r2: {}".format(
            best_checkpoint_epoch, best_checkpoint_step, best_checkpoint_r2
        )
    )
    
    # Finish wandb run
    if not args.disable_wandb:
        wandb.finish()

def validate(model, val_dataloader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_true = []
    y_pred = []
    pdbid_list = []
    pose_list = []

    for batch in tqdm(val_dataloader):
        batch = [x for x in batch if x is not None]
        if len(batch) < 1:
            continue
            
        # Batch the data properly
        data_list = [x[2] for x in batch]
        batched_data = Batch.from_data_list(data_list)
        
        if torch.cuda.is_available():
            batched_data = batched_data.to(device)
            
        y_ = model(batched_data)
        y = torch.cat([x[2].y for x in batch])

        pdbid_list.extend([x[0] for x in batch])
        pose_list.extend([x[1] for x in batch])
        y_true.append(y.cpu().data.numpy())
        y_pred.append(y_.cpu().data.numpy())

    y_true = np.concatenate(y_true).reshape(-1, 1)
    y_pred = np.concatenate(y_pred).reshape(-1, 1)

    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    pearsonr = stats.pearsonr(y_true.reshape(-1), y_pred.reshape(-1))
    spearmanr = stats.spearmanr(y_true.reshape(-1), y_pred.reshape(-1))

    tqdm.write(
        str(
            "r2: {}\tmae: {}\tmse: {}\tpearsonr: {}\t spearmanr: {}".format(
                r2, mae, mse, pearsonr, spearmanr
            )
        )
    )
    model.train()
    return {
        "r2": r2,
        "mse": mse,
        "mae": mae,
        "pearsonr": pearsonr,
        "spearmanr": spearmanr,
        "y_true": y_true,
        "y_pred": y_pred,
        "pdbid": pdbid_list,
        "pose": pose_list,
    }

def checkpoint_model(model, dataloader, epoch, step, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    validate_dict = validate(model, dataloader)
    model.train()

    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "step": step,
        "epoch": epoch,
        "validate_dict": validate_dict,
    }

    torch.save(checkpoint_dict, output_path)
    return checkpoint_dict

def main():
    train()

if __name__ == "__main__":
    main()