################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Spatial Graph Convolutional Network evaluation script
################################################################################

import os
import os.path as osp
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import h5py
import math
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from glob import glob
from torch_geometric.data import DataLoader as GeometricDataLoader
from torch_geometric.nn import DataParallel as GeometricDataParallel
from torch_geometric.data import Data, Batch, DataListLoader
from data_utils import PDBBindDataset
from model import PotentialNetParallel
import argparse


def calculate_r2(y_true, y_pred):
    """Calculate R² (coefficient of determination)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def collate_fn_none_filter(batch):
    return [x for x in batch if x is not None]


def worker_init_fn(worker_id):
    np.random.seed(int(0))


def test(args):
    # Load model checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available() and not args.force_cpu:
        model_train_dict = torch.load(args.checkpoint)
    else:
        model_train_dict = torch.load(args.checkpoint, map_location=torch.device('cpu'), weights_only=False)

    # Get feature size from training arguments
    if model_train_dict["args"]["feature_type"] == "pybel":
        feature_size = 22 + 1  # 22 features from pybel + 1 for the van der waals
    else:
        feature_size = 75

    # Create model with same architecture as training
    if torch.cuda.is_available() and not args.force_cpu:
        model = GeometricDataParallel(
            PotentialNetParallel(
                in_channels=feature_size,
                out_channels=1,
                covalent_gather_width=model_train_dict["args"]["covalent_gather_width"],
                non_covalent_gather_width=model_train_dict["args"]["non_covalent_gather_width"],
                covalent_k=model_train_dict["args"]["covalent_k"],
                non_covalent_k=model_train_dict["args"]["non_covalent_k"],
                covalent_neighbor_threshold=model_train_dict["args"]["covalent_threshold"],
                non_covalent_neighbor_threshold=model_train_dict["args"]["non_covalent_threshold"],
            )
        ).float()
    else:
        model = PotentialNetParallel(
            in_channels=feature_size,
            out_channels=1,
            covalent_gather_width=model_train_dict["args"]["covalent_gather_width"],
            non_covalent_gather_width=model_train_dict["args"]["non_covalent_gather_width"],
            covalent_k=model_train_dict["args"]["covalent_k"],
            non_covalent_k=model_train_dict["args"]["non_covalent_k"],
            covalent_neighbor_threshold=model_train_dict["args"]["covalent_threshold"],
            non_covalent_neighbor_threshold=model_train_dict["args"]["non_covalent_threshold"],
        ).float()

    model.load_state_dict(model_train_dict["model_state_dict"])
    model.to(device)

    if args.print_model:
        print(model)
    print("{} total parameters.".format(sum(p.numel() for p in model.parameters())))

    # Load dataset
    dataset_list = []
    for data in args.test_data:
        dataset_list.append(
            PDBBindDataset(
                data_file=data,
                dataset_name=args.dataset_name,
                feature_type=model_train_dict["args"]["feature_type"],
                preprocessing_type=model_train_dict["args"]["preprocessing_type"],
                output_info=True,
                cache_data=False,
                use_docking=model_train_dict["args"]["use_docking"],
            )
        )

    dataset = ConcatDataset(dataset_list)
    print("{} complexes in dataset".format(len(dataset)))

    dataloader = DataListLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        worker_init_fn=worker_init_fn
    )

    model.eval()

    # Arrays to store predictions and targets (matching 3DCNN structure)
    ytrue_arr = []
    ypred_arr = []
    pred_list = []
    hidden_features_list = []
    complex_ids = []  # Store complex IDs (matching 3DCNN naming)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            batch = [x for x in batch if x is not None]
            if len(batch) < 1:
                print("empty batch, skipping to next batch")
                continue

            # Create a proper batch from the list of data objects
            data_list = [x[2] for x in batch]
            batched_data = Batch.from_data_list(data_list)
            
            # Move to device
            batched_data = batched_data.to(device)
            
            # Get predictions and hidden features
            if hasattr(model, 'module'):
                # For DataParallel models
                (
                    covalent_feature,
                    non_covalent_feature,
                    pool_feature,
                    fc0_feature,
                    fc1_feature,
                    y_pred,
                ) = model.module(batched_data, return_hidden_feature=True)
            else:
                # For regular models
                (
                    covalent_feature,
                    non_covalent_feature,
                    pool_feature,
                    fc0_feature,
                    fc1_feature,
                    y_pred,
                ) = model(batched_data, return_hidden_feature=True)

            y_true = torch.cat([x[2].y for x in batch])

            # Move to CPU for numpy operations
            y_true_np = y_true.cpu().data.numpy().flatten()
            y_pred_np = y_pred.cpu().data.numpy().flatten()

            # Collect all data
            ytrue_arr.extend(y_true_np)
            ypred_arr.extend(y_pred_np)

            # Extract complex IDs from batch (create consistent format with 3DCNN)
            batch_complex_ids = [x[0] for x in batch]  # pdbid_pose format
            complex_ids.extend(batch_complex_ids)

            # Store predictions for CSV output (matching 3DCNN structure)
            if args.save_pred:
                for i, (complex_id, y_t, y_p) in enumerate(zip(batch_complex_ids, y_true_np, y_pred_np)):
                    # Using same structure as 3DCNN: [cid, complex_id, label, pred]
                    cid = len(pred_list)  # Sequential ID like 3DCNN
                    pred_list.append([cid, complex_id, y_t, y_p])

            # Store hidden features
            if args.save_feat:
                hidden_features = np.concatenate(
                    (
                        covalent_feature.cpu().data.numpy(),
                        non_covalent_feature.cpu().data.numpy(),
                        pool_feature.cpu().data.numpy(),
                        fc0_feature.cpu().data.numpy(),
                        fc1_feature.cpu().data.numpy(),
                    ),
                    axis=1,
                )
                hidden_features_list.append(hidden_features)

            print("[%d/%d] evaluating" % (batch_idx + 1, len(dataloader)))

    # Convert to numpy arrays
    ytrue_arr = np.array(ytrue_arr)
    ypred_arr = np.array(ypred_arr)

    # Calculate metrics (matching 3DCNN output)
    rmse = math.sqrt(mean_squared_error(ytrue_arr, ypred_arr))
    mae = mean_absolute_error(ytrue_arr, ypred_arr)
    r2 = r2_score(ytrue_arr, ypred_arr)
    r2_custom = calculate_r2(ytrue_arr, ypred_arr)
    pearson, ppval = pearsonr(ytrue_arr, ypred_arr)
    spearman, spval = spearmanr(ytrue_arr, ypred_arr)
    mean_pred = np.mean(ypred_arr)
    std_pred = np.std(ypred_arr)

    print("Evaluation Summary:")
    print("RMSE: %.3f, MAE: %.3f, R^2 score: %.3f, R^2 (custom): %.3f, Pearson: %.3f, Spearman: %.3f, mean/std: %.3f/%.3f" % 
          (rmse, mae, r2, r2_custom, pearson, spearman, mean_pred, std_pred))

    # Create output directory
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Save predictions as CSV (matching 3DCNN structure)
    if args.save_pred:
        csv_fpath = os.path.join(args.output, f"{args.output_file_name}_pred.csv")
        df = pd.DataFrame(pred_list, columns=["cid", "complex_id", "label", "pred"])
        df.to_csv(csv_fpath, index=False)
        print(f"Predictions saved to: {csv_fpath}")

    # Save hidden features and complex IDs as numpy arrays (matching 3DCNN structure)
    if args.save_feat:
        if hidden_features_list:
            all_hidden_features = np.concatenate(hidden_features_list, axis=0)
            feat_base_path = os.path.join(args.output, args.output_file_name)
            
            # Save features and complex IDs (matching 3DCNN naming)
            np.save(feat_base_path + "_feat.npy", all_hidden_features)
            np.save(feat_base_path + "_complex_ids.npy", np.array(complex_ids))
            
            print(f"Features saved to: {feat_base_path}_feat.npy")
            print(f"Complex IDs saved to: {feat_base_path}_complex_ids.npy")

    # Also save detailed HDF5 output (original functionality)
    if args.save_hdf5:
        output_f = os.path.join(args.output, f"{args.output_file_name}.hdf")
        
        with h5py.File(output_f, "w") as f:
            batch_idx = 0
            for batch in tqdm(dataloader):
                batch = [x for x in batch if x is not None]
                if len(batch) < 1:
                    continue

                for item in batch:
                    name = item[0]
                    pose = item[1]
                    data = item[2]

                    name_grp = f.require_group(str(name))
                    name_pose_grp = name_grp.require_group(str(pose))

                    y = data.y
                    name_pose_grp.attrs["y_true"] = y

                    # Get predictions and features
                    data_batch = Batch().from_data_list([data]).to(device)
                    
                    if hasattr(model, 'module'):
                        (
                            covalent_feature,
                            non_covalent_feature,
                            pool_feature,
                            fc0_feature,
                            fc1_feature,
                            y_,
                        ) = model.module(data_batch, return_hidden_feature=True)
                    else:
                        (
                            covalent_feature,
                            non_covalent_feature,
                            pool_feature,
                            fc0_feature,
                            fc1_feature,
                            y_,
                        ) = model(data_batch, return_hidden_feature=True)

                    name_pose_grp.attrs["y_pred"] = y_.cpu().data.numpy()
                    hidden_features = np.concatenate(
                        (
                            covalent_feature.cpu().data.numpy(),
                            non_covalent_feature.cpu().data.numpy(),
                            pool_feature.cpu().data.numpy(),
                            fc0_feature.cpu().data.numpy(),
                            fc1_feature.cpu().data.numpy(),
                        ),
                        axis=1,
                    )

                    name_pose_grp.create_dataset(
                        "hidden_features",
                        (hidden_features.shape[0], hidden_features.shape[1]),
                        data=hidden_features,
                    )
                batch_idx += 1
        print(f"HDF5 output saved to: {output_f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="path to model checkpoint", required=True)
    parser.add_argument(
        "--preprocessing-type",
        type=str,
        choices=["raw", "processed"],
        help="indicate raw pdb or (chimera) processed",
        default="processed",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        choices=["pybel", "rdkit"],
        help="indicate pybel (openbabel) or rdkit features",
        default="pybel",
    )
    parser.add_argument("--dataset-name", type=str, default="pdbbind")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="batch size to use for dataloader"
    )
    parser.add_argument(
        "--num-workers", default=0, type=int, help="number of workers for dataloader"
    )
    parser.add_argument("--test-data", nargs="+", required=True)
    parser.add_argument("--output", help="path to output directory", required=True)
    parser.add_argument(
        "--use-docking",
        help="flag to indicate if dataset contains docking info",
        default=False,
        action="store_true",
    )
    parser.add_argument("--output-file-name", help="output file name", required=True)
    parser.add_argument("--print-model", action="store_true", help="bool flag to determine whether to print the model")
    parser.add_argument("--save-pred", action="store_true", default=True, help="whether to save prediction results in csv")
    parser.add_argument("--save-feat", action="store_true", default=True, help="whether to save hidden features in npy")
    parser.add_argument("--save-hdf5", action="store_true", default=False, help="whether to save detailed HDF5 output")
    parser.add_argument("--force-cpu", action="store_true", help="force CPU usage even if CUDA is available")

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    main()