#!/usr/bin/env python
"""
Extract 54-dim hidden features from the trained SGCNN (PotentialNet) model.

Run AFTER generate_mlhdf_rdkit.py has created the ML-HDF file.

Steps performed:
  1. Load ML-HDF
  2. Build graph per complex (same logic as PDBBindDataset in data_utils.py)
  3. Run forward pass through PotentialNetParallel with return_hidden_feature=True
  4. Concatenate [covalent(16), non_covalent(12), pool(12), fc0(8), fc1(6)] → 54-dim
  5. Save as NPZ: {pdbid: array(54,)}

The SGCNN node features are: [vdw_radius(1), atom_feats(19)] = 20 dims.

NOTE: A trained SGCNN checkpoint is required. If you do not have one, run
      the SGCNN train.py first. An example checkpoint path:
        C:/bindingaffinity/FAST-master/model/sgcnn/src/sgcnn/checkpoint/best_model.pth

Usage:
    python extract_sgcnn_features.py \
        --hdf    C:/bindingaffinity/FAST-master/model/quantum_fusion/refined_all.hdf \
        --ckpt   <path/to/sgcnn_checkpoint.pth> \
        --output C:/bindingaffinity/FAST-master/model/quantum_fusion/refined_sgcnn_features.npz
"""

import os
import sys
import argparse
import functools

# ── make sgcnn code importable ───────────────────────────────────────────────
_SGCNN_DIR = os.path.join(os.path.dirname(__file__), "..", "sgcnn", "src", "sgcnn")
sys.path.insert(0, os.path.abspath(_SGCNN_DIR))

import h5py
import numpy as np
import torch
import scipy
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

# safe globals for torch.load on newer PyTorch
torch.serialization.add_safe_globals([
    functools.partial, getattr,
    np._core.multiarray.scalar, np.dtype,
])

from model import PotentialNetParallel


def _load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Load PotentialNetParallel from checkpoint (wraps in module like test.py)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    net = PotentialNetParallel(
        in_channels=20,
        out_channels=1,
        covalent_gather_width=args["covalent_gather_width"],
        non_covalent_gather_width=args["non_covalent_gather_width"],
        covalent_k=args["covalent_k"],
        non_covalent_k=args["non_covalent_k"],
        covalent_neighbor_threshold=args["covalent_threshold"],
        non_covalent_neighbor_threshold=args["non_covalent_threshold"],
    ).float()

    # test.py wraps in a plain nn.Module to add 'module.' prefix
    wrapper = torch.nn.Module()
    wrapper.add_module("module", net)

    from collections import OrderedDict
    new_sd = OrderedDict()
    for k, v in ckpt["model_state_dict"].items():
        new_sd["module." + k] = v
    wrapper.load_state_dict(new_sd)
    wrapper.to(device)
    wrapper.eval()
    return wrapper


def _hdf_to_graph(pdbid: str, hdf: h5py.File) -> Data:
    """Build a PyG Data object from one HDF complex (same as data_utils.py)."""
    sub = hdf[pdbid]["pybel"]["processed"]["pdbbind"]
    raw = np.array(sub["data"], dtype=np.float32)         # (N, 22)
    vdw = sub.attrs["van_der_waals"].reshape(-1, 1)        # (N, 1)
    affinity = float(hdf[pdbid].attrs["affinity"])

    coords     = raw[:, :3]                                 # (N, 3)
    atom_feats = raw[:, 3:22]                               # (N, 19)
    node_feats = np.concatenate([vdw, atom_feats], axis=1) # (N, 20)

    dists = pairwise_distances(coords, metric="euclidean")
    edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dists).float())

    return Data(
        x=torch.from_numpy(node_feats).float(),
        edge_index=edge_index,
        edge_attr=edge_attr.view(-1, 1),
        y=torch.FloatTensor([[affinity]]),
    )


def extract(hdf_path: str, ckpt_path: str, output_path: str):
    if not os.path.exists(ckpt_path):
        print(f"ERROR: SGCNN checkpoint not found: {ckpt_path}")
        print("       Train the SGCNN first (see FAST-master/model/sgcnn/src/sgcnn/train.py)")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = _load_model(ckpt_path, device)
    print(f"Using device: {device}")
    print(f"Loaded SGCNN checkpoint: {ckpt_path}")
    print(f"  {sum(p.numel() for p in model.parameters())} parameters")

    feat_dict: dict = {}
    with h5py.File(hdf_path, "r") as hf:
        pdbids = list(hf.keys())
        print(f"Processing {len(pdbids)} complexes")

        for pdbid in tqdm(pdbids, desc="SGCNN features"):
            try:
                data = _hdf_to_graph(pdbid, hf)
            except Exception as e:
                print(f"  skip {pdbid}: {e}")
                continue

            with torch.no_grad():
                batch = Batch().from_data_list([data]).to(device)
                (
                    _y,
                    cov_feat,
                    noncov_feat,
                    pool_feat,
                    fc0_feat,
                    fc1_feat,
                ) = model.module(
                    batch,
                    return_hidden_feature=True,
                )

            hidden = np.concatenate([
                cov_feat.cpu().numpy(),
                noncov_feat.cpu().numpy(),
                pool_feat.cpu().numpy(),
                fc0_feat.cpu().numpy(),
                fc1_feat.cpu().numpy(),
            ], axis=1)                      # (1, 54)

            feat_dict[pdbid] = hidden[0]    # (54,)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez(output_path, **feat_dict)
    print(f"\nSaved {len(feat_dict)} feature vectors → {output_path}")
    if feat_dict:
        sample = next(iter(feat_dict.values()))
        print(f"  Feature shape: {sample.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract SGCNN hidden features (54-dim) from ML-HDF"
    )
    parser.add_argument(
        "--hdf",
        default="C:/bindingaffinity/FAST-master/model/quantum_fusion/refined_all.hdf",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="path to trained SGCNN .pth checkpoint",
    )
    parser.add_argument(
        "--output",
        default="C:/bindingaffinity/FAST-master/model/quantum_fusion/refined_sgcnn_features.npz",
    )
    args = parser.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    extract(
        hdf_path=args.hdf,
        ckpt_path=args.ckpt,
        output_path=args.output,
    )
