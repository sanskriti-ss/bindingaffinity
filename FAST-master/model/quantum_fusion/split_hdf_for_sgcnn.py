"""Split refined_all.hdf into train / val HDF files for SGCNN training.

Usage:
    python split_hdf_for_sgcnn.py \
        --input  FAST-master/model/quantum_fusion/refined_all.hdf \
        --outdir FAST-master/model/quantum_fusion \
        --val-frac 0.15 \
        --seed 42
"""
import argparse
import os
import random
import h5py
import numpy as np


def copy_group(src_grp, dst_file, pdbid: str) -> None:
    """Copy a single pdbid group (with all subgroups/datasets/attrs) to dst_file."""
    src_file = src_grp.file
    src_file.copy(src_grp, dst_file, name=pdbid)


def split(input_path: str, outdir: str, val_frac: float, seed: int) -> None:
    os.makedirs(outdir, exist_ok=True)

    with h5py.File(input_path, "r") as src:
        pdbids = list(src.keys())
        print(f"Total complexes: {len(pdbids)}")

        rng = random.Random(seed)
        shuffled = pdbids[:]
        rng.shuffle(shuffled)

        n_val   = max(1, int(len(shuffled) * val_frac))
        n_train = len(shuffled) - n_val
        train_ids = shuffled[:n_train]
        val_ids   = shuffled[n_train:]
        print(f"Train: {len(train_ids)}  Val: {len(val_ids)}")

        train_path = os.path.join(outdir, "sgcnn_train.hdf")
        val_path   = os.path.join(outdir, "sgcnn_val.hdf")

        print(f"Writing {train_path} ...")
        with h5py.File(train_path, "w") as dst:
            for pid in train_ids:
                copy_group(src[pid], dst, pid)

        print(f"Writing {val_path} ...")
        with h5py.File(val_path, "w") as dst:
            for pid in val_ids:
                copy_group(src[pid], dst, pid)

    print("Done.")
    print(f"  Train → {train_path}")
    print(f"  Val   → {val_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",    default="FAST-master/model/quantum_fusion/refined_all.hdf")
    p.add_argument("--outdir",   default="FAST-master/model/quantum_fusion")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed",     type=int,   default=42)
    args = p.parse_args()
    split(args.input, args.outdir, args.val_frac, args.seed)
