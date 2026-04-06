import h5py
import random
import os

# Set random seed for reproducibility
random.seed(42)

_FAST_DATA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../FAST-master/data"))


def split_hdf(source_file, out_train, out_val, out_test, train_frac=0.8, val_frac=0.1):
    with h5py.File(source_file, 'r') as src:
        all_ids = list(src.keys())

    random.shuffle(all_ids)
    n = len(all_ids)
    train_ids = all_ids[:int(n * train_frac)]
    val_ids   = all_ids[int(n * train_frac):int(n * (train_frac + val_frac))]
    test_ids  = all_ids[int(n * (train_frac + val_frac)):]

    print(f"{os.path.basename(source_file)}: total={n}  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    def copy_proteins(protein_list, dst_file):
        with h5py.File(source_file, 'r') as src, h5py.File(dst_file, 'w') as dst:
            for pid in protein_list:
                src.copy(pid, dst)

    copy_proteins(train_ids, out_train)
    copy_proteins(val_ids,   out_val)
    copy_proteins(test_ids,  out_test)
    print(f"  Written: {out_train}")
    print(f"  Written: {out_val}")
    print(f"  Written: {out_test}")


# 3D CNN: pre-voxelized format from step5a.hdf
split_hdf(
    source_file = os.path.join(_FAST_DATA, "step5a.hdf"),
    out_train   = os.path.join(_FAST_DATA, "3dcnn_train.hdf"),
    out_val     = os.path.join(_FAST_DATA, "3dcnn_val.hdf"),
    out_test    = os.path.join(_FAST_DATA, "3dcnn_test.hdf"),
)

# SGCNN: atom-level pafnucy format from pafuncy_out/refined.hdf
split_hdf(
    source_file = os.path.join(_FAST_DATA, "pafuncy_out/refined.hdf"),
    out_train   = os.path.join(_FAST_DATA, "sgcnn_train.hdf"),
    out_val     = os.path.join(_FAST_DATA, "sgcnn_val.hdf"),
    out_test    = os.path.join(_FAST_DATA, "sgcnn_test.hdf"),
)
