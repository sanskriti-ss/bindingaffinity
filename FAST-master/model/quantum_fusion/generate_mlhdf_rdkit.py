#!/usr/bin/env python
"""
Generate ML-HDF from PDBBind refined-set using RDKit.

Replicates the pybel/tfbio 19-feature atom featurization so the output is
compatible with the existing 3DCNN (main_eval.py) and SGCNN (test.py) eval
scripts.

Output HDF structure (per complex):
    {pdbid}/
        .attrs['affinity'] = float (-logKd/Ki)
        pybel/processed/pdbbind/
            data          : (N_atoms, 22) float32  [xyz(3) + 19 features]
            .attrs['van_der_waals'] : (N_atoms,) float32

19 features match tfbio Featurizer defaults:
    [0-8]  : atom-type one-hot  B,C,N,O,P,S,Se,halogen,metal
    [9]    : hybridization int  (SP=1, SP2=2, SP3=3, SP3D=4, SP3D2=5, else=6)
    [10]   : heavydegree        (number of heavy-atom neighbours)
    [11]   : heterodegree       (number of N/O/S/halogen neighbours)
    [12]   : partialcharge      (Gasteiger)
    [13]   : molcode            (+1.0 ligand, -1.0 pocket)
    [14-18]: SMARTS             (hydrophobic, aromatic, acceptor, donor, ring)

Usage:
    python generate_mlhdf_rdkit.py \
        --refined-dir  C:/bindingaffinity/data/refined-set \
        --csv          C:/bindingaffinity/data/affinity_data_fixed.csv \
        --output       C:/bindingaffinity/FAST-master/model/quantum_fusion/refined_all.hdf
"""

import os
import sys
import argparse

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType

# ---------------------------------------------------------------------------
# VDW radii (Bondi 1964, Angstroms), keyed by atomic number
# ---------------------------------------------------------------------------
_VDW = {
    1: 1.20, 5: 1.80, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
    11: 2.27, 12: 1.73, 14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75,
    19: 2.75, 20: 2.00, 26: 2.00, 27: 2.00, 28: 1.63, 29: 1.40,
    30: 1.39, 35: 1.85, 34: 1.90, 53: 1.98,
}
_DEFAULT_VDW = 1.80


def _vdw(anum):
    return _VDW.get(anum, _DEFAULT_VDW)


# ---------------------------------------------------------------------------
# Atom-type codes (same as tfbio Featurizer defaults)
# ---------------------------------------------------------------------------
_METALS = (
    [3, 4, 11, 12, 13]
    + list(range(19, 32))
    + list(range(37, 51))
    + list(range(55, 84))
    + list(range(87, 104))
)
_ATOM_CLASSES = [
    (5,  "B"),
    (6,  "C"),
    (7,  "N"),
    (8,  "O"),
    (15, "P"),
    (16, "S"),
    (34, "Se"),
    ([9, 17, 35, 53], "halogen"),
    (_METALS,         "metal"),
]
NUM_ATOM_CLASSES = len(_ATOM_CLASSES)           # 9

_ATOM_CODES: dict = {}
for _code, (_atom, _name) in enumerate(_ATOM_CLASSES):
    if isinstance(_atom, list):
        for _a in _atom:
            _ATOM_CODES[_a] = _code
    else:
        _ATOM_CODES[_atom] = _code


# ---------------------------------------------------------------------------
# SMARTS patterns (same as tfbio defaults)
# ---------------------------------------------------------------------------
_SMARTS_DEFS = [
    "[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]",   # hydrophobic
    "[a]",                                                        # aromatic
    "[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]",  # acceptor
    "[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]",                  # donor
    "[r]",                                                        # ring
]
_SMARTS_PATS = [Chem.MolFromSmarts(s) for s in _SMARTS_DEFS]


def _hyb_int(hyb: HybridizationType) -> float:
    _map = {
        HybridizationType.SP:    1.0,
        HybridizationType.SP2:   2.0,
        HybridizationType.SP3:   3.0,
        HybridizationType.SP3D:  4.0,
        HybridizationType.SP3D2: 5.0,
    }
    return _map.get(hyb, 6.0)


def _smarts_flags(mol, n_atoms: int) -> np.ndarray:
    """Return (5, n_atoms) boolean array for SMARTS matches."""
    flags = np.zeros((len(_SMARTS_PATS), n_atoms), dtype=np.float32)
    for pi, pat in enumerate(_SMARTS_PATS):
        if pat is None:
            continue
        try:
            for match in mol.GetSubstructMatches(pat):
                for idx in match:
                    if idx < n_atoms:
                        flags[pi, idx] = 1.0
        except Exception:
            pass
    return flags


_HETERO = {7, 8, 9, 15, 16, 17, 34, 35, 53}


def featurize_mol(mol, molcode: float):
    """
    Compute per-atom (coords, feats, vdw) for *mol*.

    Returns
    -------
    coords : (N, 3) float32
    feats  : (N, 19) float32
    vdw    : (N,)   float32
    """
    if mol is None or not mol.GetNumConformers():
        return None, None, None

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    # Ensure ring info is initialised (may be absent when sanitize=False was used)
    try:
        Chem.FastFindRings(mol)
    except Exception:
        pass

    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    sf = _smarts_flags(mol, n)

    coords = np.zeros((n, 3), dtype=np.float32)
    feats  = np.zeros((n, 19), dtype=np.float32)
    vdw    = np.zeros(n, dtype=np.float32)

    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        coords[i] = [pos.x, pos.y, pos.z]

        anum = atom.GetAtomicNum()
        vdw[i] = _vdw(anum)

        # atom-type one-hot [0-8]
        code = _ATOM_CODES.get(anum)
        if code is not None:
            feats[i, code] = 1.0

        # hybridization [9]
        feats[i, 9] = _hyb_int(atom.GetHybridization())

        # heavy degree [10]
        feats[i, 10] = float(sum(
            1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() != 1
        ))

        # hetero degree [11]
        feats[i, 11] = float(sum(
            1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() in _HETERO
        ))

        # Gasteiger partial charge [12]
        try:
            pc = float(atom.GetDoubleProp("_GasteigerCharge"))
            if not np.isfinite(pc):
                pc = 0.0
        except Exception:
            pc = 0.0
        feats[i, 12] = pc

        # molcode [13]
        feats[i, 13] = molcode

        # SMARTS [14-18]
        feats[i, 14:19] = sf[:, i]

    return coords, feats, vdw


def process_complex(pdbid: str, refined_dir: str):
    """Read pocket + ligand, return (data_22col, vdw_arr) or None."""
    base    = os.path.join(refined_dir, pdbid)
    lig_sdf = os.path.join(base, f"{pdbid}_ligand.sdf")
    pkt_pdb = os.path.join(base, f"{pdbid}_pocket.pdb")

    if not os.path.exists(lig_sdf) or not os.path.exists(pkt_pdb):
        return None

    # Load ligand (molcode = +1)
    supp = Chem.SDMolSupplier(lig_sdf, sanitize=True, removeHs=True)
    lig  = next((m for m in supp if m is not None), None)
    if lig is None:
        supp = Chem.SDMolSupplier(lig_sdf, sanitize=False, removeHs=True)
        lig  = next((m for m in supp if m is not None), None)

    # Load pocket (molcode = -1)
    pkt = Chem.MolFromPDBFile(pkt_pdb, sanitize=True, removeHs=True)
    if pkt is None:
        pkt = Chem.MolFromPDBFile(pkt_pdb, sanitize=False, removeHs=True)

    lc, lf, lv = featurize_mol(lig,  molcode= 1.0)
    pc, pf, pv = featurize_mol(pkt, molcode=-1.0)

    if lc is None or pc is None:
        return None

    # pocket atoms first (molcode=-1), then ligand (molcode=+1)
    coords = np.vstack([pc, lc])           # (N, 3)
    feats  = np.vstack([pf, lf])           # (N, 19)
    vdw_arr = np.concatenate([pv, lv])     # (N,)
    data   = np.hstack([coords, feats]).astype(np.float32)  # (N, 22)

    return data, vdw_arr.astype(np.float32)


def generate_hdf(refined_dir: str, csv_path: str, output_path: str,
                 max_samples: int = None, shuffle_seed: int = None):
    df = pd.read_csv(csv_path).dropna(subset=["pdbid", "-logKd/Ki"])
    if shuffle_seed is not None:
        df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
    if max_samples:
        df = df.head(max_samples)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    written = 0
    skipped = 0
    with h5py.File(output_path, "w") as hf:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building HDF"):
            pdbid    = str(row["pdbid"]).strip()
            affinity = float(row["-logKd/Ki"])

            result = process_complex(pdbid, refined_dir)
            if result is None:
                skipped += 1
                continue

            data, vdw_arr = result

            grp = hf.require_group(pdbid)
            grp.attrs["affinity"] = affinity
            sub = grp.require_group("pybel/processed/pdbbind")
            sub.create_dataset("data", data=data, compression="gzip",
                               compression_opts=4)
            sub.attrs["van_der_waals"] = vdw_arr
            written += 1

    print(f"\nDone. Written: {written}  Skipped: {skipped}  "
          f"Output: {output_path}")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ML-HDF from PDBBind refined-set using RDKit"
    )
    parser.add_argument(
        "--refined-dir", default="C:/bindingaffinity/data/refined-set",
        help="path to refined-set directory"
    )
    parser.add_argument(
        "--csv", default="C:/bindingaffinity/data/affinity_data_fixed.csv",
        help="CSV with columns pdbid, -logKd/Ki"
    )
    parser.add_argument(
        "--output",
        default="C:/bindingaffinity/FAST-master/model/quantum_fusion/refined_all.hdf",
        help="output .hdf file path"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="limit number of complexes (useful for testing)"
    )
    args = parser.parse_args()

    generate_hdf(
        refined_dir=args.refined_dir,
        csv_path=args.csv,
        output_path=args.output,
        max_samples=args.max_samples,
    )
