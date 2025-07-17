################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Generate ML-HDF for sgcnn and 3dcnn
################################################################################


import os
import numpy as np
import h5py
import argparse
from openbabel import pybel, openbabel
import warnings
#from data_generator.atomfeat_util import read_pdb, rdkit_atom_features, rdkit_atom_coords
#from data_generator.chem_info import g_atom_vdw_ligand, g_atom_vdw_protein
import xml.etree.ElementTree as ET
from rdkit.Chem.rdmolfiles import MolFromMol2File
import rdkit
from rdkit import Chem
from rdkit.Chem import rdchem
import pandas as pd
from tqdm import tqdm
from glob import glob


# TODO: compute rdkit features and store them in the output hdf5 file
# TODO: instead of making a file for each split, squash into one?


# TODO: not sure setting these to defaults is a good idea...
parser = argparse.ArgumentParser()
parser.add_argument("--input-pdbbind", default="/home/karen/Projects/pdbbind/PDBbind_v2020_refined/refined-set")
parser.add_argument("--input-docking", default="/home/karen/Projects/pdbbind/PDBbind_v2020_refined/refined-set")
parser.add_argument("--use-docking", default=False, action="store_true")
parser.add_argument("--use-exp", default=True, action="store_true")  # Changed to True since you have experimental data
parser.add_argument("--output", default="/home/karen/Projects/FAST/data")
parser.add_argument("--metadata", default="/home/karen/Projects/FAST/data_util/metadata_for_extract.csv")
args = parser.parse_args()


def parse_element_description(desc_file):
    element_info_dict = {}
    element_info_xml = ET.parse(desc_file)
    for element in element_info_xml.iter():
        if "comment" in element.attrib.keys():
            continue
        else:
            element_info_dict[int(element.attrib["number"])] = element.attrib

    return element_info_dict


def parse_mol_vdw(mol, element_dict):
    vdw_list = []

    if isinstance(mol, pybel.Molecule):
        for atom in mol.atoms:
            # NOTE: to be consistent between featurization methods, throw out the hydrogens
            if int(atom.atomicnum) == 1:
                continue
            if int(atom.atomicnum) == 0:
                continue
            else:
                # Check if element exists in dictionary before adding
                if atom.atomicnum in element_dict:
                    vdw_list.append(float(element_dict[atom.atomicnum]["vdWRadius"]))
                else:
                    # Use a default VDW radius for unknown elements
                    vdw_list.append(2.0)  # Default VDW radius

    elif isinstance(mol, rdkit.Chem.rdchem.Mol):
        for atom in mol.GetAtoms():
            # NOTE: to be consistent between featurization methods, throw out the hydrogens
            if int(atom.GetAtomicNum()) == 1:
                continue
            else:
                # Check if element exists in dictionary before adding
                if atom.GetAtomicNum() in element_dict:
                    vdw_list.append(float(element_dict[atom.GetAtomicNum()]["vdWRadius"]))
                else:
                    # Use a default VDW radius for unknown elements
                    vdw_list.append(2.0)  # Default VDW radius
    else:
        raise RuntimeError("must provide a pybel mol or an RDKIT mol")

    return np.asarray(vdw_list)


def featurize_pybel_complex(ligand_mol, pocket_mol, name, dataset_name):
    """
    Creates 19-channel features for each atom:
    0-8: one-hot element (B, C, N, O, P, S, Se, halogen, metal)
    9: hybridisation (1, 2, 3 → sp, sp2, sp3; 0 = "other")
    10: heavy-atom bonds (# neighbours with Z > 1)
    11: hetero-atom bonds (# neighbours with Z not in {1, 6})
    12-16: one-hot structural (hydrophobic, aromatic, acceptor, donor, ring)
    17: partial charge (Gasteiger)
    18: molecule type (–1 protein, +1 ligand)
    """
    # Ensure Gasteiger partial charges are computed
    charge_model = openbabel.OBChargeModel.FindType("gasteiger")
    if charge_model is None:
        raise RuntimeError("Gasteiger charge model not available in OpenBabel")
    charge_model.ComputeCharges(ligand_mol.OBMol)
    charge_model.ComputeCharges(pocket_mol.OBMol)

    def get_features(mol, mol_type):
        """
        mol_type: +1 for ligand, -1 for protein
        """
        coords = []
        feats = []
        
        for atom in mol.atoms:
            # Skip hydrogens (consistent with parse_mol_vdw)
            if int(atom.atomicnum) == 1:
                continue
            if int(atom.atomicnum) == 0:
                continue
                
            obatom = atom.OBAtom
            x, y, z = atom.coords
            coords.append([x, y, z])
            
            # Initialize 19-dimensional feature vector
            features = np.zeros(19)
            
            # 0-8: One-hot element encoding
            atomic_num = obatom.GetAtomicNum()
            if atomic_num == 5:    # B
                features[0] = 1
            elif atomic_num == 6:  # C
                features[1] = 1
            elif atomic_num == 7:  # N
                features[2] = 1
            elif atomic_num == 8:  # O
                features[3] = 1
            elif atomic_num == 15: # P
                features[4] = 1
            elif atomic_num == 16: # S
                features[5] = 1
            elif atomic_num == 34: # Se
                features[6] = 1
            elif atomic_num in [9, 17, 35, 53]:  # F, Cl, Br, I (halogens)
                features[7] = 1
            elif atomic_num in [11, 12, 13, 19, 20, 25, 26, 27, 28, 29, 30]:  # metals
                features[8] = 1
            # If none of the above, all one-hot elements remain 0
            
            # 9: Hybridization (1=sp, 2=sp2, 3=sp3, 0=other)
            hyb = obatom.GetHyb()
            if hyb in [1, 2, 3]:
                features[9] = hyb
            else:
                features[9] = 0
            
            # 10: Heavy-atom bonds (bonds to atoms with Z > 1)
            heavy_bonds = 0
            # 11: Hetero-atom bonds (bonds to atoms with Z not in {1, 6})
            hetero_bonds = 0
            
            for bond in openbabel.OBAtomBondIter(obatom):
                nbr = bond.GetNbrAtom(obatom)
                nbr_atomic_num = nbr.GetAtomicNum()
                if nbr_atomic_num > 1:
                    heavy_bonds += 1
                    if nbr_atomic_num not in [1, 6]:
                        hetero_bonds += 1
            
            features[10] = heavy_bonds
            features[11] = hetero_bonds
            
            # 12-16: One-hot structural features
            # 12: Hydrophobic (carbon atoms)
            if atomic_num == 6:
                features[12] = 1
            
            # 13: Aromatic
            if obatom.IsAromatic():
                features[13] = 1
            
            # 14: Acceptor
            if obatom.IsHbondAcceptor():
                features[14] = 1
            
            # 15: Donor
            if obatom.IsHbondDonor():
                features[15] = 1
            
            # 16: Ring
            if obatom.IsInRing():
                features[16] = 1
            
            # 17: Partial charge
            features[17] = atom.partialcharge
            
            # 18: Molecule type (-1 for protein, +1 for ligand)
            features[18] = mol_type
            
            feats.append(features)
        
        return np.array(coords, dtype=float), np.array(feats, dtype=float)

    # Process ligand (mol_type = +1)
    ligand_coords, ligand_features = get_features(ligand_mol, mol_type=1)
    if ligand_features.shape[0] == 0:
        raise RuntimeError(f"No heavy atoms found in ligand {name} ({dataset_name} set)")
    if not (ligand_features[:, 17] != 0).any():  # Check partial charges
        raise RuntimeError(f"invalid charges for the ligand {name} ({dataset_name} set)")

    # Process pocket (mol_type = -1)
    pocket_coords, pocket_features = get_features(pocket_mol, mol_type=-1)
    if pocket_features.shape[0] == 0:
        raise RuntimeError(f"No heavy atoms found in pocket {name} ({dataset_name} set)")
    if not (pocket_features[:, 17] != 0).any():  # Check partial charges
        raise RuntimeError(f"invalid charges for the pocket {name} ({dataset_name} set)")

    # Center everything on the ligand centroid
    centroid = ligand_coords.mean(axis=0)
    ligand_coords -= centroid
    pocket_coords -= centroid

    # Stack ligand + pocket
    all_coords = np.vstack((ligand_coords, pocket_coords))
    all_feats = np.vstack((ligand_features, pocket_features))

    # Combine coords and features into final array
    data = np.hstack((all_coords, all_feats))
    return data


def main(): 
 
    affinity_data = pd.read_csv(args.metadata)

    element_dict = parse_element_description("data_util/elements.xml")
 
    failure_dict = {"name": [], "partition": [], "set": [], "error": []}

    for dataset_name, data in tqdm(affinity_data.groupby('set')):
        print("found {} complexes in {} set".format(len(data), dataset_name))

        if not os.path.exists(args.output):
            os.makedirs(args.output) 

        with h5py.File('%s/%s.hdf' % (args.output, dataset_name), 'w') as f:

            for idx, row in tqdm(data.iterrows(), total=data.shape[0]):

                name = row['name']

                affinity = row['-logKd/Ki']

                receptor_path = row['receptor_path']
            

                '''
                    here is where the ligand(s) for both the experimental structure and the docking data need to be loaded.
                    * In order to do this, need an input path for both the experimental data as well as the docking data
                    * For docking data:
                        > Need to know how many poses there are, potentially up to 10 but not always the case
                        > May not have ligand/pocket data for names, need to handle this possibility

                    ######################################################################################################



                            BREAK THE MAIN LOOP INTO TWO PARTS....PROCESS DOCKING and PROCESS CRYSTAL STRUCTURES



                    ######################################################################################################

                '''

                ############################## CREATE THE PDB GROUP ##################################################
                # this is here in order to ensure any dataset that is created has passed the quality check, i.e. no failed complexes enter the output file

                
                grp = f.create_group(str(name))
                grp.attrs['affinity'] = affinity
                pybel_grp = grp.create_group("pybel")
                processed_grp = pybel_grp.create_group("processed")

                
                ############################### PROCESS THE DOCKING DATA ###############################
                if args.use_docking:
                    # READ THE DOCKING LIGAND POSES

                    # pose_path_list = glob("{}/{}/{}_ligand_pose_*.pdb".format(args.input_docking, name, name)) 
                    pose_path_list = glob("{}/{}/{}_ligand_pose_*.mol2".format(args.input_docking, name, name)) 

                    # if there are poses to read then we will read them, otherwise skip to the crystal structure loop
                    if len(pose_path_list) > 0: 

                        # READ THE DOCKING POCKET DATA
                        
                        #docking_pocket_file = "{}/{}/{}_pocket.mol2".format(args.input_docking, name, name)
                        docking_pocket_file = receptor_path

                        if not os.path.exists(docking_pocket_file):
                            warnings.warn("{} does not exists...this is likely due to failure in chimera preprocessing step, skipping to next complex...".format(docking_pocket_file))
                            # NOTE: not putting a continue here because there may be crystal structure data
                        else:
                    
                            # some docking files are corrupt (have nans for coords) and pybel doesn't do a great job of handling that
                            with open(docking_pocket_file, 'r') as handle:
                                data = handle.read()
                                if "nan" in data:
                                    warnings.warn("{} contains corrupt data, nan's".format(docking_pocket_file))
                                    #continue #TODO: THIS MAY PREVENT THE CRYSTAL STRUCTURE DATA FROM BEING PROCESSED

                                else:                    

                                    pose_pocket_vdw = []
 
                                    try:
                                        #docking_pocket = next(pybel.readfile('pdb', docking_pocket_file))
                                        docking_pocket = next(pybel.readfile('mol2', docking_pocket_file))
                                        pose_pocket_vdw = parse_mol_vdw(mol=docking_pocket, element_dict=element_dict)

                                    except StopIteration:
                                        error = "pybel failed to read {} docking pocket file".format(name)
                                        warnings.warn(error) 
                                        failure_dict["name"].append(name), failure_dict["partition"].append("docking") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error)
                
                                    # in some, albeit strange, cases the pocket consists purely of hydrogen, skip over these if that is the case
                                    if len(pose_pocket_vdw) < 1:
                                        error = "{} docking pocket contains no heavy atoms, unable to store vdw radii".format(name)
                                        warnings.warn(error) 
                                        failure_dict["name"].append(name), failure_dict["partition"].append("docking") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error) 
                    
                                    else:

                                        docking = processed_grp.create_group("docking")
                                        for pose_path in pose_path_list:

                                            try: 
                                                #pose_ligand = next(pybel.readfile('pdb', pose_path))
                                                pose_ligand = next(pybel.readfile('mol2', pose_path))
                                                # do not add the hydrogens! they were already added in chimera and it would reset the charges
                                            except:
                                                error = "no ligand for {} ({} set)".format(name, dataset_name)
                                                warnings.warn(error)
                                                failure_dict["name"].append(name), failure_dict["partition"].append("docking") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error)
                                                continue #TODO:THIS MAY PREVENT THE CRYSTAL STRUCTURE DATA FROM BEING PROCESSED

                                            # extract the van der waals radii for the ligand/pocket
                                            pose_ligand_vdw = parse_mol_vdw(mol=pose_ligand, element_dict=element_dict) 

                                            # in case the ligand consists purely of hydrogen, skip over these if that is the case
                                            if len(pose_ligand_vdw) < 1:
                                                error = "{} ligand consists purely of hydrogen, no heavy atoms to featurize".format(name)
                                                warnings.warn(error) 
                                                failure_dict["name"].append(name), failure_dict["partition"].append("docking") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error)
                                                continue #TODO: THIS MAY PREVENT THE CRYSTAL STRUCTURE DATA FROM BEING PROCESSED
                        
                                            try:
                                                pose_data = featurize_pybel_complex(ligand_mol=pose_ligand, pocket_mol=docking_pocket, name=name, dataset_name=dataset_name)
                                            except RuntimeError as error:
                                                failure_dict["name"].append(name), failure_dict["partition"].append("docking") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error)
                                                continue  #TODO:THIS MAY PREVENT THE CRYSTAL STRUCTURE DATA FROM BEING PROCESSED

                                            pose_ligand_pocket_vdw = np.concatenate([pose_ligand_vdw.reshape(-1), 
                                                                                pose_pocket_vdw.reshape(-1)], axis=0)

                                            # enforce a constraint that the number of atoms for which we have features is equal to number for which we have VDW radii  
                                            assert pose_ligand_pocket_vdw.shape[0] == pose_data.shape[0] 


                                            # CREATE THE DOCKING POSE GROUP
                                            #pose_idx = pose_path.split(".pdb")[0].split("_")[-1]
                                            pose_idx = pose_path.split(".mol2")[0].split("_")[-1]
                                            pose_grp = docking.create_group(pose_idx) 

                                            # Now that we have passed the try/except blocks, featurize and store the docking data 
                                            pose_grp.attrs["van_der_waals"] = pose_ligand_pocket_vdw
                        
                                            pose_dataset = pose_grp.create_dataset("data", data=pose_data, 
                                                                shape=pose_data.shape, dtype='float32', compression='lzf') 

                        
                else: 
                    error = "{} does not contain any pose data".format(name)
                    tqdm.write(error)
                    failure_dict["name"].append(name), failure_dict["partition"].append("docking") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error) 

                
                ############################### PROCESS THE CRYSTAL STRUCTURE DATA ###############################
                

                    if args.use_exp: 
                        # Use the receptor_path from metadata and construct ligand_path from it
                        crystal_pocket_path = receptor_path  # This is already correct from metadata
                        crystal_ligand_path = crystal_pocket_path.replace('_pocket.mol2', '_ligand.mol2')
                        
                        # BEGIN QUALITY CONTROL: do not create the dataset until data has been verified
                        try:
                            crystal_ligand = next(pybel.readfile('mol2', crystal_ligand_path))
                        except:
                            error = "no ligand for {} ({} set) at path: {}".format(name, dataset_name, crystal_ligand_path)
                            warnings.warn(error)
                            failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error) 
                            continue

                        try:
                            crystal_pocket = next(pybel.readfile('mol2', crystal_pocket_path))
                        except:
                            error = "no pocket for {} ({} set) at path: {}".format(name, dataset_name, crystal_pocket_path)
                            warnings.warn(error)
                            failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error)
                            continue


                    # extract the van der waals radii for the ligand/pocket
                    crystal_ligand_vdw = parse_mol_vdw(mol=crystal_ligand, element_dict=element_dict) 
                
                    # in some, albeit strange, cases the pocket consists purely of hydrogen, skip over these if that is the case
                    if len(crystal_ligand_vdw) < 1:
                        error = "{} ligand consists purely of hydrogen, no heavy atoms to featurize".format(name)
                        warnings.warn(error) 
                        failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error)
                        continue
 
                    crystal_pocket_vdw = parse_mol_vdw(mol=crystal_pocket, element_dict=element_dict)
                    # in some, albeit strange, cases the pocket consists purely of hydrogen, skip over these if that is the case
                    if len(crystal_pocket_vdw) < 1:
                        error = "{} pocket consists purely of hydrogen, no heavy atoms to featurize".format(name)
                        warnings.warn(error) 
                        failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error)
                        continue

                    crystal_ligand_pocket_vdw = np.concatenate([crystal_ligand_vdw.reshape(-1), crystal_pocket_vdw.reshape(-1)], axis=0)
                    try:
                        crystal_data = featurize_pybel_complex(ligand_mol=crystal_ligand, pocket_mol=crystal_pocket, name=name, dataset_name=dataset_name)
                    except RuntimeError as error:
                        failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error)
                        continue
                
                    # enforce a constraint that the number of atoms for which we have features is equal to number for which we have VDW radii 
                    print(crystal_ligand_pocket_vdw.shape[0], crystal_data.shape[0])
                    assert crystal_ligand_pocket_vdw.shape[0] == crystal_data.shape[0]
    
                    # END QUALITY CONTROL: made it past the try/except blocks....now featurize the data and store into the .hdf file 
                    crystal_grp = processed_grp.create_group("pdbbind")
                    crystal_grp.attrs["van_der_waals"] = crystal_ligand_pocket_vdw 
                    crystal_dataset = crystal_grp.create_dataset("data", data=crystal_data, 
                                                        shape=crystal_data.shape, dtype='float32', compression='lzf') 
                    
      
    failure_df = pd.DataFrame(failure_dict)
    failure_df.to_csv("{}/failure_summary.csv".format(args.output), index=False)

if __name__ == "__main__":
    main()

