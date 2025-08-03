# Fusion models for Atomic and molecular STructures (FAST)

Predicting accurate protein-ligand binding affinity is important in drug discovery. This code implements fusion network model to benefit from Spatial Grach CNN and 3D CNN models to improve the binding affinity prediction. The code is written in python with Tensorflow and Pytorch.  

 

 ## Getting Started

 ### Prerequisites

 ~~- Tensorflow 1.14 or higher~~
 - [PyTorch 1.4 or higher](https://pytorch.org)
 - [PyTorch Geometric Feature)](https://github.com/rusty1s/pytorch_geometric)
 - [openbabel](https://pypi.org/project/openbabel/)
 - [rdkit](rdkit.org) (optional)
 - [pybel](https://github.com/pybel/pybel)  (optional)
 - [pdbfixer](https://github.com/openmm/pdbfixer)  (optional)
 - [tfbio](https://gitlab.com/cheminfIBB/tfbio)  (optional)


 ### Running the application

 #### Data format

 The implemented networks use a 3D atomic representation as input data in a Hierarchical Data Format (HDF5). 
 Each complex/pocket data is comprised of a list of atoms with their features including 3D coordinates of the atoms (x, y, z) and associated features such as atomic number and charges. For more detail, please refer to the paper in the Citing LIST section.  


 #### 3D-CNN

 Note that the original 3D-CNN implementation used in the paper below has been moved to 3dcnn_tf. A new version using pytorch has been released in `model/3dcnn`


 ##### 3D-CNN tensorflow version (used in the paper)

 To train or test 3D-CNN, run `model/3dcnn_tf/main_3dcnn_pdbbind.py`. 
 Here is an example comand to test a pre-trained 3D-CNN model:

 ```
 python main_3dcnn_pdbbind.py --main-dir "pdbbind_3dcnn" --model-subdir "pdbbind2016_refined" --run-mode 5 --external-hdftype 3 --external-testhdf "eval_set.hdf" --external-featprefix "eval_3dcnn" --external-dir "pdbbind_2019"
 ```

 ##### 3D-CNN pytorch version (new version)

 In this new version, the voxelization process is done on GPU, which improves performance/speed-up. The new version is located in `model/3dcnn`

 To train, run `model/3dcnn/main_train.py`
 To test/evaluate, run `model/3dcnn/model_eval.py`


 #### SG-CNN

 To train or test SG-CNN, run `model/sgcnn/src/train.py` or `model/sgcnn/src/test.py`. 

 For an example training script, see `model/sgcnn/scripts/train_pybel_pdbbind_2016_general_refined.sh`





 ## FAST Resources

 Original FAST Github Repository: https://github.com/LLNL/FAST

 FAST paper
 Jones, D., Kim, H., Zhang, X., Zemla, A., Stevenson, G., Bennett, W. F. D., Kirshner, D., Wong, S. E., Lightstone, F. C., & Allen, J. E. (2021). Improved Protein-Ligand Binding Affinity Prediction with Structure-Based Deep Fusion Inference. Journal of Chemical Information and Modeling, 61(4), 1583–1592. https://doi.org/10.1021/acs.jcim.0c01306

 
