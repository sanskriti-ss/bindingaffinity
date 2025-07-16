# Fusion models for Atomic and molecular STructures (FAST)

Predicting accurate protein-ligand binding affinity is important in drug discovery. This code implements fusion network model to benefit from Spatial Grach CNN and 3D CNN models to improve the binding affinity prediction. The code is written in python with Tensorflow and Pytorch.  

 

## Getting Started

### Prerequisites

~~- Tensorflow 1.14 or higher~~
- [PyTorch 1.4 or higher](https://pytorch.org)
- [PyTorch Geometric Feature)](https://github.com/rusty1s/pytorch_geometric)
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

DEBUGGING
--> R2 of 0, y_true is all zeros, y_mean/std is 0
in get_item function
In SG-CNN, they have f['1bzc'].attrs('affinity'), which is returned as y attr


python main_train.py --device-name "cpu" --dataset-type 1 --epoch-count 5 --batch-size 10 --learning-rate 1e-3 --checkpoint-iter 1

--data-dir "data" --mlhdf-fn "train.hdf" --vmlhdf-fn "val.hdf"



--csv-fn ??
--vcsv-fn ??
--model-path ??

--decay-rate 0.95 change?? 
--decay-iter 100 change??



parser.add_argument("--device-name", default="cuda:0", help="use cpu or cuda:0, cuda:1 ...")
parser.add_argument("--data-dir", default="/home/kim63/data", help="dataset directory")
parser.add_argument("--dataset-type", type=float, default=1, help="ml-hdf version, (1: for fusion, 1.5: for cfusion 2: ml-hdf v2)")
parser.add_argument("--mlhdf-fn", default="pdbbind2019_crystal_refined_ml.hdf", help="training ml-hdf path")
parser.add_argument("--csv-fn", default="pdbbind2019_crystal_refined.csv", help="training csv file name")
parser.add_argument("--vmlhdf-fn", default="", help="validation ml-hdf path")
parser.add_argument("--vcsv-fn", default="", help="validation csv file name")
parser.add_argument("--model-path", default="/home/kim63/data/pdbbind2019_crystal_refined_model_20201216.pth", help="model checkpoint file path")
parser.add_argument("--complex-type", type=int, default=1, help="1: crystal, 2: docking")
parser.add_argument("--rmsd-weight", action='store_false', default=0, help="whether rmsd-based weighted loss is used or not")
parser.add_argument("--rmsd-threshold", type=float, default=2, help="rmsd cut-off threshold in case of docking data and/or --rmsd-weight is true")
parser.add_argument("--epoch-count", type=int, default=50, help="number of training epochs")
parser.add_argument("--batch-size", type=int, default=50, help="mini-batch size")
parser.add_argument("--learning-rate", type=float, default=0.0007, help="initial learning rate")
parser.add_argument("--decay-rate", type=float, default=0.95, help="learning rate decay")
parser.add_argument("--decay-iter", type=int, default=100, help="learning rate decay")
parser.add_argument("--checkpoint-iter", type=int, default=50, help="checkpoint save rate")
parser.add_argument("--verbose", type=int, default=0, help="print all input/output shapes or not")

parser.add_argument("--multi-gpus", default=False, action="store_true", help="whether to use multi-gpus")


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

example evaluation: 
`python main_eval.py  --data-dir [directory storing data hdf and csv files]  --mlhdf-fn [hdf file name]  --model-path [full path to model checkpoint file (.pth)] --complex-type [1: crystal, 2: docking]`

`python main_eval.py  --data-dir /a/b/c  --mlhdf-fn data_ml.hdf  --model-path d/e/model_3dcnn_01.pth --complex-type 2 --save-pred --save-feat`

Note that `model/3dcnn/data_reader.py` is a default data reader that reads our ML-HDF format described above. Please use your own data_reader to read your own format.




#### Pre-trained weights (checkpoint files)

We trained all of the networks above on [pdbbind 2016 datasets](http://www.pdbbind.org.cn). Particularly, we used general and refined datasets for training and validation, and evaluated the model on the core set (see sample_data/core_test.hdf). 

The checkpoint files for the models are made available under the Creative Commons BY 4.0 license. See the license section below for the terms of the license. The files can be found here: `ftp://gdo-bioinformatics.ucllnl.org/fast/pdbbind2016_model_checkpoints/`. 

Note that the new 3dcnn checkpoint for pytorch (model_checkpoint_3dcnn.tgz) was trained on pdbbind 2019 refined dataset.  






