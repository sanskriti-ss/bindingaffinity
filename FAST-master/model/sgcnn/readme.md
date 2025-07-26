# Spatial Graph Neural Network (SG-CNN)

This directory contains the source code for the SG-CNN and is organized as:

        * src: train.py, test.py, ggcnn.py, model.py, data_utils.py
        * scripts: contains various scripts for training different models



### notes Manvi
cd into src folder



#### Sample run

python train.py --checkpoint-dir="checkpoint-test3" --epochs=1 --checkpoint-iter=1 --train-data  data/pdbbind_2020_refined_train.hdf --val-data data/pdbbind_2020_refined_val.hdf

python train.py --checkpoint-dir="checkpoint-test" --epochs=2 --checkpoint-iter=1 --train-data  data/data-200-proteins/train.hdf --val-data data/data-200-proteins/val.hdf


python test.py --checkpoint "checkpoint/best_checkpoint.pth"  --num-workers 1 --output "sg-eval" --test-data "data/pdbbind2016_core_test.hdf" --output-file-name "test_sgcnn.hdf"


python test.py --checkpoint "checkpoint-test/best_checkpoint.pth" --output "sg-eval2" --test-data data/test.hdf --output-file-name="test_sgcnn.hdf"


#### Actual



python train.py --checkpoint=True --num-workers=6 --batch-size=32  --non-covalent-k=2 --checkpoint-iter=1 --train-data  data/train.hdf --val-data data/val.hdf  --checkpoint-dir="checkpoint-test3"  --epochs=100


python test.py --checkpoint "checkpoint/best_checkpoint.pth"  --preprocessing-type=processed --feature-type=pybel --dataset-name pdbbind --num-workers 1 --output "sg-eval" --test-data data/test.hdf --output-file-name="data/pdbbind2016_core_test.hdf"

python test.py --checkpoint "checkpoint/best_checkpoint.pth"  --dataset-name pdbbind --num-workers 1 --output "sg-eval" --test-data data/val.hdf --output-file-name="val_sgcnn.hdf"

python test.py  --num-workers 1 --output "sg-eval-refined-2020" --test-data data/pdbbind_2020_refined_train.hdf --output-file-name="train_sgcnn.hdf"
python test.py  --num-workers 1 --output "sg-eval-refined-2020" --test-data data/pdbbind_2020_refined_val.hdf --output-file-name="val_sgcnn.hdf"
python test.py  --num-workers 1 --output "sg-eval-refined-2020" --test-data data/pdbbind2016_core_test.hdf --output-file-name="test_sgcnn.hdf"


