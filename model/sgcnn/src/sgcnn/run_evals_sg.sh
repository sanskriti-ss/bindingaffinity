#!/bin/bash

# Set common parameters
DATA_DIR="/home/karen/Projects/FAST/data/refined_splits"
MODEL_PATH="/home/karen/Projects/FAST/models/sgcnn-feat22.pth"
OUTPUT_DIR="/home/karen/Projects/FAST/results/sgcnn"
BATCH_SIZE=32

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "Evaluating SGCNN on training set..."
python test.py \
  --checkpoint ${MODEL_PATH} \
  --test-data ${DATA_DIR}/refined_train.hdf \
  --output ${OUTPUT_DIR} \
  --output-file-name refined_train \
  --batch-size ${BATCH_SIZE} \
  --save-pred \
  --save-feat \
  --force-cpu

echo "Evaluating SGCNN on validation set..."
python test.py \
  --checkpoint ${MODEL_PATH} \
  --test-data ${DATA_DIR}/refined_val.hdf \
  --output ${OUTPUT_DIR} \
  --output-file-name refined_val \
  --batch-size ${BATCH_SIZE} \
  --save-pred \
  --save-feat \
  --force-cpu

echo "Evaluating SGCNN on test set..."
python test.py \
  --checkpoint ${MODEL_PATH} \
  --test-data ${DATA_DIR}/refined_test.hdf \
  --output ${OUTPUT_DIR} \
  --output-file-name refined_test \
  --batch-size ${BATCH_SIZE} \
  --save-pred \
  --save-feat \
  --force-cpu

echo "All SGCNN evaluations completed!"