# Set common parameters
DATA_DIR="/home/karen/Projects/FAST/data/splits"
MODEL_PATH="/home/karen/Projects/FAST/models/3dcnn_dropout_02_long_22f_best_val.pth"
BATCH_SIZE=50
DEVICE="cpu"

# Evaluate on training set
python main_eval.py \
  --data-dir ${DATA_DIR} \
  --mlhdf-fn refined_3d_train.hdf \
  --csv-fn refined_3d_train.csv \
  --model-path ${MODEL_PATH} \
  --batch-size ${BATCH_SIZE} \
  --save-pred \
  --save-feat \
  --device-name ${DEVICE}

# Evaluate on validation set
python main_eval.py \
  --data-dir ${DATA_DIR} \
  --mlhdf-fn refined_3d_val.hdf \
  --csv-fn refined_3d_val.csv \
  --model-path ${MODEL_PATH} \
  --batch-size ${BATCH_SIZE} \
  --save-pred \
  --save-feat \
  --device-name ${DEVICE}

# Evaluate on test set
python main_eval.py \
  --data-dir ${DATA_DIR} \
  --mlhdf-fn refined_3d_test.hdf \
  --csv-fn refined_3d_test.csv \
  --model-path ${MODEL_PATH} \
  --batch-size ${BATCH_SIZE} \
  --save-pred \
  --save-feat \
  --device-name ${DEVICE}