### 3D-CNN

Note that the original 3D-CNN implementation used in the paper below has been moved to 3dcnn_tf. This folder contstains new version written using pytorch.


python main_train.py --checkpoint-dir checkpoint-manvi --checkpoint-iter 1 --epoch-count 2 --batch-size 32 --learning-rate 1e-3 --mlhdf-fn pdbbind_2020_refined_train_voxelised.hdf --vmlhdf-fn pdbbind_2020_refined_val_voxelised.hdf



python main_train.py --device-name "cuda:0" --dataset-type 1 --epoch-count 2 --batch-size 10 --learning-rate 1e-3 --checkpoint-iter 1





### Testing

[TEST-lR=7E-4] Evaluation Summary:
RMSE: 1.660, MAE: 1.317, R^2 score: 0.415, Pearson: 0.673, Spearman: 0.669, mean/std: 6.643/1.070
Predictions saved to checkpoint_3dcnn_refined_2020/best_checkpoint_pdbbind2016_core_test_pred.csv
Features saved to checkpoint_3dcnn_refined_2020/best_checkpoint_pdbbind2016_core_test_feat.npz

python main_eval.py --mlhdf-fn "pdbbind2016_core_test.hdf" --model-path "checkpoint_3dcnn_refined_2020_lr_1e-3/best_checkpoint.pth" --batch-size 32
python main_eval.py --mlhdf-fn "pdbbind_2020_refined_val.hdf" --model-path "checkpoint_3dcnn_refined_2020_lr_1e-3/best_checkpoint.pth" --batch-size 32
python main_eval.py --mlhdf-fn "pdbbind_2020_refined_train.hdf" --model-path "checkpoint_3dcnn_refined_2020_lr_1e-3/best_checkpoint.pth" --batch-size 32

[TEST-lR=1E-3]Evaluation Summary:
RMSE: 1.658, MAE: 1.319, R^2 score: 0.417, Pearson: 0.680, Spearman: 0.654, mean/std: 6.592/1.023
Predictions saved to checkpoint_3dcnn_refined_2020_lr_1e-3/best_checkpoint_pdbbind2016_core_test_pred.csv
Features saved to checkpoint_3dcnn_refined_2020_lr_1e-3/best_checkpoint_pdbbind2016_core_test_feat.npz



