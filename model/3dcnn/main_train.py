################################################################################
# Copyright 2019-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# File utility functions
################################################################################

import os
import sys
sys.stdout.flush()
sys.path.insert(0, "../common")
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import wandb

from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Adam, RMSprop, lr_scheduler
from torch.utils.data import Dataset, DataLoader, Subset

from model import Model_3DCNN, strip_prefix_if_present
from data_reader import Dataset_MLHDF
from img_util import GaussianFilter, Voxelizer3D
from file_util import *
import wandb
from sklearn.metrics import r2_score as sklearn_r2_score


def calculate_r2(y_true, y_pred):
    """Calculate R² (coefficient of determination)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2




# seed all random number generators and set cudnn settings for deterministic: https://github.com/rusty1s/pytorch_geometric/issues/217
#random.seed(0)
#np.random.seed(0)
#torch.manual_seed(0)
#torch.cuda.manual_seed(0)
#torch.cuda.manual_seed_all(0)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False  # NOTE: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
#os.environ["PYTHONHASHSEED"] = "0"



# program arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device-name", default="cuda:0", help="use cpu or cuda:0, cuda:1 ...")
parser.add_argument("--data-dir", default="/home/kim63/data", help="dataset directory")
parser.add_argument("--dataset-type", type=float, default=1, help="ml-hdf version, (1: for fusion, 1.5: for cfusion 2: ml-hdf v2)")
parser.add_argument("--mlhdf-fn", default="pdbbind2019_crystal_refined_ml.hdf", help="training ml-hdf path")
parser.add_argument("--csv-fn", default="pdbbind2019_crystal_refined.csv", help="training csv file path")
parser.add_argument("--vmlhdf-fn", default="", help="validation ml-hdf path")
parser.add_argument("--vcsv-fn", default="", help="validation csv file path")
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
args = parser.parse_args()


# set CUDA for PyTorch
use_cuda = torch.cuda.is_available()
cuda_count = torch.cuda.device_count()
if use_cuda:
	device = torch.device(args.device_name)
	torch.cuda.set_device(int(args.device_name.split(':')[1]))
else:
	device = torch.device("cpu")
print(use_cuda, cuda_count, device)



class WeightedMSELoss(nn.Module):
	def __init__(self):
		super(WeightedMSELoss, self).__init__()

	def forward(self, y_pred, y_true, weight):
		return (weight * (y_pred - y_true) ** 2).mean()


def worker_init_fn(worker_id):
	np.random.seed(int(0))

wandb_api_key = os.environ.get("WANDB_API_KEY")

def train():
    # Initialize wandb
	if wandb_api_key:
		wandb.login(key=wandb_api_key)
	wandb.init(
		project="binding-affinity",
		config={
			"learning_rate": args.learning_rate,
			"batch_size": args.batch_size,
			"epochs": args.epoch_count,
			"decay_rate": args.decay_rate,
			"decay_iter": args.decay_iter,
			"dataset": args.mlhdf_fn,
			"complex_type": args.complex_type,
			"rmsd_weight": args.rmsd_weight
		}
	)

	# load dataset
	if args.complex_type == 1:
		is_crystal = True
	else:
		is_crystal = False
	dataset = Dataset_MLHDF(os.path.join(args.data_dir, args.mlhdf_fn), args.dataset_type, os.path.join(args.data_dir, args.csv_fn), is_crystal=is_crystal, rmsd_weight=args.rmsd_weight, rmsd_thres=args.rmsd_threshold)

	# if validation set is available
	val_dataset = None
	if len(args.vmlhdf_fn) > 0:
		val_dataset = Dataset_MLHDF(os.path.join(args.data_dir, args.vmlhdf_fn), args.dataset_type, os.path.join(args.data_dir, args.vcsv_fn), is_crystal=is_crystal, rmsd_weight=args.rmsd_weight, rmsd_thres=args.rmsd_threshold)

	# check multi-gpus
	num_workers = 0
	if args.multi_gpus and cuda_count > 1:
		num_workers = cuda_count

	# initialize data loader
	batch_count = len(dataset) // args.batch_size
	dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=None)
	
	# if validation set is available
	val_dataloader = None
	if val_dataset:
		val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=None)

	# define voxelizer, gaussian_filter
	voxelizer = Voxelizer3D(use_cuda=use_cuda, verbose=args.verbose)
	gaussian_filter = GaussianFilter(dim=3, channels=22, kernel_size=11, sigma=1, use_cuda=use_cuda)

	# define model
	model = Model_3DCNN(use_cuda=use_cuda, verbose=args.verbose, feat_dim=22, quantum=True, dropout_rate=0.2)
	model._init_normal_(dataset.labels)  # initialize mean and std for normalization
	#if use_cuda:
	#	model = model.cuda()
	if args.multi_gpus and cuda_count > 1:
		model = nn.DataParallel(model)
	model.to(device)
	
	if isinstance(model, (DistributedDataParallel, DataParallel)):
		model_to_save = model.module
	else:
		model_to_save = model

	# set loss, optimizer, decay, other parameters
	if args.rmsd_weight == True:
		loss_fn = WeightedMSELoss().float()
	else:
		loss_fn = nn.MSELoss().float()
	#optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
	optimizer = RMSprop(model.parameters(), lr=args.learning_rate)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_iter, gamma=args.decay_rate)

	# load model
	best_loss = 1e10
	best_model_path = args.model_path.replace('.pth', '_best_val.pth')
	best_val_loss = None
	epoch_start = 0
	if valid_file(args.model_path):
		checkpoint = torch.load(args.model_path, map_location=device)
		#checkpoint = torch.load(args.model_path)
		model_state_dict = checkpoint.pop("model_state_dict")
		strip_prefix_if_present(model_state_dict, "module.")
		model_to_save.load_state_dict(model_state_dict, strict=False)
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		epoch_start = checkpoint["epoch"]
		loss = checkpoint["loss"]
		
		# Load best validation loss if available
		if "best_val_loss" in checkpoint:
			best_val_loss = checkpoint["best_val_loss"]

		print("checkpoint loaded: %s" % args.model_path)

	if not os.path.exists(os.path.dirname(args.model_path)):
		os.makedirs(os.path.dirname(args.model_path))
	output_dir = os.path.dirname(args.model_path)

	step = 0
	for epoch_ind in range(epoch_start, args.epoch_count):
		vol_batch = torch.zeros((args.batch_size,19,48,48,48)).float().to(device)
		losses = []
		model.train()
		train_predictions = []
		train_targets = []
		for batch_ind, batch in enumerate(dataloader):

			# transfer to GPU
			if args.rmsd_weight == True:
				x_batch_cpu, y_batch_cpu, w_batch_cpu, _ = batch
			else:
				x_batch_cpu, y_batch_cpu, _ = batch
			x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
			
			# Check if data is already 3D (from 3D CNN HDF) or needs voxelization
			if len(x_batch.shape) == 5:  # Already 3D: [batch, channels, depth, height, width]
				vol_batch = gaussian_filter(x_batch)
			else:  # Point cloud data that needs voxelization
				vol_batch = torch.zeros((args.batch_size,19,48,48,48)).float().to(device)
				for i in range(x_batch.shape[0]):
					xyz, feat = x_batch[i,:,:3], x_batch[i,:,3:]
					vol_batch[i,:,:,:,:] = voxelizer(xyz, feat)
				vol_batch = gaussian_filter(vol_batch)
			
			# forward training
			ypred_batch, _ = model(vol_batch[:x_batch.shape[0]])

			# compute loss
			if args.rmsd_weight == True:
				loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.float(), w_batch_cpu.float())
			else:
				loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.float())
				
			losses.append(loss.cpu().data.item())
			
			# Collect training predictions and targets for R² calculation
			train_predictions.extend(ypred_batch.cpu().detach().numpy().flatten())
			train_targets.extend(y_batch_cpu.numpy().flatten())
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

			# Log batch metrics
			wandb.log({
				"batch_loss": loss.cpu().data.item(),
				"learning_rate": optimizer.param_groups[0]['lr'],
				"step": step
			})

			print("[%d/%d-%d/%d] training, loss: %.3f, lr: %.7f" % (epoch_ind+1, args.epoch_count, batch_ind+1, batch_count, loss.cpu().data.item(), optimizer.param_groups[0]['lr']))
			
			if step % args.checkpoint_iter == 0:
				checkpoint_dict = {
					"model_state_dict": model_to_save.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"loss": loss,
					"step": step,
					"epoch": epoch_ind,
					"best_val_loss": best_val_loss
				}
				torch.save(checkpoint_dict, args.model_path)
				print("checkpoint saved: %s" % args.model_path)
			step += 1

		# Calculate training epoch metrics
		train_epoch_loss = np.mean(losses)
		train_predictions = np.array(train_predictions)
		train_targets = np.array(train_targets)
		train_r2 = calculate_r2(train_targets, train_predictions)
		train_mae = np.mean(np.abs(train_targets - train_predictions))
		train_rmse = np.sqrt(np.mean((train_targets - train_predictions) ** 2))

		print("[%d/%d] training, epoch loss: %.3f, R²: %.3f, MAE: %.3f, RMSE: %.3f" % (
			epoch_ind+1, args.epoch_count, train_epoch_loss, train_r2, train_mae, train_rmse))
		
		# Log training metrics
		epoch_metrics = {
			"epoch": epoch_ind + 1,
			"train_loss": train_epoch_loss,
			"train_r2": train_r2,
			"train_mae": train_mae,
			"train_rmse": train_rmse
		}
		
		if val_dataset:
			val_losses = []
			val_predictions = []
			val_targets = []
			model.eval()
			with torch.no_grad():
				for batch_ind, batch in enumerate(val_dataloader):
					if args.rmsd_weight == True:
						x_batch_cpu, y_batch_cpu, w_batch_cpu, _ = batch
					else:
						x_batch_cpu, y_batch_cpu, _ = batch
					x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
					
					# Check if data is already 3D or needs voxelization
					if len(x_batch.shape) == 5:  # Already 3D
						vol_batch = gaussian_filter(x_batch)
					else:  # Point cloud data that needs voxelization
						vol_batch = torch.zeros((args.batch_size,19,48,48,48)).float().to(device)
						for i in range(x_batch.shape[0]):
							xyz, feat = x_batch[i,:,:3], x_batch[i,:,3:]
							vol_batch[i,:,:,:,:] = voxelizer(xyz, feat)
						vol_batch = gaussian_filter(vol_batch)
					
					ypred_batch, _ = model(vol_batch[:x_batch.shape[0]])

					if args.rmsd_weight == True:
						loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.float(), w_batch_cpu.float())
					else:
						loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.float())
						
					val_losses.append(loss.cpu().data.item())
					
					# Collect predictions and targets for R² calculation
					val_predictions.extend(ypred_batch.cpu().numpy().flatten())
					val_targets.extend(y_batch_cpu.numpy().flatten())
					
					print("[%d/%d-%d/%d] validation, loss: %.3f" % (epoch_ind+1, args.epoch_count, batch_ind+1, len(val_dataloader), loss.cpu().data.item()))

				# Calculate validation metrics
				val_epoch_loss = np.mean(val_losses)
				val_predictions = np.array(val_predictions)
				val_targets = np.array(val_targets)
				val_r2 = calculate_r2(val_targets, val_predictions)
				val_mae = np.mean(np.abs(val_targets - val_predictions))
				val_rmse = np.sqrt(np.mean((val_targets - val_predictions) ** 2))
				
				print("[%d/%d] validation, epoch loss: %.3f, R²: %.3f, MAE: %.3f, RMSE: %.3f" % (
					epoch_ind+1, args.epoch_count, val_epoch_loss, val_r2, val_mae, val_rmse))

				# Add validation metrics
				epoch_metrics.update({
					"val_loss": val_epoch_loss,
					"val_r2": val_r2,
					"val_mae": val_mae,
					"val_rmse": val_rmse
				})

				# Best validation loss checkpointing
				if best_val_loss is None:
					best_val_loss = val_epoch_loss
				elif val_epoch_loss < best_val_loss:
					best_val_loss = val_epoch_loss
					best_checkpoint_dict = {
						"model_state_dict": model_to_save.state_dict(),
						"optimizer_state_dict": optimizer.state_dict(),
						"loss": train_epoch_loss,
						"val_loss": val_epoch_loss,
						"step": step,
						"epoch": epoch_ind,
						"best_val_loss": best_val_loss,
						"train_r2": train_r2,
						"val_r2": val_r2,
						"train_mae": train_mae,
						"val_mae": val_mae,
						"train_rmse": train_rmse,
						"val_rmse": val_rmse
					}
					torch.save(best_checkpoint_dict, best_model_path)
					print("*** NEW BEST MODEL *** validation loss: %.3f, saved to: %s" % (val_epoch_loss, best_model_path))
					
					# Log best model info to wandb
					wandb.log({
						"best_val_loss": best_val_loss,
						"best_model_epoch": epoch_ind + 1
					})

		# Log all epoch metrics
		wandb.log(epoch_metrics)

	# close dataset
	dataset.close()
	if val_dataset:
		val_dataset.close()
	wandb.finish()


def main():
	train()

if __name__ == "__main__":
	main()
