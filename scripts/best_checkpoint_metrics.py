import torch
import os


import matplotlib.pyplot as plt
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


best_checkpoint_path = "checkpoint2/best_checkpoint.pth"
best_checkpoint_dict = torch.load(best_checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
print(best_checkpoint_dict.keys())
print(best_checkpoint_dict['model_state_dict'].keys())
