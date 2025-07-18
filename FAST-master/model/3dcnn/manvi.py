import torch
from file_util import *
checkpoint = torch.load("data/pdbbind2021_demo_model_20250718_a1.pth", map_location="cuda:0", weights_only=False)
#checkpoint = torch.load(args.model_path)
print("Checkpoint type", type(checkpoint))

print("Checkpoint keys", checkpoint.keys())

model_state_dict = checkpoint["model_state_dict"]
print("Checkpoint model_state_dict keys", model_state_dict.keys())

print ("\n\n")

optimizer_state_dict = checkpoint["optimizer_state_dict"]  # optimizer_state_dict
print("Checkpoint optimizer_state_dict keys", optimizer_state_dict.keys())

print ("\n\n")


optimizer_state_dict_state = optimizer_state_dict["state"]
print("Checkpoint optimizer_state_dict_state keys", type(optimizer_state_dict_state), optimizer_state_dict_state.keys())

print ("\n\n")


optimizer_state_dict_param_grp = optimizer_state_dict["state"]
print("Checkpoint optimizer_state_dict_state keys", type(optimizer_state_dict_param_grp), optimizer_state_dict_param_grp.keys())
