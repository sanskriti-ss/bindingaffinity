# import torch

# # set CUDA for PyTorch
# use_cuda = torch.cuda.is_available()
# cuda_count = torch.cuda.device_count()


# print(f"Use cuda: {use_cuda}, count: {cuda_count}")

import torch
print("PyTorch version:", torch.__version__)
print("CUDA version supported by PyTorch:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA count:", torch.cuda.device_count())
