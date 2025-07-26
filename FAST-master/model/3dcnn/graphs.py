import torch
import os


import matplotlib.pyplot as plt
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# best_checkpoint_path = "checkpoint2/best_checkpoint.pth"
# best_checkpoint_dict = torch.load(best_checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
# print(best_checkpoint_dict.keys())
# print(best_checkpoint_dict['model_state_dict'].keys())

checkpoint_dir = "checkpoint2"

r2_scores = []
epochs = []
losses = []
for epoch in range(50):
    checkpoint_path = f"{checkpoint_dir}/model-epoch-{epoch}.pth"

    if not os.path.exists(checkpoint_path):
        print(f"Missing: {checkpoint_path}")
        continue

    epoch_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    r2 = epoch_dict['validate_dict']['r2']
    loss = epoch_dict['validate_dict']['loss']
    losses.append(loss)
    r2_scores.append(r2)
    # print(epoch, ",",loss, ",", r2)
    epochs.append(epoch)


import numpy as np

r2_scores_np = np.array(r2_scores, dtype=np.float32)
losses_np = np.array(losses, dtype=np.float32)




plt.plot(epochs, r2_scores_np, label="lr=1e-3")
plt.title('3D-CNN - R2 variation over training epochs')
plt.xlabel('Epochs')
plt.ylabel('R2')
plt.legend()
plt.show()

plt.plot(epochs, losses_np, label="lr=1e-3")
plt.title('3D-CNN - Loss variation over training epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()