import torch
import os


import matplotlib.pyplot as plt


import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
# Hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # Don't put tick labels at the top
ax2.xaxis.tick_bottom()


# best_checkpoint_path = "checkpoint2/best_checkpoint.pth"
# best_checkpoint_dict = torch.load(best_checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
# print(best_checkpoint_dict.keys())
# print(best_checkpoint_dict['model_state_dict'].keys())

checkpoint_dir = "checkpoint-lr_1e-4"

r2_scores = []
epochs = []
losses = []

for epoch in range(100):
    checkpoint_path = f"{checkpoint_dir}/model-epoch-{epoch}.pth"
    # output_file(f"{checkpoint_path[:-4]}.html")


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

# Set the y-limits to only show the positive values in ax2 and negative values in ax
ax.set_ylim(0, max(r2_scores_np))
ax2.set_ylim(min(r2_scores_np), 0)

best_checkpoint_path = f"{checkpoint_dir}/best_checkpoint.pth"
best_checkpoint_dict = torch.load(best_checkpoint_path, map_location=torch.device('cpu'), weights_only=False)

best_r2 = best_checkpoint_dict['validate_dict']['r2']
if best_r2 > 0:
    ax.axhline(y=best_r2, color='red', linewidth=2, linestyle='--', label=f'Best R2: {best_r2:.4f}')

else:
    ax2.axhline(y=best_r2, color='red', linewidth=2, linestyle='--', label=f'Best R2: {best_r2:.4f}')



lr = best_checkpoint_dict['args']['learning_rate']
ax2.plot(epochs, r2_scores_np)
ax.plot(epochs, r2_scores_np)

ax.legend(handles=ax.lines)
ax2.legend(handles=ax2.lines)


fig.suptitle(f'3D-CNN - Val R2 for lr={lr}')

plt.xlabel('Epochs')
plt.ylabel('R2')
plt.legend()
fig.subplots_adjust(hspace=0)
plt.show()


# # Set the y-limits to only show the positive values in ax2 and negative values in ax
# ax.set_ylim(0, max(losses_np))
# ax2.set_ylim(min(losses_np), 0)


# ax2.plot(epochs, losses_np, label="lr=1e-3")
# ax.plot(epochs, losses_np, label="lr=1e-3")

# fig.suptitle('3D-CNN - loss variation over val epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Losses')
# plt.legend()
# plt.show()
# plt.plot(epochs, losses_np, label="lr=1e-3")
# plt.title('3D-CNN - Loss variation over training epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()