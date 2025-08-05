import torch
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def get_checkpoint_data(checkpoint_dir):

    val_r2_scores = []

    best_ckpt_val_r2 = -1e16
    best_ckpt_epoch = None
    best_ckpt_loss = None

    for epoch in range(50):
        checkpoint_path = f"{checkpoint_dir}/model-epoch-{epoch}.pth"
        # output_file(f"{checkpoint_path[:-4]}.html")


        if not os.path.exists(checkpoint_path):
            print(f"Missing: {checkpoint_path}")
            continue

        epoch_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        r2 = epoch_dict['validate_dict']['r2']

        if r2 > best_ckpt_val_r2:
            best_ckpt_val_r2 = r2
            best_ckpt_epoch = epoch
            best_ckpt_loss = epoch_dict['validate_dict']['loss']

        val_r2_scores.append(r2)

        

    return val_r2_scores,best_ckpt_val_r2, best_ckpt_epoch, best_ckpt_loss


checkpoint_dirs = [r"C:\Users\user\Documents\bindingaffinity\FAST-master\model\3dcnn\checkpoint-lr-4e-3-dr-0.95",
                   r"C:\Users\user\Documents\bindingaffinity\FAST-master\model\3dcnn\checkpoint-lr-4e-3-dr-0.9",
                   r"C:\Users\user\Documents\bindingaffinity\FAST-master\model\3dcnn\checkpoint-lr-4e-3-dr-0.85",
                   r"C:\Users\user\Documents\bindingaffinity\FAST-master\model\3dcnn\checkpoint-lr-4e-3-dr-0.5",

                   
                #    r"C:\Users\user\Documents\bindingaffinity\FAST-master\model\3dcnn\checkpoint-lr-1e-2-dr-0.95",
                   ]

# Create figure with two subplots
fig, axs = plt.subplots(2, 1, sharex=True) #figsize=(12, 10),
best_r2_axs0 = []
best_r2_ax1 = []

default_cycler = mpl.rcParams['axes.prop_cycle']
color_cycle = iter(default_cycler)

for checkpoint_dir in checkpoint_dirs:
    val_r2_scores, best_ckpt_val_r2, best_ckpt_epoch, best_ckpt_loss = get_checkpoint_data(checkpoint_dir)

    color = next(color_cycle)['color']

    epoch_0_path = f"{checkpoint_dir}/model-epoch-0.pth"
    epoch_0_dict = torch.load(epoch_0_path, map_location=torch.device('cpu'), weights_only=False)

    lr = epoch_0_dict['args']['learning_rate']
    dr = epoch_0_dict['args']['decay_rate']

    label = f'Best R2(dr={dr}): {best_ckpt_val_r2:.4f}'

    # Subplot 1: Full range with symlog
    epochs = list(range(len(val_r2_scores)))
    axs[0].plot(epochs, val_r2_scores, linestyle='-', color=color, label=label)
    axs[0].set_yscale('symlog', linthresh=1)
    axs[0].set_ylabel('Validation R²')
    axs[0].set_title('Validation R² - Full Range (symlog scale)')
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[0].legend()

    # Subplot 2: Even tighter zoom on positive region
    axs[1].plot(epochs, val_r2_scores, linestyle='-', color=color, label='val_r2')
    axs[1].set_ylim(0.2, 0.5)  # Even narrower range
    axs[1].set_ylabel('Validation R²')
    axs[1].set_xlabel('Epoch')
    axs[1].set_title('Validation R² - Zoomed In (0.2 to 0.5)')
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
 

    # best_r2_line1 = axs[0].axhline(y=best_ckpt_val_r2, color=color, linewidth=2, linestyle='--', label=label)
    # best_r2_axs0.append(best_r2_line1)

    best_r2_line2 = axs[1].axhline(y=best_ckpt_val_r2, color=color, linewidth=2, linestyle='--', label=label)
    best_r2_ax1.append(best_r2_line2)
    # axs[1].legend()

# axs[0].legend(handles=best_r2_axs0)
axs[0].legend()
axs[1].legend(handles=best_r2_ax1)

fig.suptitle(f'[3D-CNN] Validation R2 variation for lr={lr} with varying decay rates')
plt.tight_layout()
plt.show()

