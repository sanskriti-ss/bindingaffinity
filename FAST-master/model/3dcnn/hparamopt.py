import json
import os
from main_train import TrainArgs, train, _DATA_DIR

RESULTS_FILE = os.path.join(_DATA_DIR, "hparam_results_3dcnn.json")

n_epochs = 50

lrs = [5e-5, 5e-4, 5e-3]
weight_decays = [1e-4, 1e-2]
batch_sizes = [50, 100]

results = []

for lr in lrs:
    for wd in weight_decays:
        for bs in batch_sizes:
            res = train(TrainArgs(
                learning_rate=lr,
                weight_decay=wd,
                batch_size=bs,
                epoch_count=n_epochs,
                cosine_T_max=n_epochs,
                checkpoint_iter=1,
            ))
            entry = {
                "hyperparams": {"learning_rate": lr, "weight_decay": wd, "batch_size": bs},
                "validate_dict": res["validate_dict"] if res else None,
                "train_dict": res["train_dict"] if res else None,
            }
            results.append(entry)

            # Write after each run so progress isn't lost on disconnect
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)

print(f"Results saved to {RESULTS_FILE}")
