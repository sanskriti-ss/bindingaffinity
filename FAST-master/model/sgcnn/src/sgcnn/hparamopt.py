import json
import os
from train import TrainArgs, train, _DATA_DIR

RESULTS_FILE = os.path.join(_DATA_DIR, "hparam_results_sgcnn.json")

n_epochs = 50

lrs = [5e-5, 5e-4, 5e-3]
batch_sizes = [50, 100]

results = []

for lr in lrs:
    for bs in batch_sizes:
        res = train(TrainArgs(
            lr=lr,
            batch_size=bs,
            epochs=n_epochs,
            checkpoint_iter=1,
        ))
        entry = {
            "hyperparams": {"lr": lr, "batch_size": bs},
            "validate_dict": res["validate_dict"] if res else None,
        }
        results.append(entry)

        # Write after each run so progress isn't lost on disconnect
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)

print(f"Results saved to {RESULTS_FILE}")
