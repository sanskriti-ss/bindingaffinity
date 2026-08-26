import pandas as pd, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_kernel_study import (encode, quantum_feature_map, linear_quantum_kernel,
                                   eval_kernel_ridge, plot_prediction_scatter)
from main_train import load_with_model_features
from testing_random_unitaries import generate_g3_random_circuits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_qf_dir   = os.path.dirname(os.path.abspath(__file__))
_dcnn_npz  = os.path.join(_qf_dir, 'refined_3dcnn_features.npz')
_sgcnn_npz = os.path.join(_qf_dir, 'refined_sgcnn_features.npz')

df = pd.read_csv(os.path.join(_qf_dir, 'quantum_kernel_study_output', 'kernel_study_results.csv'))
best = df.loc[df['r2'].idxmax()]
circ_idx = int(best['circ_idx'])
print(f"Best circuit: #{circ_idx}  R2={best['r2']}")

sgcnn_f, cnn3d_f, labels_raw, ids = load_with_model_features(
    max_samples=6000, dcnn_npz=_dcnn_npz, sgcnn_npz=_sgcnn_npz)
X_all = np.hstack([sgcnn_f, cnn3d_f]).astype('float32')
n = len(labels_raw)
all_idx = np.arange(n)
bins = pd.qcut(labels_raw, q=min(10, max(2, int(np.sqrt(n)))), labels=False, duplicates='drop')
train_idx, test_idx = train_test_split(all_idx, test_size=0.20, random_state=42,
                                        shuffle=True, stratify=bins)
scaler = StandardScaler().fit(X_all[train_idx])
X_all_sc = scaler.transform(X_all).astype('float32')
label_mean = float(labels_raw[train_idx].mean())
label_std  = float(labels_raw[train_idx].std()) + 1e-8
y_all = (labels_raw - label_mean) / label_std
y_tr, y_te = y_all[train_idx], y_all[test_idx]

X_enc_all = encode(X_all_sc, 6, random_seed=42)
X_enc_tr  = X_enc_all[train_idx]
X_enc_te  = X_enc_all[test_idx]

circuits = generate_g3_random_circuits(6, num_gates=300, num_circuits=circ_idx + 1)
qc = circuits[circ_idx]
phi_tr = quantum_feature_map(qc, X_enc_tr, 6)
phi_te = quantum_feature_map(qc, X_enc_te, 6)
K_q_tr = linear_quantum_kernel(phi_tr, phi_tr)
K_q_te = linear_quantum_kernel(phi_te, phi_tr)
res = eval_kernel_ridge(K_q_tr, K_q_te, y_tr, y_te)

title = f"Quantum Kernel Ridge — Circuit #{circ_idx}\nR²={res['r2']:.4f}  Pearson r={res['pearson']:.4f}"
out   = os.path.join(_qf_dir, 'quantum_kernel_study_output', 'best_circuit_scatter.png')
plot_prediction_scatter(res, label_std, label_mean, title=title, out_path=out)
print("Done.")
