import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import math


class FusionDataset(Dataset):
    """Dataset for fusion model training"""
    
    def __init__(self, sgcnn_features, cnn3d_features, labels):
        self.sgcnn_features = torch.FloatTensor(sgcnn_features)
        self.cnn3d_features = torch.FloatTensor(cnn3d_features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sgcnn_features[idx], self.cnn3d_features[idx], self.labels[idx]


class FusionModel(nn.Module):
    """PyTorch implementation of the fusion model"""
    
    def __init__(self, sgcnn_dim, cnn3d_dim, dropout=0.0):
        super(FusionModel, self).__init__()
        
        # Individual feature processing layers (matching TF model_fusion_2)
        self.bn_pre_3d = nn.BatchNorm1d(cnn3d_dim)
        self.bn_pre_sgcnn = nn.BatchNorm1d(sgcnn_dim)
        self.fc11 = nn.Linear(sgcnn_dim, 1)

        self.fc12 = nn.Linear(cnn3d_dim, 1)

        self.fc3 = nn.Linear(2, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, sgcnn_feat, cnn3d_feat):
        # Process individual features
        #sgcnn_feat = self.bn_pre_sgcnn(sgcnn_feat)
        #cnn3d_feat = self.bn_pre_3d(cnn3d_feat)
        fc11_z = self.fc11(self.dropout(sgcnn_feat))
        fc11_h = torch.nn.functional.leaky_relu(fc11_z)
            
        fc12_z = self.fc12(self.dropout(cnn3d_feat))
        fc12_h = torch.nn.functional.leaky_relu(fc12_z)
        
        # Concatenate original features with processed features
        #concat = torch.cat([sgcnn_feat, cnn3d_feat, fc11_h, fc12_h], dim=1)
        
        concat = torch.cat([fc11_h, fc12_h], dim=1)
        
        # Final fusion layers
        #fc2_z = self.fc2(concat)
        #fc2_h = torch.nn.functional.relu(self.bn2(fc2_z))

        output = self.fc3(concat)
        return output


def align_features(sgcnn_feat_path, sgcnn_ids_path, cnn3d_feat_path, cnn3d_ids_path):
    """Align SGCNN and 3DCNN features using complex IDs"""
    
    # Load features and IDs
    sgcnn_features = np.load(sgcnn_feat_path)
    sgcnn_ids = np.load(sgcnn_ids_path)
    cnn3d_features = np.load(cnn3d_feat_path)
    cnn3d_ids = np.load(cnn3d_ids_path)
    
    print(f"SGCNN: {len(sgcnn_features)} samples, {sgcnn_features.shape[1]} features")
    print(f"3DCNN: {len(cnn3d_features)} samples, {cnn3d_features.shape[1]} features")
    
    # Create mapping from IDs to indices
    sgcnn_id_to_idx = {str(id_val): idx for idx, id_val in enumerate(sgcnn_ids)}
    cnn3d_id_to_idx = {str(id_val): idx for idx, id_val in enumerate(cnn3d_ids)}
    
    # Find common IDs
    common_ids = set(sgcnn_id_to_idx.keys()) & set(cnn3d_id_to_idx.keys())
    print(f"Common complexes: {len(common_ids)}")
    
    if len(common_ids) == 0:
        raise ValueError("No common complex IDs found between SGCNN and 3DCNN datasets!")
    
    # Align features
    sgcnn_aligned = []
    cnn3d_aligned = []
    aligned_ids = []
    
    for common_id in common_ids:
        sgcnn_idx = sgcnn_id_to_idx[common_id]
        cnn3d_idx = cnn3d_id_to_idx[common_id]
        sgcnn_aligned.append(sgcnn_features[sgcnn_idx])
        cnn3d_aligned.append(cnn3d_features[cnn3d_idx])
        aligned_ids.append(common_id)
    
    return np.array(sgcnn_aligned), np.array(cnn3d_aligned), aligned_ids


def load_labels_from_csv(csv_path, complex_ids):
    """Load labels from CSV file matching the complex IDs"""
    df = pd.read_csv(csv_path)
    
    # Create mapping from complex_id to label
    id_to_label = {}
    for _, row in df.iterrows():
        complex_id = str(row['complex_id'])
        label = float(row['label'])
        id_to_label[complex_id] = label
    
    # Get labels for our aligned complex IDs
    labels = []
    for complex_id in complex_ids:
        if complex_id in id_to_label:
            labels.append(id_to_label[complex_id])
        else:
            raise ValueError(f"Complex ID {complex_id} not found in CSV file")
    
    return np.array(labels)


def evaluate_model(model, dataloader, device):
    """Evaluate model performance"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sgcnn_feat, cnn3d_feat, labels in dataloader:
            sgcnn_feat = sgcnn_feat.to(device)
            cnn3d_feat = cnn3d_feat.to(device)
            labels = labels.to(device)
            
            preds = model(sgcnn_feat, cnn3d_feat,)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    rmse = math.sqrt(mean_squared_error(all_labels, all_preds))
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    pearson, _ = pearsonr(all_labels, all_preds)
    spearman, _ = spearmanr(all_labels, all_preds)
    
    return rmse, mae, r2, pearson, spearman, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgcnn-results-dir", default="/home/karen/Projects/FAST/results/sgcnn", help="SGCNN results directory")
    parser.add_argument("--cnn3d-results-dir", default="/home/karen/Projects/FAST/models", help="3DCNN results directory")
    parser.add_argument("--output-dir", default="/home/karen/Projects/FAST/results/fusion", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu or cuda)")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # File paths
    datasets = ['train', 'val', 'test']
    sgcnn_prefix = "refined"
    cnn3d_prefix = "3dcnn_dropout_02_long_22f_best_val_refined_3d"
    
    # Load and align all datasets
    aligned_data = {}
    
    for dataset in datasets:
        print(f"\nLoading {dataset} dataset...")
        
        sgcnn_feat_path = os.path.join(args.sgcnn_results_dir, f"{sgcnn_prefix}_{dataset}_feat.npy")
        sgcnn_ids_path = os.path.join(args.sgcnn_results_dir, f"{sgcnn_prefix}_{dataset}_complex_ids.npy")
        cnn3d_feat_path = os.path.join(args.cnn3d_results_dir, f"{cnn3d_prefix}_{dataset}_feat.npy")
        cnn3d_ids_path = os.path.join(args.cnn3d_results_dir, f"{cnn3d_prefix}_{dataset}_complex_ids.npy")
        sgcnn_csv_path = os.path.join(args.sgcnn_results_dir, f"{sgcnn_prefix}_{dataset}_pred.csv")
        
        # Align features
        sgcnn_feat, cnn3d_feat, common_ids = align_features(
            sgcnn_feat_path, sgcnn_ids_path, cnn3d_feat_path, cnn3d_ids_path
        )
        
        # Load labels
        labels = load_labels_from_csv(sgcnn_csv_path, common_ids)
        
        aligned_data[dataset] = {
            'sgcnn_feat': sgcnn_feat,
            'cnn3d_feat': cnn3d_feat,
            'labels': labels,
            'ids': common_ids
        }
        
        print(f"{dataset}: {len(labels)} aligned samples")
    
    # Create datasets and dataloaders
    train_dataset = FusionDataset(
        aligned_data['train']['sgcnn_feat'],
        aligned_data['train']['cnn3d_feat'],
        aligned_data['train']['labels']
    )
    
    val_dataset = FusionDataset(
        aligned_data['val']['sgcnn_feat'],
        aligned_data['val']['cnn3d_feat'],
        aligned_data['val']['labels']
    )
    
    test_dataset = FusionDataset(
        aligned_data['test']['sgcnn_feat'],
        aligned_data['test']['cnn3d_feat'],
        aligned_data['test']['labels']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    sgcnn_dim = aligned_data['train']['sgcnn_feat'].shape[1]
    cnn3d_dim = aligned_data['train']['cnn3d_feat'].shape[1]
    
    model = FusionModel(sgcnn_dim, cnn3d_dim, dropout=0.1)  # TF model uses dropout=1.0 (no dropout)
    model.to(device)
    
    print(f"\nModel architecture:")
    print(f"SGCNN features: {sgcnn_dim}")
    print(f"3DCNN features: {cnn3d_dim}")
    print(f"Fusion concatenation: {sgcnn_dim + cnn3d_dim + 10}")
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    patience = 50
    improved = 0

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for sgcnn_feat, cnn3d_feat, labels in train_loader:
            sgcnn_feat = sgcnn_feat.to(device)
            cnn3d_feat = cnn3d_feat.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            preds = model(sgcnn_feat, cnn3d_feat,)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for sgcnn_feat, cnn3d_feat, labels in val_loader:
                sgcnn_feat = sgcnn_feat.to(device)
                cnn3d_feat = cnn3d_feat.to(device)
                labels = labels.to(device)
                
                preds = model(sgcnn_feat, cnn3d_feat,)
                loss = criterion(preds, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Step scheduler
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            improved = 0
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, os.path.join(args.output_dir, 'best_fusion_model.pth'))
            print(f"Best model saved with validation loss: {val_loss:.4f}")
        else:
            improved += 1
            if improved >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_fusion_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on all datasets
    print(f"\nFinal evaluation:")
    
    for dataset, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        rmse, mae, r2, pearson, spearman, labels, preds = evaluate_model(model, loader, device)
        print(f"{dataset.capitalize()}: RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}, Pearson: {pearson:.3f}, Spearman: {spearman:.3f}")
        
        # Save predictions
        pred_df = pd.DataFrame({
            'cid': range(len(labels)),
            'complex_id': aligned_data[dataset]['ids'],
            'label': labels,
            'pred': preds
        })
        pred_df.to_csv(os.path.join(args.output_dir, f'fusion_{dataset}_pred.csv'), index=False)
    
    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, args.epochs + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    history_df.to_csv(os.path.join(args.output_dir, 'training_history.csv'), index=False)
    
    print(f"\nTraining completed! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()