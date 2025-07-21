import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import os
import math

def plot_actual_vs_predicted(y_true, y_pred, model_name, dataset_name, save_path=None):
    """
    Plot actual vs predicted values with comprehensive metrics
    """
    # Compute metrics
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pearson, _ = pearsonr(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)
    
    # Set up figure
    plt.figure(figsize=(10, 8))
    
    # Scatter of actual vs predicted
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k', s=50)
    
    # Perfect prediction line
    mins = min(np.min(y_true), np.min(y_pred))
    maxs = max(np.max(y_true), np.max(y_pred))
    plt.plot([mins, maxs], [mins, maxs],
             'r--', linewidth=2, label='Perfect Prediction')
    
    # Labels & title
    plt.xlabel('Actual Binding Energy (kcal/mol)', fontsize=12)
    plt.ylabel('Predicted Binding Energy (kcal/mol)', fontsize=12)
    plt.title(f'{model_name}: Actual vs Predicted Binding Energies ({dataset_name})', fontsize=14)
    
    # Square axis & grid
    plt.xlim(mins, maxs)
    plt.ylim(mins, maxs)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Metrics annotation in the upper-left of the axes
    metrics_text = (f'R² = {r2:.3f}\n'
                   f'RMSE = {rmse:.3f}\n'
                   f'MAE = {mae:.3f}\n'
                   f'Pearson = {pearson:.3f}\n'
                   f'Spearman = {spearman:.3f}\n'
                   f'N = {len(y_true)}')
    
    plt.text(0.05, 0.95, metrics_text,
             transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Legend
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    
    plt.show()

def load_and_plot_3dcnn_results():
    """
    Load and plot 3DCNN results for all datasets
    """
    # Define file paths for 3DCNN
    base_path = "/home/karen/Projects/FAST/models"
    datasets = ['test', 'train', 'val']
    
    print("=== 3DCNN Results ===")
    
    for dataset in datasets:
        csv_file = f"{base_path}/3dcnn_dropout_02_long_22f_best_val_refined_3d_{dataset}_pred.csv"
        
        if os.path.exists(csv_file):
            print(f"\nProcessing 3DCNN {dataset} dataset...")
            
            # Load data
            df = pd.read_csv(csv_file)
            y_true = df['label'].values
            y_pred = df['pred'].values
            
            print(f"Loaded {len(df)} samples from {dataset} dataset")
            print(f"Label range: {y_true.min():.2f} to {y_true.max():.2f}")
            print(f"Prediction range: {y_pred.min():.2f} to {y_pred.max():.2f}")
            
            # Create save path
            save_path = f"/home/karen/Projects/FAST/plots/3dcnn_{dataset}_actual_vs_predicted.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Plot
            plot_actual_vs_predicted(
                y_true, y_pred, 
                "3DCNN", 
                dataset.capitalize(),
                save_path
            )
        else:
            print(f"File not found: {csv_file}")

def load_and_plot_sgcnn_results():
    """
    Load and plot SGCNN results for all datasets
    """
    # Define file paths for SGCNN
    base_path = "/home/karen/Projects/FAST/results/sgcnn"
    datasets = ['test', 'train', 'val']
    
    print("\n=== SGCNN Results ===")
    
    for dataset in datasets:
        csv_file = f"{base_path}/refined_{dataset}_pred.csv"
        
        if os.path.exists(csv_file):
            print(f"\nProcessing SGCNN {dataset} dataset...")
            
            # Load data
            df = pd.read_csv(csv_file)
            y_true = df['label'].values
            y_pred = df['pred'].values
            
            print(f"Loaded {len(df)} samples from {dataset} dataset")
            print(f"Label range: {y_true.min():.2f} to {y_true.max():.2f}")
            print(f"Prediction range: {y_pred.min():.2f} to {y_pred.max():.2f}")
            print(f"Sample PDB IDs: {df['pdbid'].head(3).tolist()}")
            
            # Create save path
            save_path = f"/home/karen/Projects/FAST/plots/sgcnn_{dataset}_actual_vs_predicted.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Plot
            plot_actual_vs_predicted(
                y_true, y_pred, 
                "SGCNN", 
                dataset.capitalize(),
                save_path
            )
        else:
            print(f"File not found: {csv_file}")

def compare_models_side_by_side():
    """
    Create side-by-side comparison plots for each dataset
    """
    print("\n=== Model Comparison Plots ===")
    
    datasets = ['test', 'train', 'val']
    
    for dataset in datasets:
        # Load 3DCNN data
        cnn3d_file = f"/home/karen/Projects/FAST/models/3dcnn_dropout_02_long_22f_best_val_refined_3d_{dataset}_pred.csv"
        sgcnn_file = f"/home/karen/Projects/FAST/results/sgcnn/refined_{dataset}_pred.csv"
        
        if os.path.exists(cnn3d_file) and os.path.exists(sgcnn_file):
            print(f"\nCreating comparison plot for {dataset} dataset...")
            
            # Load data
            cnn3d_df = pd.read_csv(cnn3d_file)
            sgcnn_df = pd.read_csv(sgcnn_file)
            
            # Create side-by-side plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # 3DCNN plot
            y_true_3d = cnn3d_df['label'].values
            y_pred_3d = cnn3d_df['pred'].values
            r2_3d = r2_score(y_true_3d, y_pred_3d)
            
            mins_3d = min(np.min(y_true_3d), np.min(y_pred_3d))
            maxs_3d = max(np.max(y_true_3d), np.max(y_pred_3d))
            
            ax1.scatter(y_true_3d, y_pred_3d, alpha=0.6, edgecolor='k', s=50)
            ax1.plot([mins_3d, maxs_3d], [mins_3d, maxs_3d], 'r--', linewidth=2)
            ax1.set_xlabel('Actual Binding Energy (kcal/mol)')
            ax1.set_ylabel('Predicted Binding Energy (kcal/mol)')
            ax1.set_title(f'3DCNN ({dataset.capitalize()})')
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.text(0.05, 0.95, f'R² = {r2_3d:.3f}\nN = {len(y_true_3d)}',
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # SGCNN plot
            y_true_sg = sgcnn_df['label'].values
            y_pred_sg = sgcnn_df['pred'].values
            r2_sg = r2_score(y_true_sg, y_pred_sg)
            
            mins_sg = min(np.min(y_true_sg), np.min(y_pred_sg))
            maxs_sg = max(np.max(y_true_sg), np.max(y_pred_sg))
            
            ax2.scatter(y_true_sg, y_pred_sg, alpha=0.6, edgecolor='k', s=50, color='orange')
            ax2.plot([mins_sg, maxs_sg], [mins_sg, maxs_sg], 'r--', linewidth=2)
            ax2.set_xlabel('Actual Binding Energy (kcal/mol)')
            ax2.set_ylabel('Predicted Binding Energy (kcal/mol)')
            ax2.set_title(f'SGCNN ({dataset.capitalize()})')
            ax2.grid(True, linestyle='--', alpha=0.5)
            ax2.text(0.05, 0.95, f'R² = {r2_sg:.3f}\nN = {len(y_true_sg)}',
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            
            # Save comparison plot
            save_path = f"/home/karen/Projects/FAST/plots/comparison_{dataset}_actual_vs_predicted.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot: {save_path}")
            plt.show()

def main():
    """
    Main function to generate all plots
    """
    print("Generating actual vs predicted plots for all models and datasets...")
    
    # Create plots directory
    os.makedirs("/home/karen/Projects/FAST/plots", exist_ok=True)
    
    # Generate individual model plots
    load_and_plot_3dcnn_results()
    load_and_plot_sgcnn_results()
    
    # Generate comparison plots
    compare_models_side_by_side()
    
    print("\n=== Summary ===")
    print("Generated plots:")
    print("- 3 individual 3DCNN plots (train, val, test)")
    print("- 3 individual SGCNN plots (train, val, test)")
    print("- 3 side-by-side comparison plots (train, val, test)")
    print("Total: 9 plots saved to /home/karen/Projects/FAST/plots/")

if __name__ == "__main__":
    main()