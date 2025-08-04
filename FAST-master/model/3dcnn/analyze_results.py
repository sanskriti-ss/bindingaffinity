#!/usr/bin/env python3
"""
Comprehensive analysis and visualization of 3D-CNN model predictions and features.
Analyzes the CSV predictions and NPZ features to generate metrics and plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import os
import argparse

# Set style for better plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_and_analyze_predictions(csv_path):
    """Load predictions CSV and calculate all metrics."""
    print(f"Loading predictions from: {csv_path}")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract true and predicted values
    y_true = df['label'].values
    y_pred = df['pred'].values
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, pearson_p = pearsonr(y_true, y_pred)
    spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    
    # Mean and std
    mean_true = np.mean(y_true)
    std_true = np.std(y_true)
    mean_pred = np.mean(y_pred)
    std_pred = np.std(y_pred)
    
    # Print metrics
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    print(f"RMSE:                {rmse:.3f}")
    print(f"MAE:                 {mae:.3f}")
    print(f"R² score:            {r2:.3f}")
    print(f"Pearson correlation: {pearson_corr:.3f} (p={pearson_p:.2e})")
    print(f"Spearman correlation:{spearman_corr:.3f} (p={spearman_p:.2e})")
    print(f"True values -   Mean: {mean_true:.3f}, Std: {std_true:.3f}")
    print(f"Predicted values - Mean: {mean_pred:.3f}, Std: {std_pred:.3f}")
    print("="*60)
    
    return df, {
        'rmse': rmse, 'mae': mae, 'r2': r2,
        'pearson': pearson_corr, 'pearson_p': pearson_p,
        'spearman': spearman_corr, 'spearman_p': spearman_p,
        'mean_true': mean_true, 'std_true': std_true,
        'mean_pred': mean_pred, 'std_pred': std_pred
    }

def load_and_analyze_features(npz_path):
    """Load and analyze features from NPZ file."""
    print(f"\nLoading features from: {npz_path}")
    
    if not os.path.exists(npz_path):
        print(f"Features file not found: {npz_path}")
        return None
    
    # Load NPZ file
    features_data = np.load(npz_path)
    
    print("Available arrays in NPZ file:")
    print(f"  Found {len(features_data.files)} compounds, each with features")
    
    # Show a few examples
    for i, key in enumerate(features_data.files[:5]):
        print(f"  {key}: shape {features_data[key].shape}")
    if len(features_data.files) > 5:
        print(f"  ... and {len(features_data.files) - 5} more")
    
    return features_data

def create_prediction_plots(df, metrics, output_dir="plots"):
    """Create comprehensive prediction analysis plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    y_true = df['label'].values
    y_pred = df['pred'].values
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Scatter plot: True vs Predicted
    ax1 = plt.subplot(2, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    # Regression line
    slope, intercept, _, _, _ = stats.linregress(y_true, y_pred)
    line = slope * np.array([min_val, max_val]) + intercept
    plt.plot([min_val, max_val], line, 'g-', lw=2, label=f'Linear fit (slope={slope:.2f})')
    
    plt.xlabel('True Binding Affinity (pKd/pKi)')
    plt.ylabel('Predicted Binding Affinity')
    plt.title(f'True vs Predicted Values\nR² = {metrics["r2"]:.3f}, RMSE = {metrics["rmse"]:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Residuals plot
    ax2 = plt.subplot(2, 3, 2)
    residuals = y_pred - y_true
    plt.scatter(y_true, residuals, alpha=0.6, s=50)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('True Binding Affinity')
    plt.ylabel('Residuals (Predicted - True)')
    plt.title(f'Residuals Plot\nMAE = {metrics["mae"]:.3f}')
    plt.grid(True, alpha=0.3)
    
    # 3. Distribution comparison
    ax3 = plt.subplot(2, 3, 3)
    plt.hist(y_true, bins=30, alpha=0.7, label='True', density=True)
    plt.hist(y_pred, bins=30, alpha=0.7, label='Predicted', density=True)
    plt.xlabel('Binding Affinity')
    plt.ylabel('Density')
    plt.title('Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Error distribution
    ax4 = plt.subplot(2, 3, 4)
    errors = np.abs(residuals)
    plt.hist(errors, bins=30, alpha=0.7, color='orange')
    plt.axvline(metrics['mae'], color='red', linestyle='--', lw=2, label=f'MAE = {metrics["mae"]:.3f}')
    plt.axvline(metrics['rmse'], color='purple', linestyle='--', lw=2, label=f'RMSE = {metrics["rmse"]:.3f}')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Q-Q plot for residuals normality
    ax5 = plt.subplot(2, 3, 5)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals\n(Check for Normality)')
    plt.grid(True, alpha=0.3)
    
    # 6. Performance by binding affinity range
    ax6 = plt.subplot(2, 3, 6)
    
    # Bin the data by true values
    bins = np.linspace(min(y_true), max(y_true), 6)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_maes = []
    bin_counts = []
    
    for i in range(len(bins)-1):
        mask = (y_true >= bins[i]) & (y_true < bins[i+1])
        if np.sum(mask) > 0:
            bin_mae = np.mean(np.abs(residuals[mask]))
            bin_maes.append(bin_mae)
            bin_counts.append(np.sum(mask))
        else:
            bin_maes.append(0)
            bin_counts.append(0)
    
    bars = plt.bar(bin_centers, bin_maes, width=(bins[1]-bins[0])*0.8, alpha=0.7)
    
    # Add count labels on bars
    for bar, count in zip(bars, bin_counts):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('True Binding Affinity Range')
    plt.ylabel('Mean Absolute Error')
    plt.title('Performance by Affinity Range')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_features_analysis(features_data, df, output_dir="plots"):
    """Analyze and visualize the extracted features."""
    if features_data is None:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all features into a matrix
    compound_ids = list(features_data.files)
    n_compounds = len(compound_ids)
    n_features = features_data[compound_ids[0]].shape[0]
    
    # Create feature matrix
    feature_array = np.zeros((n_compounds, n_features))
    compound_names = []
    
    for i, cid in enumerate(compound_ids):
        feature_array[i] = features_data[cid]
        compound_names.append(cid)
    
    print(f"Analyzing features: {n_compounds} compounds x {n_features} features")
    
    # Create feature analysis plots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Feature distribution
    ax1 = plt.subplot(2, 3, 1)
    plt.hist(feature_array.flatten(), bins=50, alpha=0.7)
    plt.xlabel('Feature Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of All Feature Values')
    plt.grid(True, alpha=0.3)
    
    # 2. Feature correlation with true labels
    ax2 = plt.subplot(2, 3, 2)
    
    # Match compounds with true labels
    y_true_matched = []
    feature_array_matched = []
    
    for i, cid in enumerate(compound_names):
        if cid in df['cid'].values:
            idx = df[df['cid'] == cid].index[0]
            y_true_matched.append(df.loc[idx, 'label'])
            feature_array_matched.append(feature_array[i])
    
    if len(y_true_matched) > 0:
        y_true_matched = np.array(y_true_matched)
        feature_array_matched = np.array(feature_array_matched)
        
        # Calculate correlation between each feature dimension and true labels
        feature_corrs = []
        for i in range(feature_array_matched.shape[1]):
            corr, _ = pearsonr(feature_array_matched[:, i], y_true_matched)
            feature_corrs.append(corr)
        
        plt.plot(feature_corrs, 'o-')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Correlation with True Labels')
        plt.title('Feature-Label Correlations')
        plt.grid(True, alpha=0.3)
        
        # 3. Top correlated features scatter
        ax3 = plt.subplot(2, 3, 3)
        top_idx = np.argmax(np.abs(feature_corrs))
        plt.scatter(feature_array_matched[:, top_idx], y_true_matched, alpha=0.6)
        plt.xlabel(f'Feature {top_idx} (highest correlation)')
        plt.ylabel('True Binding Affinity')
        plt.title(f'Best Feature vs True Labels\n(r = {feature_corrs[top_idx]:.3f})')
        plt.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No matching compounds found', ha='center', va='center', transform=ax2.transAxes)
        ax3 = plt.subplot(2, 3, 3)
        ax3.text(0.5, 0.5, 'No matching compounds found', ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Feature variance
    ax4 = plt.subplot(2, 3, 4)
    feature_vars = np.var(feature_array, axis=0)
    plt.plot(feature_vars, 'o-')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Variance')
    plt.title('Feature Variances')
    plt.grid(True, alpha=0.3)
    
    # 5. Feature mean values
    ax5 = plt.subplot(2, 3, 5)
    feature_means = np.mean(feature_array, axis=0)
    plt.plot(feature_means, 'o-')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Mean Value')
    plt.title('Feature Means')
    plt.grid(True, alpha=0.3)
    
    # 6. Heatmap of feature correlations
    ax6 = plt.subplot(2, 3, 6)
    feature_corr_matrix = np.corrcoef(feature_array.T)
    im = plt.imshow(feature_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im)
    plt.title(f'Feature Correlation Matrix\n({n_features} features)')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'features_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_metrics_comparison_plot(metrics, output_dir="plots"):
    """Create a comparison plot with baseline metrics from README."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Baseline metrics from README
    baselines = {
        'Baseline 1E-3 (batch 12)': {'rmse': 1.729, 'r2': 0.365, 'pearson': 0.649},
        'Baseline 7E-4': {'rmse': 1.660, 'r2': 0.415, 'pearson': 0.673},
        'Your Model (clyde-train-3)': {'rmse': metrics['rmse'], 'r2': metrics['r2'], 'pearson': metrics['pearson']}
    }
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = list(baselines.keys())
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    
    # RMSE comparison
    rmse_values = [baselines[model]['rmse'] for model in models]
    bars1 = axes[0].bar(models, rmse_values, color=colors)
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE Comparison')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars1, rmse_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # R² comparison
    r2_values = [baselines[model]['r2'] for model in models]
    bars2 = axes[1].bar(models, r2_values, color=colors)
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('R² Comparison')
    axes[1].tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars2, r2_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # Pearson correlation comparison
    pearson_values = [baselines[model]['pearson'] for model in models]
    bars3 = axes[2].bar(models, pearson_values, color=colors)
    axes[2].set_ylabel('Pearson Correlation')
    axes[2].set_title('Pearson Correlation Comparison')
    axes[2].tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars3, pearson_values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze 3D-CNN model predictions and features')
    parser.add_argument('--csv', default='clyde-train-3/best_checkpoint_pdbbind2016_core_test_voxelised_pred.csv',
                       help='Path to predictions CSV file')
    parser.add_argument('--npz', default='clyde-train-3/best_checkpoint_pdbbind2016_core_test_voxelised_feat.npz',
                       help='Path to features NPZ file')
    parser.add_argument('--output-dir', default='analysis_plots-1',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    print("3D-CNN Model Analysis")
    print("====================")
    
    # Load and analyze predictions
    df, metrics = load_and_analyze_predictions(args.csv)
    
    # Load and analyze features
    features_data = load_and_analyze_features(args.npz)
    
    # Create visualizations
    print(f"\nCreating plots in directory: {args.output_dir}")
    
    # Prediction analysis
    create_prediction_plots(df, metrics, args.output_dir)
    
    # Features analysis
    create_features_analysis(features_data, df, args.output_dir)
    
    # Metrics comparison
    create_metrics_comparison_plot(metrics, args.output_dir)
    
    print(f"\nAnalysis complete! All plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
