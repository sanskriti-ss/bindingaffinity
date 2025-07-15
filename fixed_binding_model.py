#!/usr/bin/env python3
"""
Binding Affinity Prediction with sklearn models for now

This tries to align NPY data with CSV targets and improves parsing.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')


def parse_binding_value(value):
    """Parse binding values like 'Ki=15.0uM' to extract numeric value in nM"""
    if pd.isna(value) or value == '' or value == 0:
        return np.nan
    
    if isinstance(value, (int, float)) and value != 0:
        return float(value)
    
    # Convert to string and extract number
    value_str = str(value).lower().strip()
    
    # Skip if zero
    if value_str == '0' or value_str == '0.0':
        return np.nan
    
    # Remove prefixes like 'ki=', 'kd=', 'ic50='
    for prefix in ['ki=', 'kd=', 'ic50=']:
        if value_str.startswith(prefix):
            value_str = value_str[len(prefix):]
            break
    
    # Extract number and unit
    import re
    match = re.search(r'([\d.]+(?:e[+-]?\d+)?)\s*([a-z]*)', value_str)
    if match:
        number = float(match.group(1))
        unit = match.group(2).lower()
        
        # Convert to nM (nanomolar) as standard
        if unit in ['um', 'μm', 'microm']:
            number *= 1000  # μM to nM
        elif unit in ['mm', 'millim']:
            number *= 1000000  # mM to nM  
        elif unit in ['pm', 'picom']:
            number /= 1000  # pM to nM
        elif unit in ['m', 'molar']:
            number *= 1000000000  # M to nM
        # Default assume nM
        
        return number
    
    return np.nan


def load_and_parse_csv():
    """Load and properly parse the CSV file"""
    csv_file = 'pdbbind_with_dG.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return None
    
    print(f"Loading and parsing {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Original CSV shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Parse binding values
    df['Ki_numeric'] = df['Ki'].apply(parse_binding_value)
    df['Kd_numeric'] = df['Kd'].apply(parse_binding_value)
    
    # Convert resolution to numeric
    df['resolution'] = pd.to_numeric(df['resolution'], errors='coerce')
    
    # Create a unified binding constant (prefer Ki, then Kd)
    df['binding_constant_nM'] = df['Ki_numeric'].fillna(df['Kd_numeric'])
    
    # Convert binding constant to ΔG if ΔG is missing
    # ΔG = RT * ln(Kd) where R = 1.987 cal/(mol·K), T = 298K
    RT = 1.987 * 298.15 / 1000  # kcal/mol
    df['calculated_dG'] = RT * np.log(df['binding_constant_nM'] * 1e-9)  # Convert nM to M
    
    # Use provided ΔG or calculated one
    df['final_dG'] = df['ΔG_kcal_per_mol'].fillna(df['calculated_dG'])
    
    # Clean data
    print(f"\\nData cleaning:")
    print(f"  Original samples: {len(df)}")
    
    # Remove samples with missing binding data
    valid_mask = ~pd.isna(df['final_dG'])
    df_clean = df[valid_mask].copy()
    print(f"  After removing missing ΔG: {len(df_clean)}")
    
    # Remove outliers (optional)
    q1 = df_clean['final_dG'].quantile(0.01)
    q99 = df_clean['final_dG'].quantile(0.99)
    outlier_mask = (df_clean['final_dG'] >= q1) & (df_clean['final_dG'] <= q99)
    df_clean = df_clean[outlier_mask].copy()
    print(f"  After removing outliers: {len(df_clean)}")
    
    print(f"\\nFinal dataset statistics:")
    print(f"  ΔG range: {df_clean['final_dG'].min():.2f} to {df_clean['final_dG'].max():.2f} kcal/mol")
    print(f"  ΔG mean: {df_clean['final_dG'].mean():.2f} ± {df_clean['final_dG'].std():.2f}")
    
    return df_clean


def load_npy_data():
    """Load and process all NPY data files"""
    data_files = {
        'ligand_grids': 'data/processed_ligand_data/ligand_grids.npy',
        'pocket_grids': 'data/processed_pocket_data/pocket_grids.npy', 
        'protein_grids': 'data/processed_protein_data/protein_grids.npy'
    }
    
    loaded_data = {}
    
    for name, path in data_files.items():
        if os.path.exists(path):
            print(f"Loading {name} from {path}...")
            data = np.load(path)
            print(f"  Shape: {data.shape}")
            loaded_data[name] = data
        else:
            print(f"Warning: {path} not found")
    
    return loaded_data


def extract_advanced_features(grids, method='comprehensive'):
    """Extract comprehensive features from 3D grid data"""
    if len(grids.shape) != 5:
        print(f"Warning: Expected 5D grids, got {grids.shape}")
        return grids.reshape(grids.shape[0], -1)
    
    n_samples, n_channels, d1, d2, d3 = grids.shape
    print(f"Processing {n_samples} samples with {n_channels} channels, grid size {d1}×{d2}×{d3}")
    
    features = []
    
    for i in range(n_samples):
        sample_features = []
        
        for ch in range(n_channels):
            grid = grids[i, ch]
            
            # Basic statistics (expanded)
            sample_features.extend([
                np.mean(grid),
                np.std(grid), 
                np.var(grid),
                np.max(grid),
                np.min(grid),
                np.median(grid),
                np.percentile(grid, 10),
                np.percentile(grid, 25),
                np.percentile(grid, 75),
                np.percentile(grid, 90),
                np.percentile(grid, 95),
                np.percentile(grid, 99),
            ])
            
            # Advanced statistics
            sample_features.extend([
                np.sum(grid > 0),  # Positive voxels
                np.sum(grid < 0),  # Negative voxels
                np.sum(np.abs(grid) > 0.1),  # Significant voxels
                np.sum(np.abs(grid) > 0.5),  # Strong voxels
                np.sum(np.abs(grid) > 1.0),  # Very strong voxels
            ])
            
            # Spatial features
            if method == 'comprehensive':
                # Center of mass
                total_mass = np.sum(np.abs(grid))
                if total_mass > 0:
                    coords = np.indices(grid.shape)
                    cx = np.sum(coords[0] * np.abs(grid)) / total_mass
                    cy = np.sum(coords[1] * np.abs(grid)) / total_mass  
                    cz = np.sum(coords[2] * np.abs(grid)) / total_mass
                    sample_features.extend([cx, cy, cz])
                else:
                    sample_features.extend([0, 0, 0])
                
                # Moments
                try:
                    flat_grid = grid.flatten()
                    from scipy import stats
                    sample_features.extend([
                        stats.skew(flat_grid),
                        stats.kurtosis(flat_grid),
                    ])
                except:
                    sample_features.extend([0, 0])
                
                # Energy-like features
                sample_features.extend([
                    np.sum(grid**2),  # L2 norm
                    np.sum(np.abs(grid)),  # L1 norm
                    np.sqrt(np.sum(grid**2)),  # Euclidean norm
                ])
            
        features.append(sample_features)
    
    features = np.array(features)
    print(f"Extracted features shape: {features.shape}")
    
    # Remove features with zero variance
    feature_std = np.std(features, axis=0)
    valid_features = feature_std > 1e-8
    features = features[:, valid_features]
    print(f"After removing zero-variance features: {features.shape}")
    
    return features


def align_data_with_csv(npy_data, df_clean):
    """Align NPY data with CSV data using protein IDs"""
    print("\\nAligning NPY data with CSV...")
    
    # For now, we'll use the first N samples from each NPY file
    # In a real scenario, you'd need protein ID mapping
    min_samples = min([len(data) for data in npy_data.values()])
    csv_samples = len(df_clean)
    
    # Use the smaller of the two
    n_samples = min(min_samples, csv_samples)
    print(f"Using {n_samples} aligned samples")
    
    # Extract features from each NPY file
    all_features = []
    feature_names = []
    
    for name, grids in npy_data.items():
        print(f"\\nProcessing {name}...")
        features = extract_advanced_features(grids[:n_samples], method='comprehensive')
        all_features.append(features)
        feature_names.extend([f"{name}_{i}" for i in range(features.shape[1])])
    
    # Combine features
    if all_features:
        X = np.concatenate(all_features, axis=1)
        print(f"Combined features shape: {X.shape}")
    else:
        return None, None, None
    
    # Get corresponding targets
    y = df_clean['final_dG'].values[:n_samples]
    
    return X, y, feature_names


def train_optimized_models(X_train, y_train, X_test, y_test):
    """Train optimized models with hyperparameter tuning"""
    
    # Feature selection
    print("\\nPerforming feature selection...")
    selector = SelectKBest(f_regression, k=min(200, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    print(f"Selected {X_train_selected.shape[1]} features")
    
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ),
        'Ridge': Ridge(alpha=10.0),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
    }
    
    results = {}
    predictions = {}
    
    print("\\nTraining optimized models...")
    
    for name, model in models.items():
        print(f"\\nTraining {name}...")
        start_time = time.time()
        
        try:
            # Use selected features
            X_tr, X_te = X_train_selected, X_test_selected
            
            if name in ['Ridge', 'ElasticNet']:
                # Scale for linear models
                scaler = RobustScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)
            
            model.fit(X_tr, y_train)
            y_pred_train = model.predict(X_tr)
            y_pred_test = model.predict(X_te)
            
            # Evaluate
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)

            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)

            train_pearsonr = stats.pearsonr(y_train, y_pred_train)
            test_pearsonr = stats.pearsonr(y_test, y_pred_test)

            train_spearmanr = stats.spearmanr(y_train, y_pred_train)
            test_spearmanr = stats.spearmanr(y_test, y_pred_test)


            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_pearsonr': train_pearsonr,
                'test_pearsonr': test_pearsonr,
                'train_spearmanr': train_spearmanr,
                'test_spearmanr': test_spearmanr,
                'cv_r2_mean': cv_mean,
                'cv_r2_std': cv_std,
                'time': time.time() - start_time
            }
            
            predictions[name] = y_pred_test
            
            print(f"  {name}:")
            print(f"    Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
            print(f"    CV R²: {cv_mean:.4f} ± {cv_std:.4f}")
            print(f"    Time: {time.time() - start_time:.1f}s")
            
        except Exception as e:
            print(f"  {name} failed: {e}")
            continue
    
    return results, predictions


def main():
    print("Fixed High-Performance Binding Affinity Prediction")
    print("=" * 55)
    
    # Load and parse CSV
    df_clean = load_and_parse_csv()
    if df_clean is None:
        return None, None
    
    # Load NPY data
    npy_data = load_npy_data()
    if not npy_data:
        print("No NPY files found!")
        return None, None
    
    # Align data
    X, y, feature_names = align_data_with_csv(npy_data, df_clean)
    if X is None:
        print("Failed to align data!")
        return None, None
    
    print(f"\\nFinal aligned dataset:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Target range: {y.min():.2f} to {y.max():.2f}")
    print(f"  Target mean: {y.mean():.2f} ± {y.std():.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=None
    )
    
    print(f"\\nData split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # Train models
    results, predictions = train_optimized_models(X_train, y_train, X_test, y_test)
    
    if not results:
        print("No models trained successfully!")
        return None, None
    
    import json
    with open('fixed_binding_model_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create ensemble
    print("\\nCreating ensemble...")
    # Weight by CV performance
    weights = np.array([results[name]['cv_r2_mean'] for name in predictions.keys()])
    weights = np.maximum(weights, 0)  # Ensure non-negative
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(weights)) / len(weights)
    
    pred_array = np.array(list(predictions.values())).T
    ensemble_pred = np.average(pred_array, axis=1, weights=weights)
    
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    
    # Results summary
    print("\\n" + "="*55)
    print("FINAL RESULTS SUMMARY")
    print("="*55)
    
    for name, metrics in results.items():
        print(f"{name:15} - Test R²: {metrics['test_r2']:7.4f}, CV R²: {metrics['cv_r2_mean']:7.4f} ± {metrics['cv_r2_std']:.3f}")
    
    print(f"{'Ensemble':15} - Test R²: {ensemble_r2:7.4f}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_r2 = results[best_model]['test_r2']
    
    print(f"\\n🏆 Best model: {best_model} (R² = {best_r2:.4f})")
    print(f"🚀 Ensemble R²: {ensemble_r2:.4f}")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Model comparison
    plt.subplot(2, 3, 1)
    model_names = list(results.keys()) + ['Ensemble']
    r2_scores = [results[name]['test_r2'] for name in results.keys()] + [ensemble_r2]
    
    bars = plt.bar(model_names, r2_scores, alpha=0.7)
    plt.ylabel('Test R²')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Add value labels
    for bar, score in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Best vs Ensemble comparison
    top_3_models = sorted(results.keys(), key=lambda x: results[x]['test_r2'], reverse=True)[:3]
    
    for i, model_name in enumerate(top_3_models):
        plt.subplot(2, 3, i+2)
        y_pred = predictions[model_name]
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual ΔG (kcal/mol)')
        plt.ylabel('Predicted ΔG (kcal/mol)')
        plt.title(f'{model_name}\\nR² = {results[model_name]["test_r2"]:.3f}')
        plt.grid(True, alpha=0.3)
    
    # Ensemble prediction
    plt.subplot(2, 3, 5)
    plt.scatter(y_test, ensemble_pred, alpha=0.6, color='gold', s=50)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual ΔG (kcal/mol)')
    plt.ylabel('Predicted ΔG (kcal/mol)')
    plt.title(f'Ensemble Model\\nR² = {ensemble_r2:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Feature importance (for best tree-based model)
    plt.subplot(2, 3, 6)
    tree_models = ['RandomForest', 'ExtraTrees', 'GradientBoosting']
    best_tree = None
    for model in tree_models:
        if model in results:
            if best_tree is None or results[model]['test_r2'] > results[best_tree]['test_r2']:
                best_tree = model
    
    if best_tree:
        # This is a simplified importance plot
        plt.text(0.5, 0.5, f'Best Tree Model:\\n{best_tree}\\nR² = {results[best_tree]["test_r2"]:.3f}', 
                transform=plt.gca().transAxes, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.title('Best Tree Model Info')
    
    plt.tight_layout()
    plt.savefig('fixed_binding_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\\n✅ Prediction completed!")
    print(f"Results saved to: fixed_binding_prediction_results.png")
    
    return results, (X_train, X_test, y_train, y_test, ensemble_pred)


if __name__ == "__main__":
    try:
        results, data = main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
