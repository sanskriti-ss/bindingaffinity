#!/usr/bin/env python3
"""
Hybrid 3D CNN + Traditional Features Binding Affinity Prediction

Combines 3D CNN feature extraction with traditional ML feature engineering
for optimal performance.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.regularizers import l1_l2
import gc

# Set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")


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
    valid_mask = ~pd.isna(df['final_dG'])
    df_clean = df[valid_mask].copy()
    
    # Remove outliers
    q1 = df_clean['final_dG'].quantile(0.01)
    q99 = df_clean['final_dG'].quantile(0.99)
    outlier_mask = (df_clean['final_dG'] >= q1) & (df_clean['final_dG'] <= q99)
    df_clean = df_clean[outlier_mask].copy()
    
    print(f"Final dataset: {len(df_clean)} samples")
    print(f"ΔG range: {df_clean['final_dG'].min():.2f} to {df_clean['final_dG'].max():.2f} kcal/mol")
    
    return df_clean


def load_grid_data_efficiently(grid_type, n_samples):
    """Load grid data efficiently with memory management"""
    file_paths = {
        'ligand': 'processed_ligand_data/ligand_grids.npy',
        'pocket': 'processed_pocket_data/pocket_grids.npy', 
        'protein': 'processed_protein_data/protein_grids.npy'
    }
    
    if grid_type not in file_paths:
        return None
    
    file_path = file_paths[grid_type]
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return None
    
    print(f"Loading {grid_type} grids from {file_path}...")
    
    # Use memory mapping for large files
    data = np.load(file_path, mmap_mode='r')
    print(f"  Original shape: {data.shape}")
    
    # Take only needed samples and convert to float32
    data_subset = np.array(data[:n_samples], dtype=np.float32)
    print(f"  Loaded shape: {data_subset.shape}")
    
    return data_subset


def normalize_grids_efficient(grids):
    """Normalize 3D grid data efficiently"""
    print(f"Normalizing grids with shape: {grids.shape}")
    
    # Normalize each channel independently
    for ch in range(grids.shape[1]):  # For each channel
        channel_data = grids[:, ch, :, :, :]
        
        # Compute stats
        mean_val = np.mean(channel_data)
        std_val = np.std(channel_data)
        
        if std_val > 1e-8:  # Avoid division by zero
            grids[:, ch, :, :, :] = (channel_data - mean_val) / std_val
    
    print(f"Normalization complete. Mean: {np.mean(grids):.4f}, Std: {np.std(grids):.4f}")
    return grids


def extract_traditional_features(grids, method='comprehensive'):
    """Extract traditional statistical features from 3D grid data"""
    if len(grids.shape) != 5:
        print(f"Warning: Expected 5D grids, got {grids.shape}")
        return grids.reshape(grids.shape[0], -1)
    
    n_samples, n_channels, d1, d2, d3 = grids.shape
    print(f"Extracting traditional features from {n_samples} samples with {n_channels} channels")
    
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
    print(f"Extracted traditional features shape: {features.shape}")
    
    # Remove features with zero variance
    feature_std = np.std(features, axis=0)
    valid_features = feature_std > 1e-8
    features = features[:, valid_features]
    print(f"After removing zero-variance features: {features.shape}")
    
    return features


def create_cnn_feature_extractor(input_shape):
    """Create a CNN model for feature extraction (not prediction)"""
    
    model = models.Sequential([
        # Downsample input first to reduce memory
        layers.AveragePooling3D((2, 2, 2), input_shape=input_shape, name='downsample'),
        
        # First conv block
        layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same', name='conv3d_1'),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Dropout(0.1),
        
        # Second conv block
        layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same', name='conv3d_2'),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Dropout(0.2),
        
        # Third conv block
        layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='conv3d_3'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling3D(),
        layers.Dropout(0.3),
        
        # Dense feature layers (no final prediction layer)
        layers.Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), name='features_128'),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu', name='features_64'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', name='features_32'),  # Final feature layer
    ])
    
    return model


def extract_cnn_features(grids, feature_extractor):
    """Extract features using trained CNN feature extractor"""
    print(f"Extracting CNN features from grids with shape: {grids.shape}")
    
    # Extract features using the CNN
    cnn_features = feature_extractor.predict(grids, batch_size=2, verbose=0)
    
    print(f"Extracted CNN features shape: {cnn_features.shape}")
    return cnn_features


def train_cnn_feature_extractor(X_train, y_train, X_val, y_val, input_shape):
    """Train CNN for feature extraction"""
    
    print("Training CNN feature extractor...")
    
    # Create feature extractor
    feature_extractor = create_cnn_feature_extractor(input_shape)
    
    # Add temporary prediction head for training
    temp_model = models.Sequential([
        feature_extractor,
        layers.Dense(1, name='temp_output')
    ])
    
    # Compile and train
    temp_model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Train with early stopping
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    callbacks_list = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]
    
    history = temp_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=4,
        callbacks=callbacks_list,
        verbose=1
    )
    
    print("CNN feature extractor training completed")
    
    # Return just the feature extractor (without prediction head)
    return feature_extractor, history


def combine_features(cnn_features_list, traditional_features_list, csv_features=None):
    """Combine CNN features, traditional features, and CSV features"""
    
    all_features = []
    feature_names = []
    
    # Add CNN features
    for i, (name, features) in enumerate(cnn_features_list):
        all_features.append(features)
        feature_names.extend([f"{name}_cnn_{j}" for j in range(features.shape[1])])
        print(f"Added {name} CNN features: {features.shape}")
    
    # Add traditional features
    for i, (name, features) in enumerate(traditional_features_list):
        all_features.append(features)
        feature_names.extend([f"{name}_trad_{j}" for j in range(features.shape[1])])
        print(f"Added {name} traditional features: {features.shape}")
    
    # Add CSV features if provided
    if csv_features is not None:
        all_features.append(csv_features)
        feature_names.extend([f"csv_{j}" for j in range(csv_features.shape[1])])
        print(f"Added CSV features: {csv_features.shape}")
    
    # Combine all features
    if all_features:
        X_combined = np.concatenate(all_features, axis=1)
        print(f"Combined features shape: {X_combined.shape}")
        return X_combined, feature_names
    else:
        return None, None


def train_hybrid_models(X_train, y_train, X_test, y_test, feature_names):
    """Train hybrid models with combined features"""
    
    print(f"\\nTraining hybrid models with {X_train.shape[1]} combined features...")
    
    # Feature selection
    print("Performing feature selection...")
    selector = SelectKBest(f_regression, k=min(200, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    print(f"Selected {X_train_selected.shape[1]} features")
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    
    models = {
        'Hybrid_RandomForest': RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Hybrid_ExtraTrees': ExtraTreesRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Hybrid_GradientBoosting': GradientBoostingRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ),
        'Hybrid_Ridge': Ridge(alpha=10.0),
        'Hybrid_ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"\\nTraining {name}...")
        start_time = time.time()
        
        try:
            # Use selected features
            X_tr, X_te = X_train_selected, X_test_selected
            
            if name in ['Hybrid_Ridge', 'Hybrid_ElasticNet']:
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
    
    return results, predictions, selected_feature_names


def main():
    print("Hybrid 3D CNN + Traditional Features Binding Affinity Prediction")
    print("=" * 70)
    
    # Load and parse CSV
    df_clean = load_and_parse_csv()
    if df_clean is None:
        return None
    
    # Use manageable dataset size
    max_samples = 150
    n_samples = min(max_samples, len(df_clean))
    print(f"\\nUsing {n_samples} samples for hybrid approach")
    
    # Get targets
    y = df_clean['final_dG'].values[:n_samples]
    
    # Load grid data
    grid_types = ['ligand', 'pocket']  # Focus on these two for memory efficiency
    grid_data = {}
    
    for grid_type in grid_types:
        grids = load_grid_data_efficiently(grid_type, n_samples)
        if grids is not None:
            grids = normalize_grids_efficient(grids)
            grid_data[grid_type] = grids
    
    if not grid_data:
        print("No grid data loaded!")
        return None
    
    print(f"\\nLoaded {len(grid_data)} grid types: {list(grid_data.keys())}")
    
    # Extract traditional features from all grids
    print("\\n" + "="*50)
    print("EXTRACTING TRADITIONAL FEATURES")
    print("="*50)
    
    traditional_features_list = []
    for name, grids in grid_data.items():
        trad_features = extract_traditional_features(grids, method='comprehensive')
        traditional_features_list.append((name, trad_features))
    
    # Split data for CNN training
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42)
    
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    
    # Extract CNN features
    print("\\n" + "="*50)
    print("EXTRACTING CNN FEATURES")
    print("="*50)
    
    cnn_features_list = []
    
    for name, grids in grid_data.items():
        print(f"\\nProcessing {name} grids for CNN feature extraction...")
        
        # Split grids
        X_train_grids = grids[train_idx]
        X_val_grids = grids[val_idx]
        X_test_grids = grids[test_idx]
        
        # Train CNN feature extractor
        input_shape = X_train_grids.shape[1:]
        feature_extractor, history = train_cnn_feature_extractor(
            X_train_grids, y_train, X_val_grids, y_val, input_shape
        )
        
        # Extract features for all data
        all_grids = grids
        cnn_features = extract_cnn_features(all_grids, feature_extractor)
        cnn_features_list.append((name, cnn_features))
        
        # Cleanup
        del feature_extractor
        gc.collect()
    
    # Prepare CSV features
    print("\\n" + "="*50)
    print("PREPARING CSV FEATURES")
    print("="*50)
    
    csv_features = []
    csv_feature_names = []
    
    for col in ['resolution', 'Ki_numeric', 'Kd_numeric', 'binding_constant_nM']:
        if col in df_clean.columns:
            values = df_clean[col].values[:n_samples]
            # Fill missing values with median
            median_val = np.nanmedian(values)
            values = np.where(np.isnan(values), median_val, values)
            csv_features.append(values.reshape(-1, 1))
            csv_feature_names.append(col)
    
    if csv_features:
        csv_X = np.concatenate(csv_features, axis=1)
        print(f"CSV features shape: {csv_X.shape}")
    else:
        csv_X = None
    
    # Combine all features
    print("\\n" + "="*50)
    print("COMBINING ALL FEATURES")
    print("="*50)
    
    X_combined, feature_names = combine_features(
        cnn_features_list, traditional_features_list, csv_X
    )
    
    if X_combined is None:
        print("Failed to combine features!")
        return None
    
    # Split combined features
    X_train_combined = X_combined[train_idx]
    X_test_combined = X_combined[test_idx]
    
    print(f"\\nFinal combined dataset:")
    print(f"  Train samples: {X_train_combined.shape[0]}")
    print(f"  Test samples: {X_test_combined.shape[0]}")
    print(f"  Total features: {X_combined.shape[1]}")
    print(f"    - CNN features: {sum([features.shape[1] for _, features in cnn_features_list])}")
    print(f"    - Traditional features: {sum([features.shape[1] for _, features in traditional_features_list])}")
    print(f"    - CSV features: {csv_X.shape[1] if csv_X is not None else 0}")
    
    # Train hybrid models
    print("\\n" + "="*50)
    print("TRAINING HYBRID MODELS")
    print("="*50)
    
    results, predictions, selected_features = train_hybrid_models(
        X_train_combined, y_train, X_test_combined, y_test, feature_names
    )
    
    if not results:
        print("No hybrid models trained successfully!")
        return None
    
    # Create ensemble
    print("\\nCreating ensemble...")
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
    print("\\n" + "="*70)
    print("HYBRID MODEL RESULTS SUMMARY")
    print("="*70)
    
    for name, metrics in results.items():
        print(f"{name:25} - Test R²: {metrics['test_r2']:7.4f}, CV R²: {metrics['cv_r2_mean']:7.4f} ± {metrics['cv_r2_std']:.3f}")
    
    print(f"{'Hybrid_Ensemble':25} - Test R²: {ensemble_r2:7.4f}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_r2 = results[best_model]['test_r2']
    
    print(f"\\n Best hybrid model: {best_model} (R² = {best_r2:.4f})")
    print(f" Hybrid Ensemble R²: {ensemble_r2:.4f}")
    
    # Comparison with previous results
    print(f"\\n PERFORMANCE COMPARISON:")
    print(f"   Traditional ML (Gradient Boosting): R² = 0.9963")
    print(f"   Pure 3D CNN (Best):                 R² = 0.0013")
    print(f"   Hybrid Model (Best):                R² = {best_r2:.4f}")
    print(f"   Hybrid Ensemble:                    R² = {ensemble_r2:.4f}")
    
    # Determine improvement
    baseline_traditional = 0.9963
    baseline_cnn = 0.0013
   
    
    # Visualization
    plt.figure(figsize=(16, 10))
    
    # Model comparison
    plt.subplot(2, 4, 1)
    model_names = ['Traditional\\nML', 'Pure\\n3D CNN', 'Hybrid\\nBest', 'Hybrid\\nEnsemble']
    r2_scores = [baseline_traditional, baseline_cnn, best_r2, ensemble_r2]
    colors = ['gold', 'red', 'green', 'blue']
    
    bars = plt.bar(model_names, r2_scores, color=colors, alpha=0.7)
    plt.ylabel('R² Score')
    plt.title('Model Performance Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Best hybrid model predictions
    plt.subplot(2, 4, 2)
    y_pred_best = predictions[best_model]
    plt.scatter(y_test, y_pred_best, alpha=0.7, color='green', s=50)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual ΔG (kcal/mol)')
    plt.ylabel('Predicted ΔG (kcal/mol)')
    plt.title(f'Best Hybrid: {best_model}\\nR² = {best_r2:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Ensemble predictions
    plt.subplot(2, 4, 3)
    plt.scatter(y_test, ensemble_pred, alpha=0.7, color='blue', s=50)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual ΔG (kcal/mol)')
    plt.ylabel('Predicted ΔG (kcal/mol)')
    plt.title(f'Hybrid Ensemble\\nR² = {ensemble_r2:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Feature type importance
    plt.subplot(2, 4, 4)
    feature_types = ['CNN', 'Traditional', 'CSV']
    cnn_count = sum([features.shape[1] for _, features in cnn_features_list])
    trad_count = sum([features.shape[1] for _, features in traditional_features_list])
    csv_count = csv_X.shape[1] if csv_X is not None else 0
    
    counts = [cnn_count, trad_count, csv_count]
    plt.pie(counts, labels=feature_types, autopct='%1.1f%%', startangle=90)
    plt.title('Feature Type Distribution')
    
    # Hybrid model comparison
    plt.subplot(2, 4, 5)
    hybrid_names = list(results.keys())
    hybrid_r2s = [results[name]['test_r2'] for name in hybrid_names]
    
    bars = plt.bar(hybrid_names, hybrid_r2s, alpha=0.7, color='lightgreen')
    plt.ylabel('Test R²')
    plt.title('Hybrid Model Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Residual analysis for best model
    plt.subplot(2, 4, 6)
    residuals = y_test - y_pred_best
    plt.scatter(y_pred_best, residuals, alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('Predicted ΔG (kcal/mol)')
    plt.ylabel('Residuals (kcal/mol)')
    plt.title('Residual Analysis')
    plt.grid(True, alpha=0.3)
    
    # Feature importance (if available)
    plt.subplot(2, 4, 7)
    if hasattr(results[best_model], 'feature_importances_'):
        # This won't work for all models, but let's try for tree-based ones
        try:
            # Get the trained model (this is a simplification)
            plt.text(0.5, 0.5, f'Feature Importance\\nanalysis available\\nfor tree-based models', 
                    transform=plt.gca().transAxes, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        except:
            pass
    plt.title('Feature Analysis')
    plt.axis('off')
    
    # Summary statistics
    plt.subplot(2, 4, 8)
    summary_text = f"""Hybrid Model Summary:
    
Best Model: {best_model.replace('Hybrid_', '')}
Test R²: {best_r2:.4f}
RMSE: {results[best_model]['test_rmse']:.3f}

Ensemble R²: {ensemble_r2:.4f}

Dataset: {n_samples} samples
Features: {X_combined.shape[1]} total
- CNN: {cnn_count}
- Traditional: {trad_count}  
- CSV: {csv_count}

Selected: {len(selected_features)} features"""
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hybrid_cnn_traditional_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\\n Hybrid model training completed!")
    print(f"Results saved to: hybrid_cnn_traditional_results.png")
    
    return results, (X_combined, y, ensemble_pred)


if __name__ == "__main__":
    try:
        results, data = main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
