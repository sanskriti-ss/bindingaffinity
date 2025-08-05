# 🔗 Integration Guide: How to Use protein_data_reader.py with Your Existing Data

## Step 1: Copy the protein_data_reader.py to your project directory

The `protein_data_reader.py` file I created provides two main classes:

1. **`ProteinGridDataset`** - Loads data from your .npy files and metadata
2. **`SimpleProteinDataset`** - Uses your existing matched arrays (recommended)

## Step 2: Import and use in your notebook

Add this cell to your notebook:

```python
# Import the data reader
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'qnn-ingenni-explore', 'data'))

from protein_data_reader import SimpleProteinDataset, create_data_loaders

# Create dataset from your existing matched arrays
dataset = SimpleProteinDataset(
    matched_proteins=matched_proteins,      # Your existing variable
    matched_ligands=matched_ligands,        # Your existing variable  
    matched_pockets=matched_pockets,        # Your existing variable
    binding_energies=binding_labels,        # Your existing variable
    mol_ids=common_ids_list,               # Your existing variable
    normalize=True
)

# Create train/validation data loaders
train_loader, val_loader = create_data_loaders(
    dataset, 
    batch_size=8,           # Adjust based on your GPU memory
    train_split=0.8,        # 80% train, 20% validation
    random_seed=42
)

print(f"✅ Dataset: {len(dataset)} samples")
print(f"📚 Training batches: {len(train_loader)}")
print(f"🔍 Validation batches: {len(val_loader)}")

# Get normalization parameters for later use
norm_params = dataset.get_normalization_params()
print(f"🎯 Binding energy normalization: mean={norm_params['binding_mean']:.3f}, std={norm_params['binding_std']:.3f}")
```

## Step 3: Modify your training loop

Replace your current training code with this:

```python
def train_with_dataloader(model, train_loader, val_loader, num_epochs=20):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            # Get data
            protein = batch['protein']
            ligand = batch['ligand'] 
            pocket = batch['pocket']
            target = batch['binding_energy']
            
            # Forward pass (adjust based on your model)
            optimizer.zero_grad()
            
            # For your model that takes separate inputs:
            output = model(protein, ligand, pocket)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation 
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                protein = batch['protein']
                ligand = batch['ligand']
                pocket = batch['pocket'] 
                target = batch['binding_energy']
                
                output = model(protein, ligand, pocket)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val Loss={val_loss/len(val_loader):.4f}")

# Use it
train_with_dataloader(your_model, train_loader, val_loader)
```

## Step 4: Make predictions

```python
def make_predictions(model, data_loader, norm_params):
    model.eval()
    predictions = []
    targets = []
    mol_ids = []
    
    with torch.no_grad():
        for batch in data_loader:
            output = model(batch['protein'], batch['ligand'], batch['pocket'])
            
            predictions.extend(output.numpy().flatten())
            targets.extend(batch['binding_energy'].numpy().flatten()) 
            mol_ids.extend(batch['mol_id'])
    
    # Denormalize predictions
    predictions = np.array(predictions) * norm_params['binding_std'] + norm_params['binding_mean']
    targets = np.array(targets) * norm_params['binding_std'] + norm_params['binding_mean']
    
    return predictions, targets, mol_ids

# Use it
pred, actual, ids = make_predictions(your_model, val_loader, norm_params)

# Create results DataFrame
results_df = pd.DataFrame({
    'Complex_ID': ids,
    'Actual_Binding_Energy': actual,
    'Predicted_Binding_Energy': pred
})
```

## Benefits of using the data reader:

 **Proper train/validation splits** - No data leakage  
 **Batching** - Memory efficient training  
 **Normalization** - Better model convergence  
 **Reproducible** - Fixed random seeds  
 **Flexible** - Easy to modify batch sizes, splits, etc.  
 **Compatible** - Works with your existing data structures

## Key differences from your current approach:

1. **Batched processing** instead of processing all data at once
2. **Automatic normalization** of binding energies
3. **Proper train/val split** that respects molecular complexes
4. **Memory efficient** - only loads what's needed per batch
5. **Standardized format** - easier to experiment with different models

