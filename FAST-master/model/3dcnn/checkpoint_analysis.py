"""
AI generated Script to analyze the performance of a saved model.
"""


def generate_predictions(model, dataloader, device):
    """
    Generate predictions using a saved model.
    
    Args:
        model: Loaded PyTorch model
        dataloader: DataLoader containing the dataset
        device: Device to run inference on
        
    Returns:
        Tuple of (ytrue_arr, ypred_arr)
    """
    model.eval()
    batch_size = dataloader.batch_size
    dataset_size = len(dataloader.dataset)
    
    # Pre-allocate arrays
    ytrue_arr = np.zeros((dataset_size,), dtype=np.float32)
    ypred_arr = np.zeros((dataset_size,), dtype=np.float32)
    
    with torch.no_grad():
        for bind, batch in enumerate(dataloader):
            # Extract data
            pdb_id_batch, x_batch_cpu, y_batch_cpu = batch
            x_batch = x_batch_cpu.to(device)
            
            # Process data (voxelization, etc.)
            vol_batch = torch.zeros((batch_size, 19, 48, 48, 48)).float().to(device)
            bsize = x_batch.shape[0]
            
            for i in range(bsize):
                xyz, feat = x_batch[i,:,:3], x_batch[i,:,3:]
                vol_batch[i,:,:,:,:] = voxelizer(xyz, feat)
            vol_batch = gaussian_filter(vol_batch)
            
            # Forward pass
            ypred_batch, _ = model(vol_batch[:x_batch.shape[0]])
            
            # Extract true and predicted values
            ytrue = y_batch_cpu.float().numpy()[:,0]
            ypred = ypred_batch.detach().cpu().float().numpy()[:,0]
            
            # Store in pre-allocated arrays
            start_idx = bind * batch_size
            end_idx = start_idx + bsize
            ytrue_arr[start_idx:end_idx] = ytrue
            ypred_arr[start_idx:end_idx] = ypred
    
    return ytrue_arr, ypred_arr


def analyze_model_performance(checkpoint_path, dataset, device='cuda'):
    # Load dataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load model
    model = YourModelClass()  # Initialize your model architecture
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Generate predictions
    print(f"Regenerating predictions for checkpoint from epoch {checkpoint['epoch']}...")
    ytrue_arr, ypred_arr = generate_predictions(model, dataloader, device)
    
    # Compute metrics
    metrics = compute_metrics(ytrue_arr, ypred_arr)
    print(f"Regenerated metrics: {metrics}")
    
    # Verify against stored metrics
    stored_metrics = checkpoint['metrics']
    print(f"Stored metrics: {stored_metrics}")
    
    # Additional analysis
    plot_prediction_scatter(ytrue_arr, ypred_arr)
    plot_error_distribution(ytrue_arr, ypred_arr)
    
    return ytrue_arr, ypred_arr, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze model predictions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    args = parser.parse_args()
    
    # Load dataset
    dataset = YourDatasetClass(args.dataset)
    
    # Analyze model
    ytrue, ypred, metrics = analyze_model_performance(args.checkpoint, dataset, args.device)
    
    # Save regenerated predictions if needed
    np.savez_compressed(
        args.checkpoint.replace('.pth', '_regenerated_predictions.npz'),
        ytrue=ytrue,
        ypred=ypred
    )