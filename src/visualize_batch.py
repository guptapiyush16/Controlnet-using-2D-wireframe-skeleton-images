import torch
from data.dataset import ControlNetDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_dataset_batch(data_dir="data/processed", output_path="output/batch_viz.png"):
    """
    Loads a batch from the dataset and saves a visualization of inputs.
    """
    dataset = ControlNetDataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    batch = next(iter(dataloader))
    
    rgb = batch["pixel_values"] # (B, 3, H, W)
    wire = batch["conditioning_pixel_values"] # (B, 3, H, W)
    
    # Denormalize
    rgb = (rgb + 1.0) * 127.5
    wire = (wire + 1.0) * 127.5
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(min(4, len(rgb))):
        # RGB Input
        axes[0, i].imshow(rgb[i].permute(1, 2, 0).numpy().astype(np.uint8))
        axes[0, i].set_title(f"Target RGB {i}")
        axes[0, i].axis("off")
        
        # Wireframe Condition
        axes[1, i].imshow(wire[i].permute(1, 2, 0).numpy().astype(np.uint8))
        axes[1, i].set_title(f"Wireframe Input {i}")
        axes[1, i].axis("off")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Batch visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_dataset_batch()
