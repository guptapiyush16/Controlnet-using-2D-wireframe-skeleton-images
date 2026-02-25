import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob

class ControlNetDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Assume data_dir contains matches like {id}_rgb.png and {id}_wire.png
        self.rgb_files = sorted(glob.glob(os.path.join(data_dir, "*_rgb.png")))
        self.wire_files = sorted(glob.glob(os.path.join(data_dir, "*_wire.png")))
        
        assert len(self.rgb_files) == len(self.wire_files), "Mismatch between RGB and wireframe images"

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        wire_path = self.wire_files[idx]
        
        rgb_image = Image.open(rgb_path).convert("RGB")
        wire_image = Image.open(wire_path).convert("RGB") # ControlNet expects RGB condition images
        
        if self.transform:
            # Note: Ensure transform is applied consistently to both if needed (e.g., random flips)
            # For now, simple resize and tensor conversion
            pass
            
        return {
            "pixel_values": self._preprocess_image(rgb_image),
            "conditioning_pixel_values": self._preprocess_image(wire_image),
            "text": "a 3D object rendering" # In real use, fetch prompt from Objaverse metadata
        }

    def _preprocess_image(self, image, size=512):
        image = image.resize((size, size), resample=Image.BICUBIC)
        # Convert to tensor and normalize to [-1, 1]
        img_arr = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
        return img_arr.permute(2, 0, 1)

import numpy as np # Added for _preprocess_image
