import os
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path

class MRLEyeDataset(Dataset):
    """
    PyTorch Dataset for MRL Eye state classification.
    """
    def __init__(self, split_dir, transform=None):
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.samples = []
        
        # Crawl through subject directories
        for sub_dir in self.split_dir.iterdir():
            if sub_dir.is_dir():
                for img_path in sub_dir.glob("*.png"):
                    # Filename: subjectID_imageID_gender_glasses_eyeState_reflections_lightingConditions_sensorID.png
                    # Eye state is index 4 (0: closed, 1: open)
                    label = int(img_path.stem.split("_")[4])
                    self.samples.append((img_path, label))
                    
        print(f"Loaded {len(self.samples)} samples from {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # Read as grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        # Resize to 32x32 for EyeStateNet
        img = cv2.resize(img, (32, 32))
        
        # Normalize 0-1
        img = img.astype("float32") / 255.0
        img = torch.from_numpy(img).unsqueeze(0) # (1, 32, 32)
        
        return img, label

if __name__ == "__main__":
    # Test dataset
    ds = MRLEyeDataset("data/processed/mrl_eye/train")
    if len(ds) > 0:
        img, lbl = ds[0]
        print(f"Sample shape: {img.shape}, Label: {lbl}")
