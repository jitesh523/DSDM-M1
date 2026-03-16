import os
import shutil
import random
import logging
import numpy as np
from pathlib import Path

class SubjectSplitter:
    """
    Ensures subject-based splitting for datasets to prevent data leakage.
    """
    def __init__(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        self.ratios = (train_ratio, val_ratio, test_ratio)
        self.logger = logging.getLogger(__name__)

    def split_subjects(self, subjects):
        """
        Splits a list of subject IDs into train, val, and test sets.
        """
        random.shuffle(subjects)
        n = len(subjects)
        train_end = int(n * self.ratios[0])
        val_end = train_end + int(n * self.ratios[1])
        
        train_subs = subjects[:train_end]
        val_subs = subjects[train_end:val_end]
        test_subs = subjects[val_end:]
        
        self.logger.info(f"Split {n} subjects: Train={len(train_subs)}, Val={len(val_subs)}, Test={len(test_subs)}")
        return train_subs, val_subs, test_subs

class DatasetManager:
    """
    Handles preprocessing and directory management for Phase 2 training data.
    """
    def __init__(self, base_path="data/datasets", processed_path="data/processed"):
        self.base_path = Path(base_path)
        self.processed_path = Path(processed_path)
        self.logger = logging.getLogger(__name__)
        
    def prepare_mrl_eye(self):
        """
        Prepares MRL Eye dataset for training.
        MRL Filename structure: sXXXX_YYYYY_S_K_L_G_E_C.png
        S: subject ID (s0001 - s0037)
        C: class (0=closed, 1=open)
        """
        mrl_dir = self.base_path / "mrl_eye" / "extracted" / "mrlEyes_2018_01"
        if not mrl_dir.exists():
            self.logger.warning(f"MRL Eye directory not found at {mrl_dir}")
            return False
            
        # Get all subdirectories (subjects)
        subjects = [d.name for d in mrl_dir.iterdir() if d.is_dir() and d.name.startswith("s")]
        splitter = SubjectSplitter()
        train_s, val_s, test_s = splitter.split_subjects(subjects)
        
        self._organize_split("mrl_eye", mrl_dir, train_s, val_s, test_s)
        return True

    def _organize_split(self, name, source_dir, train_s, val_s, test_s):
        """
        Creates symlinks or copies files into structured train/val/test directories.
        """
        dest_base = self.processed_path / name
        os.makedirs(dest_base, exist_ok=True)
        
        splits = {"train": train_s, "val": val_s, "test": test_s}
        
        for split_name, subs in splits.items():
            split_dir = dest_base / split_name
            os.makedirs(split_dir, exist_ok=True)
            for sub in subs:
                sub_src = source_dir / sub
                sub_dst = split_dir / sub
                if not sub_dst.exists():
                    os.symlink(sub_src.absolute(), sub_dst)
                    
        self.logger.info(f"Dataset {name} organized into splits at {dest_base}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = DatasetManager()
    manager.prepare_mrl_eye()
