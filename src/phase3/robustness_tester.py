import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Common Corruptions
def adjust_brightness(img, factor=0.5):
    """Adjust brightness by a factor (e.g. 0.5 for darker, 1.5 for brighter)."""
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def add_gaussian_noise(img, sigma=25):
    """Add Gaussian noise to simulate poor sensor quality in low light."""
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def add_occlusion(img, max_ratio=0.3):
    """Add a black box to simulate occlusion (e.g., thick glasses frames)."""
    h, w = img.shape
    box_w = int(w * max_ratio)
    box_h = int(h * max_ratio)
    
    x = np.random.randint(0, w - box_w)
    y = np.random.randint(0, h - box_h)
    
    img_occ = img.copy()
    img_occ[y:y+box_h, x:x+box_w] = 0
    return img_occ

class RobustnessTester:
    def __init__(self, model_path="models/eye_state_best.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        """Preprocess an already cropped eye image for the model."""
        img = cv2.resize(img, (32, 32))
        tensor = img.astype("float32") / 255.0
        tensor = np.expand_dims(np.expand_dims(tensor, axis=0), axis=0) # (1, 1, 32, 32)
        return tensor

    def predict(self, img):
        tensor = self.preprocess(img)
        out = self.session.run(None, {self.input_name: tensor})[0]
        
        # Softmax for probabilities
        exp_logits = np.exp(out[0] - np.max(out[0]))
        probs = exp_logits / exp_logits.sum()
        
        # Output [Closed, Open]
        return 0 if probs[0] > 0.5 else 1

    def evaluate(self, dataset_dir, max_samples=1000):
        dataset_dir = Path(dataset_dir)
        samples = []
        for sub_dir in dataset_dir.iterdir():
            if sub_dir.is_dir():
                for img_path in sub_dir.glob("*.png"):
                    label = int(img_path.stem.split("_")[4])
                    samples.append((str(img_path), label))
                    
        # Shuffle explicitly and limit samples for quick eval
        np.random.shuffle(samples)
        if max_samples and len(samples) > max_samples:
            samples = samples[:max_samples]
            
        print(f"Evaluating on {len(samples)} samples...")
        
        corruptions = {
            "Baseline (Clean)": lambda img: img,
            "Darker (x0.3)": lambda img: adjust_brightness(img, factor=0.3),
            "Brighter (x2.0)": lambda img: adjust_brightness(img, factor=2.0),
            "Gaussian Noise": lambda img: add_gaussian_noise(img),
            "Random Occlusion": lambda img: add_occlusion(img)
        }
        
        results = {}
        for name in corruptions.keys():
            results[name] = {"correct": 0, "total": len(samples)}
            
        for img_path, label in samples:
            base_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            for c_name, c_fn in corruptions.items():
                corrupted_img = c_fn(base_img)
                pred = self.predict(corrupted_img)
                if pred == label:
                    results[c_name]["correct"] += 1
                    
        print("\n=== Robustness Evaluation Results ===")
        for c_name, stats in results.items():
            acc = stats["correct"] / stats["total"] * 100
            print(f"{c_name:25s}: {acc:.2f}% Accuracy")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="data/processed/mrl_eye/test", help="Path to test split")
    parser.add_argument("--model", default="models/eye_state_best.onnx", help="Path to ONNX model")
    args = parser.parse_args()
    
    # Need to run from repository root explicitly
    model_path = os.path.abspath(args.model)
    test_dir = os.path.abspath(args.test_dir)
    
    tester = RobustnessTester(model_path)
    tester.evaluate(test_dir, max_samples=2000)
