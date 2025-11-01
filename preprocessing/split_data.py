import os
import shutil
import random
from tqdm import tqdm

def split_dataset(base_dir, output_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    random.seed(seed)
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Found {len(classes)} classes:", classes)
    
    for cls in tqdm(classes, desc="Splitting dataset"):
        cls_dir = os.path.join(base_dir, cls)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }
        
        for split_name, split_images in splits.items():
            split_folder = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_folder, exist_ok=True)
            for img_name in split_images:
                src = os.path.join(cls_dir, img_name)
                dst = os.path.join(split_folder, img_name)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
    
    print("\nDone! Dataset split complete.")
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            print(f"{split.capitalize()} folder created at: {split_dir}")

# Example usage
split_dataset(
    base_dir="./data/train",  # your original Kaggle dataset folder
    output_dir="./data",      # output destination for train/val/test
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2
)