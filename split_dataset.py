import os
import shutil
import random
from tqdm import tqdm

# Define paths
input_folder = "/"
output_folder = "/"

# Define categories
categories = ["Normal", "Osteopenia", "Osteoporosis"]

# Define split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create new directories for train, valid, and test
for split in ["train", "valid", "test"]:
    for category in categories:
        os.makedirs(os.path.join(output_folder, split, category), exist_ok=True)

# Function to split images
def split_data():
    for category in categories:
        image_files = os.listdir(os.path.join(input_folder, category))
        random.shuffle(image_files)  # Shuffle data
        
        # Calculate split sizes
        total_images = len(image_files)
        train_size = int(total_images * train_ratio)
        val_size = int(total_images * val_ratio)

        # Assign images to splits
        train_images = image_files[:train_size]
        val_images = image_files[train_size:train_size + val_size]
        test_images = image_files[train_size + val_size:]

        # Move images to respective folders
        for img_set, split in zip([train_images, val_images, test_images], ["train", "valid", "test"]):
            for img in tqdm(img_set, desc=f"Processing {category} - {split}"):
                src_path = os.path.join(input_folder, category, img)
                dest_path = os.path.join(output_folder, split, category, img)
                shutil.copy(src_path, dest_path)

# Run the split function
split_data()
print("✅ Dataset successfully split into Train, Validation, and Test sets!")