import os
import shutil
import random

# Paths
dataset_path = r"C:\Users\hp\Downloads\archive"       # Path to your 30,000 images dataset
output_path  = r"C:\Users\hp\Documents\shapes_for_project"    # Path where 50 images per shape will be saved

# Shape folders to process
shapes = ["triangle", "square", "circle"]

# Number of images to copy per shape
num_images = 50

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

for shape in shapes:
    # Paths for input and output
    input_folder = os.path.join(dataset_path, shape)
    output_folder = os.path.join(output_path, shape)
    
    # Make sure output subfolder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files from dataset folder
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    # Randomly choose 50 files (or fewer if dataset has less than 50)
    selected_files = random.sample(files, min(num_images, len(files)))
    
    # Copy and rename selected files
    for idx, file in enumerate(selected_files, start=1):
        src = os.path.join(input_folder, file)
        
        # Create new filename like shape_01.png
        ext = os.path.splitext(file)[1]  # keep original extension (.png/.jpg)
        new_name = f"{shape}_{idx:02d}{ext}"
        
        dst = os.path.join(output_folder, new_name)
        shutil.copy(src, dst)
    
    print(f"Copied {len(selected_files)} {shape} images to {output_folder}")

print("✅ All shapes copied and renamed successfully!")
