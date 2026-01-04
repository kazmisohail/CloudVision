"""
**Final Year Project: Cloud Detection, Haze and Shadow Mitigation, and Cloud Classification**

Notebook 1: Data Preprocessing

Purpose: To take the raw L8CCA dataset, extract it, and process it into small image and mask patches for model training.

## Step 1: Setup and Configuration
"""

#pip install rasterio --quiet
#pip install tqdm --quiet

import os
import tarfile
import rasterio
from rasterio.windows import Window
import shutil
import numpy as np
import random
import re
from tqdm.notebook import tqdm

print("Libraries imported successfully...!!")

"""## Step 2: Define Project Paths and Configuration"""

PROJECT_DIR = '/content/drive/MyDrive/Final_Year_Project' # fyp k main directory
RAW_DATA_DIR = os.path.join(PROJECT_DIR, 'L8CCA_Dataset') # dataset yha saved h with all 8 biomes
PREPROCESSED_DIR = os.path.join(PROJECT_DIR, 'Preprocessed_Data') # preprocessed data ko yha save kry gy

# configuration for patch creation
PATCH_SIZE = 256
# using Blue, Green, Red, NIR, SWIR 1, SWIR 2, and the two Thermal bands
# this provides the model with spectral, vegetative, and thermal information
BANDS_TO_USE = [2, 3, 4, 5, 6, 7, 10, 11]
MIN_VAL = 0
MAX_VAL = 40000
TILE_SIZE = 1024
print("Configuration set...!!")
print(f"Using Bands: {BANDS_TO_USE}")
print(f"Patch Size: {PATCH_SIZE}x{PATCH_SIZE}")

"""## Step 3: Extract Raw Data from .tar Files"""

def is_tar_gz_valid(tar_path):
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            for member in tar.getmembers():
                pass
        return True
    except (tarfile.ReadError, EOFError, tarfile.InvalidHeaderError):
        return False

print("Starting TAR file extraction with pre-check and cleanup...")

RAW_DATA_DIR = '/content/drive/MyDrive/Final_Year_Project/L8CCA_Dataset'
biome_folders = [f.path for f in os.scandir(RAW_DATA_DIR) if f.is_dir()]

for biome_path in biome_folders:
    print(f"\nChecking biome: {os.path.basename(biome_path)}")
    for item in os.listdir(biome_path):
        if item.endswith(".tar.gz"):
            tar_path = os.path.join(biome_path, item)
            extracted_folder_path = tar_path.replace('.tar.gz', '')

            if os.path.exists(extracted_folder_path) and os.listdir(extracted_folder_path):
                print(f"  Skipping '{item}', folder already exists and is not empty.")
                continue

            print(f"  🔎 Pre-checking '{item}'...")
            if not is_tar_gz_valid(tar_path):
                print(f"  ERROR: '{item}' is corrupted. Please re-download it.")
                if os.path.exists(extracted_folder_path):
                    shutil.rmtree(extracted_folder_path)
                continue

            print(f"  Pre-check passed. Attempting to extract...")

            try:
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(path=biome_path)
                print(f"  Successfully extracted '{item}'.")
            except Exception as e:
                print(f"  ERROR: Extraction failed for '{item}' even after pre-check.")
                print(f"     Details: {e}")
                if os.path.exists(extracted_folder_path):
                    print(f"     Cleaning up partially extracted folder...")
                    shutil.rmtree(extracted_folder_path)

print("\n--- TAR file extraction process complete! ---")

# Diagnostic Code
# Folder extraction and location wagera check kr rh h...!!

import os
RAW_DATA_DIR = '/content/drive/MyDrive/Final_Year_Project/L8CCA_Dataset'

print("--- Running Directory Diagnosis ---")
print(f"Checking inside: {RAW_DATA_DIR}\n")

if not os.path.exists(RAW_DATA_DIR):
    print(f"ERROR: The main data directory does not exist: {RAW_DATA_DIR}")
else:
    biome_folders = [f.path for f in os.scandir(RAW_DATA_DIR) if f.is_dir()]
    if not biome_folders:
        print("No biome folders (like 'barren', 'forest') found inside L8CCA_Raw_Data/")
    else:
        for biome_path in biome_folders:
            print(f"=========================================")
            print(f"[+] Contents of folder: {os.path.basename(biome_path)}")
            print(f"=========================================")
            try:
                contents = os.listdir(biome_path)
                if not contents:
                    print("    -> This folder is empty.")
                else:
                    for item in sorted(contents): # Sort for consistent order
                        full_item_path = os.path.join(biome_path, item)
                        if os.path.isdir(full_item_path):
                            print(f"    -> Found DIRECTORY: {item}")
                        else:
                            print(f"    -> Found FILE:      {item}")
            except Exception as e:
                print(f"    -> Could not read contents of this folder. Error: {e}")


print("\n--- Diagnosis Complete ---")

"""## Step 4: Helper Functions for Patch Generation"""

def get_scene_files(scene_path):
    band_dict = {}
    mask_file = None
    for filename in os.listdir(scene_path):
        full_path = os.path.join(scene_path, filename)
        band_match = re.search(r'_B(\d{1,2})\.TIF$', filename, re.IGNORECASE)
        if band_match:
            band_num = int(band_match.group(1))
            band_dict[band_num] = full_path
        elif filename.endswith('_fixedmask.img'):
            mask_file = full_path
    return band_dict, mask_file

def create_patches_from_tile(scene_name, tile_coords, multispectral_tile, mask_tile, patch_size, output_dir):
    height, width, _ = multispectral_tile.shape
    patch_count = 0

    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            img_patch = multispectral_tile[y:y+patch_size, x:x+patch_size, :]
            mask_patch_original = mask_tile[y:y+patch_size, x:x+patch_size]

            if np.sum(mask_patch_original == 0) / (patch_size * patch_size) > 0.90:
                continue

            remapped_mask = np.zeros_like(mask_patch_original, dtype=np.uint8)
            mapping = {0: 0, 64: 2, 128: 1, 192: 3, 255: 4}
            for original_value, new_value in mapping.items():
                remapped_mask[mask_patch_original == original_value] = new_value

            abs_y, abs_x = tile_coords[0] + y, tile_coords[1] + x
            patch_name = f"{scene_name}_{abs_y}_{abs_x}"
            np.save(os.path.join(output_dir, 'images', f"{patch_name}.npy"), img_patch)
            np.save(os.path.join(output_dir, 'masks', f"{patch_name}.npy"), remapped_mask)
            patch_count += 1
    return patch_count

def process_scene_with_tiling(scene_path, bands_to_use, tile_size, patch_size, output_dir):
    scene_name = os.path.basename(scene_path)
    band_dict, mask_file = get_scene_files(scene_path)

    if not band_dict or not mask_file:
        print(f"  [Warning] Skipping {scene_name}: Missing required files.")
        return 0

    try:
        band_srcs = [rasterio.open(band_dict[b]) for b in bands_to_use]
        mask_src = rasterio.open(mask_file)

        width, height = mask_src.width, mask_src.height
        total_patches_in_scene = 0

        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                window = Window(x, y, min(tile_size, width - x), min(tile_size, height - y))

                band_tiles = [src.read(1, window=window) for src in band_srcs]
                mask_tile = mask_src.read(1, window=window)

                stacked_tile = np.stack(band_tiles, axis=-1).astype(np.float32)
                stacked_tile = np.clip(stacked_tile, MIN_VAL, MAX_VAL)
                normalized_tile = (stacked_tile - MIN_VAL) / (MAX_VAL - MIN_VAL)

                patch_count = create_patches_from_tile(scene_name, (y, x), normalized_tile, mask_tile, patch_size, output_dir)
                total_patches_in_scene += patch_count

        for src in band_srcs: src.close()
        mask_src.close()

        return total_patches_in_scene
    except Exception as e:
        print(f"  [ERROR] Failed to process {scene_name}. Reason: {e}")
        return 0

print("New memory-efficient helper functions are ready.")

"""## Step 5: Main Processing Script"""

print("--- Preparing Scene Lists for Processing ---")

all_scenes = []
biome_folders = [f.path for f in os.scandir(RAW_DATA_DIR) if f.is_dir()]
for biome_path in biome_folders:
    bc_folder_path = os.path.join(biome_path, 'BC')
    if os.path.isdir(bc_folder_path):
        scenes_in_bc = [f.path for f in os.scandir(bc_folder_path) if f.is_dir() and f.name.startswith('LC')]
        all_scenes.extend(scenes_in_bc)

if not all_scenes:
    raise Exception("CRITICAL ERROR: No scene folders found inside any 'BC' folders.")

random.seed(42)
random.shuffle(all_scenes)
train_split_idx = int(0.7 * len(all_scenes))
val_split_idx = int(0.85 * len(all_scenes))
train_scenes = all_scenes[:train_split_idx]
val_scenes = all_scenes[train_split_idx:val_split_idx]
test_scenes = all_scenes[val_split_idx:]

print(f"  -> Found {len(all_scenes)} scenes. Split into: {len(train_scenes)} train, {len(val_scenes)} val, {len(test_scenes)} test.")
print("\nSetup complete. You can now run the processing snippets.")

print("\n--- Processing VALIDATION Set (Memory-Safe) ---")
split = 'validation'
output_dir = os.path.join(PREPROCESSED_DIR, split)
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

total_val_patches = 0
for scene_path in tqdm(val_scenes, desc=f"Processing {split} scenes"):
    patches_created = process_scene_with_tiling(scene_path, BANDS_TO_USE, TILE_SIZE, PATCH_SIZE, output_dir)
    total_val_patches += patches_created

print(f"\nVALIDATION SET COMPLETE! Generated {total_val_patches} patches.")

print("\n--- Processing TRAINING Set (Memory-Safe) ---")
split = 'train'
output_dir = os.path.join(PREPROCESSED_DIR, split)
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

total_train_patches = 0
for scene_path in tqdm(train_scenes, desc=f"Processing {split} scenes"):
    patches_created = process_scene_with_tiling(scene_path, BANDS_TO_USE, TILE_SIZE, PATCH_SIZE, output_dir)
    total_train_patches += patches_created

print(f"\nTRAINING SET COMPLETE! Generated {total_train_patches} patches.")

print("\n--- Processing TRAINING Set (Memory-Safe) ---")
split = 'train'
output_dir = os.path.join(PREPROCESSED_DIR, split)
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

total_train_patches = 0
for scene_path in tqdm(train_scenes, desc=f"Processing {split} scenes"):
    patches_created = process_scene_with_tiling(scene_path, BANDS_TO_USE, TILE_SIZE, PATCH_SIZE, output_dir)
    total_train_patches += patches_created

print(f"\nTRAINING SET COMPLETE! Generated {total_train_patches} patches.")

print("\n--- Processing TEST Set (Memory-Safe) ---")
split = 'test'
output_dir = os.path.join(PREPROCESSED_DIR, split)
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

total_test_patches = 0
for scene_path in tqdm(test_scenes, desc=f"Processing {split} scenes"):
    patches_created = process_scene_with_tiling(scene_path, BANDS_TO_USE, TILE_SIZE, PATCH_SIZE, output_dir)
    total_test_patches += patches_created

print(f"\nTEST SET COMPLETE! Generated {total_test_patches} patches.")
print("\n" + "="*50)
print("ALL PREPROCESSING COMPLETE! (CRASH-FREE) 🎉")
print("="*50)



