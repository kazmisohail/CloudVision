"""
**Final Year Project: Cloud Detection, Haze and Shadow Mitigation, and Cloud Classification**

Notebook 2: TFRecord Creation

Purpose: To convert the preprocessed .npy patches into efficient TFRecord files for fast model training.
"""

#pip install tensorflow
#pip install tqdm

# import libraries
import tensorflow as tf
import numpy as np
import os
import shutil
from tqdm import tqdm

print(f"Using TensorFlow version: {tf.__version__}")

# --- Configuration ---
# PROJECT_DIR = '/content/drive/MyDrive/Final_Year_Project'
PREPROCESSED_DIR = ('F:\\Final_Year_Project\\Preprocessed_Data')
TFRECORD_DIR = ('F:\\Final_Year_Project\\TFRecord_Data')

os.makedirs(TFRECORD_DIR, exist_ok=True)
print(f"TFRecord directory is: {TFRECORD_DIR}")

"""## Step 1: Helper functions to create TFRecord features"""

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_example(image_path, mask_path):
    image_array = np.load(image_path)
    mask_array = np.load(mask_path)

    height, width, channels = image_array.shape

    image_bytes = image_array.tobytes()
    mask_bytes = mask_array.tobytes()

    feature = {
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'channels': _int64_feature(channels),
        'image_raw': _bytes_feature(image_bytes),
        'mask_raw': _bytes_feature(mask_bytes),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

"""## Step 2: Main loop to write TFRecord files"""

for split in ['train', 'validation', 'test']:
    print(f"\n--- Processing '{split}' split ---")
    output_path = os.path.join(TFRECORD_DIR, f'{split}.tfrecord')

    if os.path.exists(output_path):
        print(f"Found '{output_path}'. Skipping conversion.")
        continue


    print(f"TFRecord file not found. Starting conversion process...")
    img_dir = os.path.join(PREPROCESSED_DIR, split, 'images')
    mask_dir = os.path.join(PREPROCESSED_DIR, split, 'masks')

    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f"  [Warning] Directory not found for '{split}' split. Skipping.")
        continue

    image_paths = sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir)])
    mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])

    with tf.io.TFRecordWriter(output_path) as writer:
        for img_p, mask_p in tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc=f"Writing {split} TFRecord"):
            tf_example = create_example(img_p, mask_p)
            writer.write(tf_example.SerializeToString())

    print(f"Successfully created '{output_path}'")

print("\n--- TFRecord conversion process complete! ---")







