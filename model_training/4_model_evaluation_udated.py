"""
**Final Year Project: Cloud Detection, Haze and Shadow Mitigation, and Cloud Classification**

Notebook 4: Model Evaluation updated

Purpose: To quantitatively and qualitatively evaluate the trained
8-channel Attention U-Net model on the test set.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm.notebook import tqdm
import pandas as pd

PROJECT_DIR = '/content/drive/MyDrive/Final_Year_Project'
PREPROCESSED_DIR = os.path.join(PROJECT_DIR, 'Preprocessed_Data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'Models')
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 256, 256, 8
NUM_CLASSES = 5

MODEL_FILENAME = "Attention_UNet_Balanced_Final.keras"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

CLASS_COLORS = np.array([
    [0, 0, 0],       # Fill
    [34, 139, 34],   # Clear
    [0, 0, 139],     # Shadow
    [255, 255, 0],   # Thin Cloud
    [255, 255, 255]  # Thick Cloud
], dtype=np.uint8)

# --- Visualization Colors ---
CLASS_NAMES = ['Fill', 'Clear', 'Cloud Shadow', 'Thin Cloud', 'Thick Cloud']
# CLASS_COLORS = np.array([
#     [50, 50, 50],    # Fill - Dark Gray
#     [34, 139, 34],   # Clear - Forest Green
#     [0, 0, 139],     # Shadow - Dark Blue
#     [173, 216, 230], # Thin Cloud - Light Blue
#     [255, 255, 255]  # Thick Cloud - White
# ], dtype=np.uint8)

print(f"Evaluator initialized. Target model: {MODEL_FILENAME}")

"""## Step 1: Load the Trained Model with Custom Objects"""

# final wala
def weighted_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss_map = K.categorical_crossentropy(y_true, y_pred)
        weight_map = K.sum(y_true * weights, axis=-1)
        return loss_map * weight_map
    return loss

def multiclass_soft_dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    axes = [0, 1, 2] # Batch, H, W

    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true + y_pred, axis=axes)

    dice_per_class = (2. * intersection + smooth) / (denominator + smooth)
    return 1.0 - tf.reduce_mean(dice_per_class)

def combined_loss(y_true, y_pred):
    cce = weighted_categorical_crossentropy(CLASS_WEIGHTS_TENSOR)(y_true, y_pred)
    dice = multiclass_soft_dice_loss(y_true, y_pred)
    return 0.3 * K.mean(cce) + 0.7 * dice

print("Loss function defined.")

# 1. Loss Functions
def weighted_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss_map = K.categorical_crossentropy(y_true, y_pred)
        weight_map = K.sum(y_true * weights, axis=-1)
        return K.mean(loss_map * weight_map)
    return loss

def multiclass_soft_dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    axes = [0, 1, 2]
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true + y_pred, axis=axes)
    dice_per_class = (2. * intersection + smooth) / (denominator + smooth)
    return 1.0 - tf.reduce_mean(dice_per_class)

def combined_loss(weights):
    cce = weighted_categorical_crossentropy(weights)
    def loss(y_true, y_pred):
        return 0.5 * cce(y_true, y_pred) + 0.5 * multiclass_soft_dice_loss(y_true, y_pred)
    return loss

# 2. Class Weights (Must match training)
# CLASS_WEIGHTS_DICT = {0: 3.531, 1: 0.430, 2: 15.000, 3: 1.708, 4: 0.569}
# weights = tf.constant([CLASS_WEIGHTS_DICT[i] for i in range(NUM_CLASSES)], dtype=tf.float32)
# weights = tf.clip_by_value(weights, 0.2, 10.0)
# weights = weights / tf.reduce_mean(weights)
# CLASS_WEIGHTS_TENSOR = weights

CLASS_WEIGHTS_DICT = {
    0: 0.0,   # Fill
    1: 0.5,   # Clear
    2: 3.0,   # Shadow
    3: 3.0,   # Thin Cloud
    4: 1.0    # Thick Cloud
}
CLASS_WEIGHTS_TENSOR = tf.constant([CLASS_WEIGHTS_DICT[i] for i in range(NUM_CLASSES)], dtype=tf.float32)

# 3. Custom Objects Dictionary
# custom_objects = {
#     'loss': combined_loss(CLASS_WEIGHTS_TENSOR),
#     'multiclass_soft_dice_loss': multiclass_soft_dice_loss,
#     # If you used OneHotMeanIoU during training, map 'mean_io_u' to it
#     'mean_io_u': tf.keras.metrics.OneHotMeanIoU(num_classes=NUM_CLASSES)
# }

custom_objects = {
            'loss': combined_loss,
            'combined_loss': combined_loss,

            'weighted_categorical_crossentropy': weighted_categorical_crossentropy,
            'multiclass_soft_dice_loss': multiclass_soft_dice_loss,

            'mean_io_u': tf.keras.metrics.OneHotMeanIoU(num_classes=5)
        }

print("--- Loading Model ---")
try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

print(f"--- Loading Model from {MODEL_FILENAME} ---")
try:
    custom_objects['mean_io_u'] = tf.keras.metrics.OneHotMeanIoU(num_classes=NUM_CLASSES)
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

"""## Step 2: Qualitative Evaluation (Visual Inspection)"""

# advance
def preprocess_for_inference(image_path):
    image = np.load(image_path).astype(np.float32)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    std = np.std(image, axis=(0, 1), keepdims=True)
    image = (image - mean) / (std + 1e-6)
    return image

def mask_to_rgb_no_fill(mask):
    return CLASS_COLORS[mask]

# advance
test_img_dir = os.path.join(PREPROCESSED_DIR, 'test', 'images')
test_mask_dir = os.path.join(PREPROCESSED_DIR, 'test', 'masks')
test_image_paths = sorted([os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir)])
test_mask_paths = sorted([os.path.join(test_mask_dir, f) for f in os.listdir(test_mask_dir)])

def visualize_predictions(num_samples=5):
    indices = np.random.choice(len(test_image_paths), num_samples, replace=False)
    plt.figure(figsize=(15, 5 * num_samples))

    for i, idx in enumerate(indices):
        raw_image = np.load(test_image_paths[idx])
        input_tensor = preprocess_for_inference(test_image_paths[idx])
        pred_probs = model.predict(np.expand_dims(input_tensor, axis=0), verbose=0)
        pred_mask = np.argmax(pred_probs[0], axis=-1)
        true_mask = np.load(test_mask_paths[idx])

        rgb_view = raw_image[:, :, [2, 1, 0]]
        rgb_view = (rgb_view - np.min(rgb_view)) / (np.max(rgb_view) - np.min(rgb_view))

        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(rgb_view); plt.title("Input (RGB)"); plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(mask_to_rgb_no_fill(true_mask)); plt.title("Ground Truth"); plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(mask_to_rgb_no_fill(pred_mask)); plt.title("Prediction"); plt.axis('off')

    plt.tight_layout()
    plt.show()

print("\n--- Visual Check ---")
visualize_predictions(15)

# --- Load File Paths ---
test_img_dir = os.path.join(PREPROCESSED_DIR, 'test', 'images')
test_mask_dir = os.path.join(PREPROCESSED_DIR, 'test', 'masks')
test_image_paths = sorted([os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir)])
test_mask_paths = sorted([os.path.join(test_mask_dir, f) for f in os.listdir(test_mask_dir)])

# balanced
def preprocess_for_inference(image_path):
    image = np.load(image_path).astype(np.float32)

    mean = np.mean(image, axis=(0, 1), keepdims=True)
    std = np.std(image, axis=(0, 1), keepdims=True)
    image = (image - mean) / (std + 1e-6)

    return image

def mask_to_rgb(mask):
    return CLASS_COLORS[mask]

def visualize_predictions(num_samples=5):
    indices = np.random.choice(len(test_image_paths), num_samples, replace=False)

    plt.figure(figsize=(15, 5 * num_samples))

    for i, idx in enumerate(indices):
        raw_image = np.load(test_image_paths[idx])
        input_tensor = preprocess_for_inference(test_image_paths[idx])
        input_batch = np.expand_dims(input_tensor, axis=0)

        pred_probs = model.predict(input_batch, verbose=0)
        pred_mask = np.argmax(pred_probs[0], axis=-1)

        true_mask = np.load(test_mask_paths[idx])

        rgb_view = raw_image[:, :, [2, 1, 0]]
        rgb_view = (rgb_view - np.min(rgb_view)) / (np.max(rgb_view) - np.min(rgb_view))

        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(rgb_view)
        plt.title("Input (RGB View)")
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(mask_to_rgb(true_mask))
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(mask_to_rgb(pred_mask))
        plt.title("Prediction")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# for balanced model
print("\n--- Visual Evaluation ---")
visualize_predictions(num_samples=15)

print("\n--- Visual Evaluation ---")
visualize_predictions(num_samples=15)

"""## Step 3: Quantitative Evaluation (Metrics and Confusion Matrix)"""

def run_evaluation(num_samples=500):
    print(f"\n--- Running Quantitative Evaluation on {num_samples} samples ---")

    indices = np.random.choice(len(test_image_paths), num_samples, replace=False)

    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    print("Processing batches...")
    for idx in tqdm(indices):
        input_tensor = preprocess_for_inference(test_image_paths[idx])
        input_batch = np.expand_dims(input_tensor, axis=0)

        pred_probs = model.predict(input_batch, verbose=0)
        pred_mask = np.argmax(pred_probs[0], axis=-1).flatten()

        true_mask = np.load(test_mask_paths[idx]).flatten()

        cm = confusion_matrix(true_mask, pred_mask, labels=range(NUM_CLASSES))
        total_cm += cm

    print("\n" + "="*30)
    print("     FINAL RESULTS")
    print("="*30)

    ious = []

    print(f"{'Class Name':<15} | {'IoU':<8} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 50)

    for i in range(NUM_CLASSES):
        tp = total_cm[i, i]
        fp = np.sum(total_cm[:, i]) - tp
        fn = np.sum(total_cm[i, :]) - tp

        union = tp + fp + fn
        iou = tp / union if union > 0 else 0
        ious.append(iou)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"{CLASS_NAMES[i]:<15} | {iou:.4f}   | {precision:.4f}     | {recall:.4f}")

    print("-" * 50)
    print(f"Mean IoU (All):      {np.mean(ious):.4f}")
    print(f"Mean IoU (No Fill):  {np.mean(ious[1:]):.4f}") # Ignoring Fill class often gives realistic score

    # --- Plot Confusion Matrix ---
    plt.figure(figsize=(10, 8))
    cm_normalized = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# for balanced model
run_evaluation(num_samples=50)

run_evaluation(num_samples=50)



def run_full_evaluation():
    print(f"\n--- Running Full Evaluation on {len(test_image_paths)} samples ---")

    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    BATCH_SIZE_EVAL = 32
    num_batches = int(np.ceil(len(test_image_paths) / BATCH_SIZE_EVAL))

    for i in tqdm(range(num_batches), desc="Evaluating Batches"):
        start_idx = i * BATCH_SIZE_EVAL
        end_idx = min((i + 1) * BATCH_SIZE_EVAL, len(test_image_paths))

        batch_input = []
        batch_true = []

        for idx in range(start_idx, end_idx):
            batch_input.append(preprocess_for_inference(test_image_paths[idx]))
            batch_true.append(np.load(test_mask_paths[idx]).flatten())

        batch_input = np.array(batch_input)
        batch_preds = np.argmax(model.predict(batch_input, verbose=0), axis=-1)

        for j in range(len(batch_true)):
            cm = confusion_matrix(batch_true[j], batch_preds[j].flatten(), labels=range(NUM_CLASSES))
            total_cm += cm

    class_names_report = ['Fill', 'Clear', 'Cloud Shadow', 'Thin Cloud', 'Thick Cloud']
    metrics_data = []

    for i in range(NUM_CLASSES):
        tp = total_cm[i, i]
        fp = np.sum(total_cm[:, i]) - tp
        fn = np.sum(total_cm[i, :]) - tp

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        metrics_data.append({
            'Class': class_names_report[i],
            'IoU': iou,
            'Precision': precision,
            'Recall': recall
        })

    df_metrics = pd.DataFrame(metrics_data)

    print("\n" + "="*30 + "\n     FINAL METRICS\n" + "="*30)
    print(df_metrics.to_string(index=False, float_format="%.4f"))

    mean_iou_all = df_metrics['IoU'].mean()
    mean_iou_no_fill = df_metrics['IoU'][1:].mean()

    print("-" * 50)
    print(f"Mean IoU (All):      {mean_iou_all:.4f}")
    print(f"Mean IoU (No Fill):  {mean_iou_no_fill:.4f} (Key Report Metric)")

    return df_metrics, total_cm

df_metrics, total_cm = run_full_evaluation()

CLASS_NAMES = ['Fill', 'Clear', 'Cloud Shadow', 'Thin Cloud', 'Thick Cloud']

plt.figure(figsize=(10, 6))
sns.barplot(x='Class', y='IoU', data=df_metrics, palette='viridis')
plt.title('Per-Class Intersection over Union (IoU)')
plt.ylim(0, 1)
for index, row in df_metrics.iterrows():
    plt.text(index, row.IoU + 0.02, f'{row.IoU:.2f}', color='black', ha="center")
plt.show()

plt.figure(figsize=(10, 8))
cm_normalized = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Normalized Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

df_melted = df_metrics.melt(id_vars=['Class'], value_vars=['Precision', 'Recall'], var_name='Metric', value_name='Score')
plt.figure(figsize=(10, 6))
sns.barplot(x='Class', y='Score', hue='Metric', data=df_melted, palette='muted')
plt.title('Precision vs Recall by Class')
plt.ylim(0, 1)
plt.show()



def run_full_evaluation():
    print(f"\n--- Running Full Evaluation on {len(test_image_paths)} samples ---")

    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    BATCH_SIZE_EVAL = 32
    num_batches = int(np.ceil(len(test_image_paths) / BATCH_SIZE_EVAL))

    for i in tqdm(range(num_batches), desc="Evaluating Batches"):
        start_idx = i * BATCH_SIZE_EVAL
        end_idx = min((i + 1) * BATCH_SIZE_EVAL, len(test_image_paths))

        batch_input = []
        batch_true = []

        for idx in range(start_idx, end_idx):
            batch_input.append(preprocess_for_inference(test_image_paths[idx]))
            batch_true.append(np.load(test_mask_paths[idx]).flatten())

        batch_input = np.array(batch_input)
        batch_preds = np.argmax(model.predict(batch_input, verbose=0), axis=-1)

        for j in range(len(batch_true)):
            cm = confusion_matrix(batch_true[j], batch_preds[j].flatten(), labels=range(NUM_CLASSES))
            total_cm += cm

    class_names_report = ['Fill', 'Clear', 'Cloud Shadow', 'Thin Cloud', 'Thick Cloud']
    metrics_data = []

    for i in range(NUM_CLASSES):
        tp = total_cm[i, i]
        fp = np.sum(total_cm[:, i]) - tp
        fn = np.sum(total_cm[i, :]) - tp

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        metrics_data.append({
            'Class': class_names_report[i],
            'IoU': iou,
            'Precision': precision,
            'Recall': recall
        })

    df_metrics = pd.DataFrame(metrics_data)

    print("\n" + "="*30 + "\n     FINAL METRICS\n" + "="*30)
    print(df_metrics.to_string(index=False, float_format="%.4f"))

    mean_iou_all = df_metrics['IoU'].mean()
    mean_iou_no_fill = df_metrics['IoU'][1:].mean()

    print("-" * 50)
    print(f"Mean IoU (All):      {mean_iou_all:.4f}")
    print(f"Mean IoU (No Fill):  {mean_iou_no_fill:.4f} (Key Report Metric)")

    return df_metrics, total_cm

print("\n--- Visual Check ---")
visualize_predictions(15)

def run_full_evaluation():
    print(f"\n--- Running Full Evaluation on {len(test_image_paths)} samples ---")

    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    BATCH_SIZE_EVAL = 32
    num_batches = int(np.ceil(len(test_image_paths) / BATCH_SIZE_EVAL))

    for i in tqdm(range(num_batches), desc="Evaluating Batches"):
        start_idx = i * BATCH_SIZE_EVAL
        end_idx = min((i + 1) * BATCH_SIZE_EVAL, len(test_image_paths))

        batch_input = []
        batch_true = []

        for idx in range(start_idx, end_idx):
            batch_input.append(preprocess_for_inference(test_image_paths[idx]))
            batch_true.append(np.load(test_mask_paths[idx]).flatten())

        batch_input = np.array(batch_input)
        batch_preds = np.argmax(model.predict(batch_input, verbose=0), axis=-1)

        for j in range(len(batch_true)):
            cm = confusion_matrix(batch_true[j], batch_preds[j].flatten(), labels=range(NUM_CLASSES))
            total_cm += cm

    class_names_report = ['Fill', 'Clear', 'Cloud Shadow', 'Thin Cloud', 'Thick Cloud']
    metrics_data = []

    for i in range(NUM_CLASSES):
        tp = total_cm[i, i]
        fp = np.sum(total_cm[:, i]) - tp
        fn = np.sum(total_cm[i, :]) - tp

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        metrics_data.append({
            'Class': class_names_report[i],
            'IoU': iou,
            'Precision': precision,
            'Recall': recall
        })

    df_metrics = pd.DataFrame(metrics_data)

    print("\n" + "="*30 + "\n     FINAL METRICS\n" + "="*30)
    print(df_metrics.to_string(index=False, float_format="%.4f"))

    mean_iou_all = df_metrics['IoU'].mean()
    mean_iou_no_fill = df_metrics['IoU'][1:].mean()

    print("-" * 50)
    print(f"Mean IoU (All):      {mean_iou_all:.4f}")
    print(f"Mean IoU (No Fill):  {mean_iou_no_fill:.4f} (Key Report Metric)")

    return df_metrics, total_cm

df_metrics, total_cm = run_full_evaluation()

plt.figure(figsize=(10, 6))
sns.barplot(x='Class', y='IoU', data=df_metrics, palette='viridis')
plt.title('Per-Class Intersection over Union (IoU)')
plt.ylim(0, 1)
for index, row in df_metrics.iterrows():
    plt.text(index, row.IoU + 0.02, f'{row.IoU:.2f}', color='black', ha="center")
plt.show()

plt.figure(figsize=(10, 8))
cm_normalized = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Normalized Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

df_melted = df_metrics.melt(id_vars=['Class'], value_vars=['Precision', 'Recall'], var_name='Metric', value_name='Score')
plt.figure(figsize=(10, 6))
sns.barplot(x='Class', y='Score', hue='Metric', data=df_melted, palette='muted')
plt.title('Precision vs Recall by Class')
plt.ylim(0, 1)
plt.show()

