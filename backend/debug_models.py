import tensorflow as tf
import os
import numpy as np

# Define V2 custom objects to load it successfully
def weighted_categorical_crossentropy_v2(weights):
    def loss(y_true, y_pred):
        return 0.0
    return loss

def multiclass_soft_dice_loss_v2(y_true, y_pred, smooth=1e-6):
    return 0.0

def combined_loss_v2(weights):
    def loss(y_true, y_pred):
        return 0.0
    return loss

# Dummy weights
weights_v2 = tf.constant([1.0]*5, dtype=tf.float32)

custom_objects_v2 = {
    'loss': combined_loss_v2(weights_v2),
    'combined_loss_v2': combined_loss_v2(weights_v2),
    'multiclass_soft_dice_loss_v2': multiclass_soft_dice_loss_v2,
    'weighted_categorical_crossentropy_v2': weighted_categorical_crossentropy_v2(weights_v2),
    'mean_io_u': tf.keras.metrics.OneHotMeanIoU(num_classes=5)
}

# Standard custom objects (guessing for V1/others)
def dice_loss(y_true, y_pred): return 0.0
def weighted_categorical_crossentropy(weights): 
    def loss(y_true, y_pred): return 0.0
    return loss
def combined_loss(weights):
    def loss(y_true, y_pred): return 0.0
    return loss

custom_objects_std = {
    'dice_loss': dice_loss,
    'weighted_categorical_crossentropy': weighted_categorical_crossentropy(weights_v2),
    'combined_loss': combined_loss(weights_v2),
    'loss': combined_loss(weights_v2)
}

models_dir = 'd:/CloudyyVision/backend/models'
model_files = [
    'Attention_UNet_Improved_2.keras',
    'Attention_UNet_Balanced_Final.keras',
    'Attention_UNet_Advanced_1.keras'
]

for mf in model_files:
    path = os.path.join(models_dir, mf)
    print(f"\n--- Inspecting {mf} ---")
    try:
        # Try loading with V2 objects first (most comprehensive)
        try:
            model = tf.keras.models.load_model(path, custom_objects=custom_objects_v2)
        except:
            # Fallback to standard
            model = tf.keras.models.load_model(path, custom_objects=custom_objects_std)
            
        print(f"Input Shape: {model.input_shape}")
        print(f"Output Shape: {model.output_shape}")
        
    except Exception as e:
        print(f"Failed to load: {e}")
