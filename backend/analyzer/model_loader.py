import tensorflow as tf
from tensorflow.keras import backend as K
import os

# --- V2 CONFIGURATION (Advanced) ---
NUM_CLASSES = 5
CLASS_WEIGHTS_DICT = {0: 3.531, 1: 0.430, 2: 15.000, 3: 1.708, 4: 0.569}
# Create tensor (re-creating the clip logic used in training)
weights_v2 = tf.constant([CLASS_WEIGHTS_DICT[i] for i in range(NUM_CLASSES)], dtype=tf.float32)
weights_v2 = tf.clip_by_value(weights_v2, 0.2, 10.0)
weights_v2 = weights_v2 / tf.reduce_mean(weights_v2)
CLASS_WEIGHTS_TENSOR_V2 = weights_v2

def weighted_categorical_crossentropy_v2(weights):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss_map = K.categorical_crossentropy(y_true, y_pred)
        weight_map = K.sum(y_true * weights, axis=-1)
        return K.mean(loss_map * weight_map)
    return loss

def multiclass_soft_dice_loss_v2(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    axes = [0, 1, 2]
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true + y_pred, axis=axes)
    dice_per_class = (2. * intersection + smooth) / (denominator + smooth)
    return 1.0 - tf.reduce_mean(dice_per_class)

def combined_loss_v2(weights):
    cce = weighted_categorical_crossentropy_v2(weights)
    def loss(y_true, y_pred):
        return 0.5 * cce(y_true, y_pred) + 0.5 * multiclass_soft_dice_loss_v2(y_true, y_pred)
    return loss

# --- V1 CONFIGURATION (Legacy) ---

CLASS_WEIGHTS_DICT_V1 = {0: 3.531, 1: 0.430, 2: 15.000, 3: 1.708, 4: 0.569} # Assuming same weights were used
CLASS_WEIGHTS_TENSOR_V1 = tf.constant([CLASS_WEIGHTS_DICT_V1[i] for i in range(NUM_CLASSES)], dtype=tf.float32)

def weighted_categorical_crossentropy_v1(weights):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss_map = K.categorical_crossentropy(y_true, y_pred)
        weight_map = K.sum(y_true * weights, axis=-1)
        return K.mean(loss_map * weight_map)
    return loss

def dice_loss_v1(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def combined_loss_v1(weights):
    cce = weighted_categorical_crossentropy_v1(weights)
    def loss(y_true, y_pred):
        return 0.5 * cce(y_true, y_pred) + 0.5 * dice_loss_v1(y_true, y_pred)
    return loss


# --- V3 CONFIGURATION ---
CLASS_WEIGHTS_DICT_V3 = {
    0: 0.0,   # Fill
    1: 0.5,   # Clear
    2: 3.0,   # Shadow
    3: 3.0,   # Thin Cloud
    4: 1.0    # Thick Cloud
}
CLASS_WEIGHTS_TENSOR_V3 = tf.constant([CLASS_WEIGHTS_DICT_V3[i] for i in range(NUM_CLASSES)], dtype=tf.float32)

def weighted_categorical_crossentropy_v3(weights):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss_map = K.categorical_crossentropy(y_true, y_pred)
        # Calculate weight map based on true classes
        weight_map = K.sum(y_true * weights, axis=-1)
        # Return weighted loss
        return loss_map * weight_map
    return loss

def multiclass_soft_dice_loss_v3(y_true, y_pred, smooth=1e-6):
    """Per-channel Dice Loss to handle imbalance."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    axes = [0, 1, 2] # Batch, H, W

    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true + y_pred, axis=axes)

    dice_per_class = (2. * intersection + smooth) / (denominator + smooth)
    return 1.0 - tf.reduce_mean(dice_per_class)

def combined_loss_v3(y_true, y_pred):
    """
    UPDATED: 30% CrossEntropy, 70% Dice.
    Prioritizes Overlap (IoU) over pure pixel accuracy.
    """
    cce = weighted_categorical_crossentropy_v3(CLASS_WEIGHTS_TENSOR_V3)(y_true, y_pred)
    dice = multiclass_soft_dice_loss_v3(y_true, y_pred)
    return 0.3 * K.mean(cce) + 0.7 * dice


class ModelLoader:
    _instance = None
    _models = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def load_model(self, model_key='v2'):
        if model_key in self._models:
            return self._models[model_key]

        print(f"Loading model: {model_key}...")
        
        try:
            if model_key == 'v3':
                model_path = os.getenv('MODEL_PATH_V3', 'Attention_UNet_Balanced_Final.keras')
                custom_objects = {
                    'loss': combined_loss_v3,
                    'combined_loss': combined_loss_v3,
                    'weighted_categorical_crossentropy': weighted_categorical_crossentropy_v3(CLASS_WEIGHTS_TENSOR_V3), # Note: The user's code returned a function, but here we need to match what the model expects. 
                    'weighted_categorical_crossentropy': weighted_categorical_crossentropy_v3,
                    'multiclass_soft_dice_loss': multiclass_soft_dice_loss_v3,
                    'mean_io_u': tf.keras.metrics.OneHotMeanIoU(num_classes=5)
                }
            elif model_key == 'v2':
                model_path = os.getenv('MODEL_PATH_V2', 'Attention_UNet_Advanced_1.keras')
                custom_objects = {
                    'loss': combined_loss_v2(CLASS_WEIGHTS_TENSOR_V2),
                    'combined_loss': combined_loss_v2(CLASS_WEIGHTS_TENSOR_V2), # For safety
                    'multiclass_soft_dice_loss': multiclass_soft_dice_loss_v2,
                    'mean_io_u': tf.keras.metrics.OneHotMeanIoU(num_classes=5)
                }
            else: # v1
                model_path = os.getenv('MODEL_PATH_V1', 'model.keras')
                custom_objects = {
                    'loss': combined_loss_v1(CLASS_WEIGHTS_TENSOR_V1),
                    'combined_loss': combined_loss_v1(CLASS_WEIGHTS_TENSOR_V1),
                    'dice_loss': dice_loss_v1,
                    'weighted_categorical_crossentropy': weighted_categorical_crossentropy_v1(CLASS_WEIGHTS_TENSOR_V1),
                }

            if not os.path.exists(model_path):
                print(f"Error: Model file not found at {model_path}")
                return None

            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            self._models[model_key] = model
            print(f"Model {model_key} loaded successfully.")
            return model

        except Exception as e:
            print(f"Error loading model {model_key}: {e}")
            return None

    def get_model(self, model_key='v2'):
        return self.load_model(model_key)
