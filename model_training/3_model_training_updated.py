"""
**Final Year Project: Cloud Detection, Haze and Shadow Mitigation, and Cloud Classification**

Notebook 3: Model Training

Purpose: To build, compile, and train the definitive Attention U-Net model
on the 8-channel, preprocessed TFRecord dataset.
"""

import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

print(f"Using TensorFlow version: {tf.__version__}")

"""## Step 1: Setup and Hyperparameters"""

PROJECT_DIR = '/content/drive/MyDrive/Final_Year_Project'
TFRECORD_DIR = os.path.join(PROJECT_DIR, 'TFRecord_Data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'Models')
os.makedirs(MODELS_DIR, exist_ok=True)

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 8
NUM_CLASSES = 5
BATCH_SIZE = 16
EPOCHS = 60

# --- Class Weights ---
CLASS_WEIGHTS_DICT = {
    0: 0.0,   # Fill
    1: 0.5,   # Clear
    2: 3.0,   # Shadow
    3: 3.0,   # Thin Cloud
    4: 1.0    # Thick Cloud
}
CLASS_WEIGHTS_TENSOR = tf.constant([CLASS_WEIGHTS_DICT[i] for i in range(NUM_CLASSES)], dtype=tf.float32)

print("Setup and hyperparameters are ready.")

# CLASS_WEIGHTS_DICT = {0: 3.531, 1: 0.430, 2: 15.000, 3: 1.708, 4: 0.569}
# weights = tf.constant([CLASS_WEIGHTS_DICT[i] for i in range(NUM_CLASSES)], dtype=tf.float32)
# weights = tf.clip_by_value(weights, 0.2, 10.0) # Clip extreme values
# weights = weights / tf.reduce_mean(weights)     # Normalize to keep loss scale stable
# CLASS_WEIGHTS_TENSOR = weights

print("Setup and hyperparameters are ready.")
print(f"Using softened class weights: {CLASS_WEIGHTS_TENSOR.numpy()}")

"""## Step 2: The Data Loading and Augmentation Pipeline"""

def parse_tfrecord_fn(example):
    feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'channels': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'mask_raw': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, feature_description)

    height, width, channels = example['height'], example['width'], example['channels']
    image = tf.io.decode_raw(example['image_raw'], out_type=tf.float32)
    mask = tf.io.decode_raw(example['mask_raw'], out_type=tf.uint8)

    image = tf.reshape(image, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    mask = tf.reshape(mask, (IMG_HEIGHT, IMG_WIDTH))

    ch_mean = tf.math.reduce_mean(image, axis=[0,1], keepdims=True)
    ch_std  = tf.math.reduce_std(image,  axis=[0,1], keepdims=True)
    image   = (image - ch_mean) / (ch_std + 1e-6)

    mask_one_hot = tf.one_hot(tf.cast(mask, tf.int32), depth=NUM_CLASSES)

    sample_weight = tf.where(tf.equal(mask, 0), 0.0, 1.0)
    sample_weight = tf.cast(sample_weight, tf.float32)

    sample_weight = tf.expand_dims(sample_weight, axis=-1)

    return image, mask_one_hot, sample_weight

def augment_data(image, mask, sample_weight):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
        sample_weight = tf.image.flip_left_right(sample_weight) # Now works because shape is (H,W,1)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
        sample_weight = tf.image.flip_up_down(sample_weight)

    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)
    sample_weight = tf.image.rot90(sample_weight, k)

    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05, dtype=tf.float32)
    image = image + noise

    return image, mask, sample_weight

def create_dataset(tfrecord_path, augment=False):
    dataset = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = (
        dataset
        .repeat()
        .shuffle(buffer_size=1024)
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return dataset

train_tfrecord_path = os.path.join(TFRECORD_DIR, 'train.tfrecord')
val_tfrecord_path = os.path.join(TFRECORD_DIR, 'validation.tfrecord')

train_dataset = create_dataset(train_tfrecord_path, augment=True)
val_dataset = create_dataset(val_tfrecord_path, augment=False)
print("Data pipeline ready.")

# def augment_data(image, mask):
#     if tf.random.uniform(()) > 0.5:
#         image = tf.image.flip_left_right(image)
#         mask = tf.image.flip_left_right(mask)

#     if tf.random.uniform(()) > 0.5:
#         image = tf.image.flip_up_down(image)
#         mask = tf.image.flip_up_down(mask)

#     k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
#     image = tf.image.rot90(image, k)
#     mask = tf.image.rot90(mask, k)

#     return image, mask

# def create_dataset(tfrecord_path, augment=False):
#     dataset = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=tf.data.AUTOTUNE)
#     dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

#     if augment:
#         dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

#     dataset = (
#         dataset
#         .repeat()  # <-- CHANGE: Added .repeat() to ensure training runs for all epochs
#         .shuffle(buffer_size=1024)
#         .batch(BATCH_SIZE)
#         .prefetch(buffer_size=tf.data.AUTOTUNE)
#     )
#     return dataset

# train_tfrecord_path = os.path.join(TFRECORD_DIR, 'train.tfrecord')
# val_tfrecord_path = os.path.join(TFRECORD_DIR, 'validation.tfrecord')

# train_dataset = create_dataset(train_tfrecord_path, augment=True)
# val_dataset = create_dataset(val_tfrecord_path, augment=False)
# print("Data pipeline ready.")

"""## Step 3: Loss Function and Metrics"""

def weighted_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss_map = K.categorical_crossentropy(y_true, y_pred)
        weight_map = K.sum(y_true * weights, axis=-1)
        weighted_loss = loss_map * weight_map
        return K.mean(weighted_loss)
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

print("Loss function defined.")

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
    axes = [0, 1, 2]

    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true + y_pred, axis=axes)

    dice_per_class = (2. * intersection + smooth) / (denominator + smooth)
    return 1.0 - tf.reduce_mean(dice_per_class)

def combined_loss(y_true, y_pred):
    cce = weighted_categorical_crossentropy(CLASS_WEIGHTS_TENSOR)(y_true, y_pred)
    dice = multiclass_soft_dice_loss(y_true, y_pred)
    return 0.3 * K.mean(cce) + 0.7 * dice

print("Loss function defined.")

"""## Step 4: Attention U-Net Model Architecture"""

def conv_block(inputs, num_filters, dropout_rate=0.1):
    x = layers.Conv2D(num_filters, 3, padding="same", kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(num_filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def attention_gate(g, x, num_filters):
    Wg = layers.Conv2D(num_filters, 1, padding="same")(g)
    Wg = layers.BatchNormalization()(Wg)
    Wx = layers.Conv2D(num_filters, 1, padding="same")(x)
    Wx = layers.BatchNormalization()(Wx)
    out = layers.Activation("relu")(Wg + Wx)
    out = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(out)
    return out * x

def attention_unet_model(input_shape, num_classes):
    inputs = layers.Input(input_shape)
    # Encoder
    s1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D(2)(s1)
    s2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D(2)(s2)
    s3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D(2)(s3)
    s4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D(2)(s4)
    # Bridge
    b1 = conv_block(p4, 1024, dropout_rate=0.2)
    # Decoder
    d1 = layers.Conv2DTranspose(512, 2, strides=2, padding="same")(b1)
    g1 = attention_gate(d1, s4, 512)
    d1 = layers.concatenate([g1, d1])
    d1 = conv_block(d1, 512)
    d2 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(d1)
    g2 = attention_gate(d2, s3, 256)
    d2 = layers.concatenate([g2, d2])
    d2 = conv_block(d2, 256)
    d3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(d2)
    g3 = attention_gate(d3, s2, 128)
    d3 = layers.concatenate([g3, d3])
    d3 = conv_block(d3, 128)
    d4 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(d3)
    g4 = attention_gate(d4, s1, 64)
    d4 = layers.concatenate([g4, d4])
    d4 = conv_block(d4, 64)
    # Output Layer
    outputs = layers.Conv2D(num_classes, 1, padding="same", activation="softmax")(d4)
    model = Model(inputs, outputs, name="Attention_UNet_8_Channel")
    return model

def conv_block(inputs, num_filters, dropout_rate=0.3):
    x = layers.Conv2D(num_filters, 3, padding="same", kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(num_filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def attention_gate(g, x, num_filters):
    Wg = layers.Conv2D(num_filters, 1, padding="same")(g)
    Wg = layers.BatchNormalization()(Wg)
    Wx = layers.Conv2D(num_filters, 1, padding="same")(x)
    Wx = layers.BatchNormalization()(Wx)
    out = layers.Activation("relu")(Wg + Wx)
    out = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(out)
    return out * x

def attention_unet_model(input_shape, num_classes):
    inputs = layers.Input(input_shape)
    # Encoder
    s1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D(2)(s1)
    s2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D(2)(s2)
    s3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D(2)(s3)
    s4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D(2)(s4)
    # Bridge
    b1 = conv_block(p4, 1024, dropout_rate=0.4)
    # Decoder
    d1 = layers.Conv2DTranspose(512, 2, strides=2, padding="same")(b1)
    g1 = attention_gate(d1, s4, 512)
    d1 = layers.concatenate([g1, d1])
    d1 = conv_block(d1, 512)
    d2 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(d1)
    g2 = attention_gate(d2, s3, 256)
    d2 = layers.concatenate([g2, d2])
    d2 = conv_block(d2, 256)
    d3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(d2)
    g3 = attention_gate(d3, s2, 128)
    d3 = layers.concatenate([g3, d3])
    d3 = conv_block(d3, 128)
    d4 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(d3)
    g4 = attention_gate(d4, s1, 64)
    d4 = layers.concatenate([g4, d4])
    d4 = conv_block(d4, 64)
    # Output (Float32 for mixed precision stability)
    outputs = layers.Conv2D(num_classes, 1, padding="same", activation="softmax", dtype='float32')(d4)

    return Model(inputs, outputs, name="Attention_UNet_Balanced_Final")

"""## Step 5: Training the Model"""

# --- Calculate Steps ---
def count_data_items(tfrecord_path):
    return sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path))

num_train_samples = count_data_items(train_tfrecord_path)
num_val_samples = count_data_items(val_tfrecord_path)
steps_per_epoch = math.ceil(num_train_samples / BATCH_SIZE)
validation_steps = math.ceil(num_val_samples / BATCH_SIZE)

print(f"\nTraining samples: {num_train_samples}, Validation samples: {num_val_samples}")
print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")

initial_learning_rate = 3e-4
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=steps_per_epoch * EPOCHS
)

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=1e-5
)

iou_metric = tf.keras.metrics.OneHotMeanIoU(num_classes=NUM_CLASSES, name="mean_io_u")

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model = attention_unet_model(input_shape, NUM_CLASSES)
model.compile(
    optimizer=optimizer,
    loss=combined_loss(CLASS_WEIGHTS_TENSOR),
    metrics=[iou_metric]
)
model.summary()

model_path = os.path.join(MODELS_DIR, "Attention_UNet_Advanced_1.keras")
callbacks = [
    ModelCheckpoint(model_path, save_best_only=True, monitor="val_mean_io_u", mode='max', verbose=1),
    EarlyStopping(monitor="val_loss", patience=20, verbose=1, mode='min', restore_best_weights=True)
]

# --- Let's Train! ---
print("\n--- Starting Final Model Training ---")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=callbacks
)
print("\n--- Model training complete! ---")

# --- Build Model ---
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model = attention_unet_model(input_shape, NUM_CLASSES)

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4,
    decay_steps=EPOCHS * steps_per_epoch,
    alpha=0.01
)

optimizer = optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

iou_metric = tf.keras.metrics.OneHotMeanIoU(num_classes=NUM_CLASSES, name="mean_io_u")

model.compile(
    optimizer=optimizer,
    loss=combined_loss,
    metrics=[iou_metric],
    jit_compile=True
)
model.summary()

# --- Callbacks ---
model_path = os.path.join(MODELS_DIR, "Attention_UNet_Balanced_Final.keras")
callbacks = [
    ModelCheckpoint(model_path, save_best_only=True, monitor="val_mean_io_u", mode='max', verbose=1),
    EarlyStopping(monitor="val_mean_io_u", patience=15, verbose=1, mode='max', restore_best_weights=True)
]

print("\nSTARTING FINAL BALANCED TRAINING...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=callbacks
)
print("\n--- Training Complete ---")

"""## Step 6: Plotting Training History"""

plt.figure(figsize=(12, 5))
# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(); plt.grid(True)

# Plot Mean IoU
plt.subplot(1, 2, 2)
plt.plot(history.history['mean_io_u'], label='Training Mean IoU')
plt.plot(history.history['val_mean_io_u'], label='Validation Mean IoU')
plt.title('Model Mean IoU')
plt.ylabel('Mean IoU')
plt.xlabel('Epoch')
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()

