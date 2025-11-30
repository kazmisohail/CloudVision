import numpy as np
from PIL import Image
from skimage.exposure import match_histograms
import io
import base64

def preprocess_v1(image_file):
    """
    V1 Preprocessing: Resize to 256x256, normalize to [0,1].
    Updated: Pads to 8 channels (RGB + 5 Zero Padding) as the model expects 8 channels.
    """
    img = Image.open(image_file).convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    
    # Create 8-channel input: 3 RGB + 5 Zero Padding
    input_tensor = np.zeros((256, 256, 8), dtype=np.float32)
    input_tensor[:, :, :3] = img_array
    
    return np.expand_dims(input_tensor, axis=0)

# def preprocess_v2(image_file):
#     """
#     V2 Preprocessing:
#     1. Resize to 256x256
#     2. Normalize [0, 1]
#     3. Pad to 8 channels (RGB at [0,1,2], rest zeros)
#     4. Standardization: Applied per-channel ONLY on RGB channels (0, 1, 2).
#        Padding channels (3-7) remain zeros.
#     """
#     img = Image.open(image_file).convert('RGB')
#     img = img.resize((256, 256))
#     img_array = np.array(img) / 255.0  # Normalize [0, 1]

#     # Create 8-channel input
#     input_tensor = np.zeros((256, 256, 8), dtype=np.float32)
#     input_tensor[:, :, :3] = img_array

#     # Standardization (Per channel, RGB only)
#     # We calculate mean/std per image to center the data, which helps 
#     # when global dataset stats are unknown but the model expects standardized inputs.
#     for i in range(3): # Only standardize RGB
#         channel = input_tensor[:, :, i]
#         mean = np.mean(channel)
#         std = np.std(channel)
#         if std > 0:
#             input_tensor[:, :, i] = (channel - mean) / std
#         else:
#             input_tensor[:, :, i] = (channel - mean) # Zero centered

#     return np.expand_dims(input_tensor, axis=0)

def preprocess_v2(image_file):
    """
    V2 Preprocessing (Safe Mode):
    1. Resize to 256x256
    2. Normalize [0, 1] ONLY. (Removing Standardization to fix 'worse' results)
    3. Pad to 8 channels.
    """
    img = Image.open(image_file).convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img)
    
    # 1. Normalize to 0-1 range (Safe & Stable)
    # Do NOT subtract mean or divide by std for single-image inference.
    normalized = img_array.astype(np.float32) / 255.0

    # 2. Create 8-channel input
    input_tensor = np.zeros((256, 256, 8), dtype=np.float32)
    
    # 3. Place RGB in the first 3 channels
    input_tensor[:, :, 0] = normalized[:, :, 0] # R
    input_tensor[:, :, 1] = normalized[:, :, 1] # G
    input_tensor[:, :, 2] = normalized[:, :, 2] # B
    
    return np.expand_dims(input_tensor, axis=0)

def preprocess_v3(image_file):
    """
    V3 Preprocessing (Same as V2 Safe Mode):
    1. Resize to 256x256
    2. Normalize [0, 1] ONLY.
    3. Pad to 8 channels.
    """
    return preprocess_v2(image_file)

# Alias for backward compatibility if needed, but views should switch to v1/v2
preprocess_image = preprocess_v1 

def remap_classes(prediction):
    """
    Maps raw model classes (0-5) to display classes (0-3).
    Raw: [0: Fill, 1: Clear, 2: Shadow, 3: Thin, 4: Thick, 5: Other]
    Target: [0: Clear, 1: Shadow, 2: Thin, 3: Thick]
    """
    # prediction shape: (1, 256, 256, 5) -> argmax -> (256, 256)
    pred_mask = np.argmax(prediction, axis=-1)[0]
    
    remapped_mask = np.zeros_like(pred_mask)
    
    # Map Shadow (2) -> 1
    remapped_mask[pred_mask == 2] = 1
    # Map Thin Cloud (3) -> 2
    remapped_mask[pred_mask == 3] = 2
    # Map Thick Cloud (4) -> 3
    remapped_mask[pred_mask == 4] = 3
    
    # 0, 1, 5 remain 0 (Clear)
    
    return remapped_mask

def mask_to_base64(mask):
    """
    Converts a class mask to a GRAYSCALE base64 image.
    Mapping:
    0 (Clear) -> 0 (Black/Transparent effectively, but here grayscale pixel value)
    1 (Shadow) -> 85 (Dark Gray)
    2 (Thin) -> 170 (Light Gray)
    3 (Thick) -> 255 (White)
    """
    h, w = mask.shape
    img_gray = np.zeros((h, w), dtype=np.uint8)
    
    # Map classes to grayscale intensities
    img_gray[mask == 1] = 85   # Shadow
    img_gray[mask == 2] = 170  # Thin Cloud
    img_gray[mask == 3] = 255  # Thick Cloud
    
    img = Image.fromarray(img_gray, mode='L') # 'L' mode for 8-bit grayscale
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def mitigate_shadows(original_image_file, mask):
    """
    Uses histogram matching to mitigate shadows.
    Treats Class 1 (Shadow) in the remapped mask as the region to correct.
    """
    img = Image.open(original_image_file).convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img)
    
    # Create a boolean mask for shadows
    shadow_mask = (mask == 1)
    
    if not np.any(shadow_mask):
        return img_array # No shadows to mitigate

    mitigated = img_array.copy()
    
    # Simple statistical correction
    for c in range(3): # RGB
        valid = img_array[:, :, c][~shadow_mask]
        shadow = img_array[:, :, c][shadow_mask]
        
        if len(valid) > 0 and len(shadow) > 0:
            mu_v, std_v = np.mean(valid), np.std(valid)
            mu_s, std_s = np.mean(shadow), np.std(shadow)
            
            # Normalize shadow and map to valid stats
            if std_s > 0:
                corrected = (shadow - mu_s) / std_s * std_v + mu_v
            else:
                corrected = shadow - mu_s + mu_v
                
            mitigated[:, :, c][shadow_mask] = np.clip(corrected, 0, 255)
            
    return mitigated

def image_to_base64(img_array):
    img = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
