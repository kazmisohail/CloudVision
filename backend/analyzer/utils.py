import numpy as np
from PIL import Image
from skimage.exposure import match_histograms
import io
import base64

def preprocess_v1(file_obj):
    """
    V1 Preprocessing:
    Case A: Standard Image -> Resize to 256x256, normalize [0,1], Pad to 8 channels.
    Case B: Scientific Data (.npy) -> Pass-through.
    """
    filename = getattr(file_obj, 'name', '').lower()
    
    if filename.endswith('.npy'):
        try:
            if hasattr(file_obj, 'seek'):
                file_obj.seek(0)
            data = np.load(file_obj)
            if data.shape[-1] != 8:
                 raise ValueError(f"Expected 8 channels, got {data.shape[-1]}")
            return np.expand_dims(data.astype(np.float32), axis=0)
        except Exception as e:
            raise ValueError(f"Invalid .npy file: {e}")
            
    else:
        # Existing V1 RGB Logic
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
            
        img = Image.open(file_obj).convert('RGB')
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
        
        # Create 8-channel input: 3 RGB + 5 Zero Padding
        input_tensor = np.zeros((256, 256, 8), dtype=np.float32)
        input_tensor[:, :, :3] = img_array
        
        return np.expand_dims(input_tensor, axis=0)

def preprocess_v2_legacy(image_file):
    """
    Legacy V2 Preprocessing (Safe Mode):
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

def preprocess_v2(file_obj):
    """
    Polymorphic Preprocessor for V2:
    Case A: Standard Image (JPG/PNG) -> Resize, Normalize, Pad to 8 channels.
    Case B: Scientific Data (.npy) -> Load, Verify Shape, Pass-through.
    """
    filename = getattr(file_obj, 'name', '').lower()
    
    if filename.endswith('.npy'):
        # Case B: Scientific Data (.npy)
        try:
            # Reset file pointer just in case
            if hasattr(file_obj, 'seek'):
                file_obj.seek(0)
            
            data = np.load(file_obj)
            
            # Verify shape
            if data.shape != (256, 256, 8):
                if data.shape[-1] != 8:
                     raise ValueError(f"Expected 8 channels, got {data.shape[-1]}")
                
                pass 

            return np.expand_dims(data.astype(np.float32), axis=0)
            
        except Exception as e:
            print(f"Error loading .npy file: {e}")
            # Fallback or re-raise?
            raise ValueError(f"Invalid .npy file: {e}")

    else:
        # Case A: Standard Image
        # Reset file pointer
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
            
        img = Image.open(file_obj).convert('RGB')
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0  # Normalize [0, 1]

        # Padding: Create a dummy 8-channel tensor (256, 256, 8)
        input_tensor = np.zeros((256, 256, 8), dtype=np.float32)
        
        # Fill channels 0, 1, 2 with RGB
        input_tensor[:, :, 0] = img_array[:, :, 0]
        input_tensor[:, :, 1] = img_array[:, :, 1]
        input_tensor[:, :, 2] = img_array[:, :, 2]
        
        # Leave channels 3-7 as zeros
        
        return np.expand_dims(input_tensor, axis=0)

def preprocess_v3(file_obj):
    """
    V3 Preprocessing:
    Case A: Standard Image -> Same as V2 Safe Mode (via legacy).
    Case B: Scientific Data (.npy) -> Pass-through.
    """
    filename = getattr(file_obj, 'name', '').lower()
    
    if filename.endswith('.npy'):
        try:
            if hasattr(file_obj, 'seek'):
                file_obj.seek(0)
            data = np.load(file_obj)
            if data.shape[-1] != 8:
                 raise ValueError(f"Expected 8 channels, got {data.shape[-1]}")
            return np.expand_dims(data.astype(np.float32), axis=0)
        except Exception as e:
            raise ValueError(f"Invalid .npy file: {e}")
            
    else:
        return preprocess_v2_legacy(file_obj)

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

def mitigate_shadows(input_tensor, mask):
    """
    Uses histogram matching to mitigate shadows.
    User Request: Make Clear/Fill regions look like Shadow regions.
    Target (Reference): Shadow Class (1)
    Source (To Modify): Clear/Fill Class (0)
    
    Input: input_tensor (1, 256, 256, 8) - Preprocessed data
    """
    # 1. Extract RGB and Convert to uint8 (0-255)
    data = input_tensor[0]
    rgb = data[:, :, :3]
    
    _min = np.min(rgb)
    _max = np.max(rgb)
    
    if _max > _min:
        rgb_norm = (rgb - _min) / (_max - _min)
    else:
        rgb_norm = rgb
        
    img_array = (rgb_norm * 255).astype(np.uint8)
    
    # Create masks
    # Shadow is Class 1
    shadow_mask = (mask == 1)
    clear_mask = (mask == 0)
    
    # If we don't have both shadow and clear regions, we can't match them.
    if not np.any(shadow_mask) or not np.any(clear_mask):
        return img_array 

    mitigated = img_array.copy()
    
    try:
        # Histogram Matching: Match Clear (Source) -> Shadow (Reference)
        for c in range(3): # RGB
            clear_pixels = img_array[:, :, c][clear_mask]
            shadow_pixels = img_array[:, :, c][shadow_mask]
            
            if len(clear_pixels) > 0 and len(shadow_pixels) > 0:
                # 1. Compute CDF of source (Clear)
                src_values, src_unique_indices, src_counts = np.unique(clear_pixels, return_inverse=True, return_counts=True)
                src_cdf = np.cumsum(src_counts).astype(np.float64)
                src_cdf /= src_cdf[-1]
                
                # 2. Compute CDF of reference (Shadow)
                ref_values, ref_counts = np.unique(shadow_pixels, return_counts=True)
                ref_cdf = np.cumsum(ref_counts).astype(np.float64)
                ref_cdf /= ref_cdf[-1]
                
                # 3. Map source values to reference values
                interp_values = np.interp(src_cdf, ref_cdf, ref_values)
                
                # 4. Apply mapping
                matched_clear_pixels = interp_values[src_unique_indices]
                
                # Update the mitigated image: Modify CLEAR pixels to look like SHADOW
                mitigated[:, :, c][clear_mask] = matched_clear_pixels
                
    except Exception as e:
        print(f"Error in histogram matching: {e}")
        return img_array
            
    return mitigated

def generate_preview_image(input_tensor):
    """
    Generates a displayable RGB preview from the input tensor.
    Input: (1, 256, 256, 8) or similar.
    Output: Base64 encoded PNG string.
    """
    # Remove batch dimension: (256, 256, 8)
    data = input_tensor[0]
    
    # Extract RGB bands (Assuming 0, 1, 2 are R, G, B)
    rgb = data[:, :, :3]
    
    # Normalize for display (Min-Max Scaling)
    # This handles both [0,1] normalized data and standardized (centered) data
    _min = np.min(rgb)
    _max = np.max(rgb)
    
    if _max > _min:
        rgb_norm = (rgb - _min) / (_max - _min)
    else:
        rgb_norm = rgb # Avoid divide by zero, likely all zeros or constant
        
    # Convert to uint8 [0, 255]
    rgb_uint8 = (rgb_norm * 255).astype(np.uint8)
    
    return image_to_base64(rgb_uint8)

def image_to_base64(img_array):
    img = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
import cv2

def generate_solar_heatmap(mask):
    """
    Generates a Solar Potential Heatmap from the segmentation mask.
    Logic:
    - Clear (1) -> 100% Intensity (Red)
    - Thin Cloud (3) -> 50% Intensity (Yellow)
    - Shadow (2) / Thick (4) -> 10% Intensity (Blue)
    
    Note: The input 'mask' here is the remapped mask where:
    0: Clear (Wait, remap_classes says: 0: Clear, 1: Shadow, 2: Thin, 3: Thick)
    Let's align with remap_classes output:
    0: Clear -> High Potential (100%)
    1: Shadow -> Low Potential (10%)
    2: Thin -> Medium Potential (50%)
    3: Thick -> Low Potential (10%)
    """
    h, w = mask.shape
    heatmap = np.zeros((h, w), dtype=np.uint8)
    
    # Assign intensities (0-255)
    # Clear (0) -> 255 (High)
    heatmap[mask == 0] = 255
    # Thin Cloud (2) -> 127 (Medium)
    heatmap[mask == 2] = 127
    # Shadow (1) & Thick (3) -> 25 (Low)
    heatmap[mask == 1] = 25
    heatmap[mask == 3] = 25
    
    # Apply Color Map (Jet)
    # cv2.applyColorMap expects uint8 [0, 255]
    # It returns BGR, so convert to RGB
    colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    
    return image_to_base64(colormap_rgb)
