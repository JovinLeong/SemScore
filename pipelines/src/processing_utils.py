import os
import cv2
import torch
import psutil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta

def get_timestamp():
    '''Get timestamp in YYYYMMDDHHMMSS format'''
    return (datetime.now(tz=timezone(timedelta(hours=8)))).strftime("%Y%m%d%H%M%S")

def convert_to_serializable(object):
    '''Convert numpy types to native python types for JSON serialization'''
    if isinstance(object, (np.integer,)):
        return int(object)
    elif isinstance(object, (np.floating,)):
        return float(object)
    elif isinstance(object, (np.ndarray,)):
        return object.tolist()
    else:
        raise TypeError(f"Object of type {type(object)} is not JSON serializable")

def get_class_mask(sample_masks, class_id):
    '''Get mask for a specific class id'''
    return (sample_masks == class_id)

def visualise_image(sample_image):
    '''Visualise image'''
    sample_image = sample_image.permute(1, 2, 0)
    plt.imshow(sample_image)
    plt.axis('off')
    plt.show()
    return None

def visualise_class_mask(dataloader, masks, class_id):
    plt.imshow(get_class_mask(masks, class_id), cmap='gray')
    plt.axis('off')
    plt.title(f"Mask class: {dataloader.class_index_key_to_object_key(class_id)}")
    plt.show()
    return None

def create_selected_semantic_mask(all_semantic_masks, class_ids):
    '''Select out semantic regions for selected classes'''
    # Instantiate mask
    semantic_mask = np.zeros((all_semantic_masks.shape[0], all_semantic_masks.shape[1]), dtype=np.uint8)

    # Iterate across classes and set semantic mask to 1 for class regions
    for class_id in class_ids:
        semantic_mask = np.add(semantic_mask, np.where(all_semantic_masks == class_id, 1, 0))
    
    return semantic_mask

def report_cpu():
    pid = os.getpid()
    mem = psutil.Process(pid).memory_info().rss / 1024 ** 2
    print(f"[CPU] Memory usage: {mem:.2f} MB")

def report_gpu():
    if torch.cuda.is_available():
        print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"[GPU] Reserved : {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def semantic_spuriosity_score(saliency_map, semantic_mask, threshold=None, hard_threshold=False):
    '''Calculate semantic spuriosity score and allow for thresholding'''
    
    # Determine if values need to be thresholded and hard/soft
    if threshold and hard_threshold:
        saliency_map = np.where(saliency_map > threshold, 1, 0).copy()
    elif threshold and not hard_threshold:
        saliency_map = np.where(saliency_map > threshold, saliency_map, 0).copy()
        
    # Create boolean mask and determine which values are inside and outside the mask
    boolean_mask = semantic_mask.astype(bool).copy()
    saliency_map_in = np.sum(saliency_map[boolean_mask]).astype(np.int64)
    saliency_map_out = np.sum(saliency_map[~boolean_mask]).astype(np.int64)
    
    # Calculate score and scale to [0, 1] and invert
    score = (saliency_map_in - saliency_map_out) / np.sum(saliency_map)
    scaled_score = 1 - (score + 1) / 2
    return scaled_score

def semantic_relevance_score(saliency_map, semantic_mask, threshold=None, hard_threshold=False):
    '''Calculate semantic relevance score and allow for thresholding'''
    
    # Determine if values need to be thresholded and hard/soft
    if threshold and hard_threshold:
        saliency_map = np.where(saliency_map > threshold, 1, 0).copy()
    elif threshold and not hard_threshold:
        saliency_map = np.where(saliency_map > threshold, saliency_map, 0).copy()

    # Get boolean mask and etermine overlap ratio
    boolean_mask = semantic_mask.astype(bool).copy()
    saliency_map_in = np.sum(saliency_map[boolean_mask])
    score = np.sum(saliency_map_in) / np.sum(saliency_map)
    return score, saliency_map

def make_reshape_transform(height, width, hierarchical=False):
    
    if hierarchical:
        print("Using hierarchical reshape transform")
        def reshape_transform(tensor, height=height, width=width):
            result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
            return result.transpose(2, 3).transpose(1, 2)
    else:
        def reshape_transform(tensor, height=height, width=width):
            '''From pytorch-grad-cam example'''
            result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
            return result.transpose(2, 3).transpose(1, 2)
    
    return reshape_transform

# Function to resize images and masks for memory-intensive methods
def resize_image_and_masks(image_tensor, mask_tensor, target_size=(320, 320)):
    """
    Resize image and mask tensors to a smaller size to reduce memory usage
    
    Args:
        image_tensor: Original image tensor [C, H, W]
        mask_tensor: Original mask tensor [H, W]
        target_size: Target size (width, height) for resizing
        
    Returns:
        tuple: Resized image tensor, resized mask tensor
    """
    # Convert image tensor to numpy array for resizing
    image_np = image_tensor.permute(1, 2, 0).numpy()
    
    # Resize image
    resized_image = cv2.resize(image_np, target_size, interpolation=cv2.INTER_AREA)
    
    # Resize mask (nearest neighbor to preserve class IDs)
    mask_np = mask_tensor.numpy()
    resized_mask = cv2.resize(mask_np, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Convert back to tensors
    resized_image_tensor = torch.from_numpy(resized_image).permute(2, 0, 1)
    resized_mask_tensor = torch.from_numpy(resized_mask).long()
    
    return resized_image_tensor, resized_mask_tensor

def draw_boxes(boxes, labels, classes, image, colors):
    '''From pytorch-grad-cam example'''
    for i, box in enumerate(boxes):
        color = colors[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image

def resize_for_memory_efficiency(image, input_tensor, max_size=800):
    """
    Resize image and tensor if either dimension exceeds max_size
    Returns resized input tensor and a function to resize boxes back to original size
    """
    original_height, original_width = image.shape[:2]
    
    # Check if resizing is needed
    if max(original_height, original_width) <= max_size:
        # No resizing needed
        return input_tensor, lambda boxes: boxes
    
    # Calculate new dimensions
    if original_width > original_height:
        new_width = max_size
        new_height = int(original_height * (max_size / original_width))
    else:
        new_height = max_size
        new_width = int(original_width * (max_size / original_height))
    
    # Resize using torchvision's functional transforms
    from torchvision.transforms import functional as F
    resized_tensor = F.resize(input_tensor, [new_height, new_width])
    
    # Create a function to scale boxes back to original size
    def rescale_boxes(boxes):
        # Scale boxes back to original dimensions
        scale_x = original_width / new_width
        scale_y = original_height / new_height
        
        # Clone boxes to avoid modifying the original
        scaled_boxes = boxes.clone()
        
        # Scale coordinates
        scaled_boxes[:, 0] *= scale_x  # x1
        scaled_boxes[:, 1] *= scale_y  # y1
        scaled_boxes[:, 2] *= scale_x  # x2
        scaled_boxes[:, 3] *= scale_y  # y2
        
        return scaled_boxes
    
    print(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height} for memory efficiency")
    return resized_tensor, rescale_boxes

def vlm_prompt_from_class_labels(class_labels):
    return [f"{label}" for label in class_labels]

def determine_valid_classes(predictions, filter_coco_stuff, min_confidence):
            
    if filter_coco_stuff:
        return [i for i, (k, v) in enumerate(predictions.items()) if v > min_confidence and k < 92]
    else:
        return [i for i, (k, v) in enumerate(predictions.items()) if v > min_confidence]

def select_target_layer(model, target_layer: str) -> list:
    if target_layer == "attention":
        target_layers = [(block_no.attn, block_no.attn.proj) for no, block_no in enumerate(model.blocks[:])]
    elif target_layer == 'blocks[-1].norm1':
        target_layers = [model.blocks[-1].norm1]
    elif target_layer == 'blocks[-1].norm2':
        target_layers = [model.blocks[-1].norm2]
    elif target_layer == 'blocks[-1].mlp.fc':
        target_layers = [model.blocks[-1].mlp.fc]
    elif target_layer == 'blocks[-1].attn.proj':
        target_layers = [model.blocks[-1].attn.proj]
    elif target_layer == 'layers[-1].blocks[-1].norm2':
        target_layers = [model.layers[-1].blocks[-1].norm2]
    elif target_layer == 'stages[-1].blocks[-1].norm2':
        target_layers = [model.stages[-1].blocks[-1].norm2]
    elif target_layer == "backbone.body.layer4":
        target_layers = model.backbone.body.layer4
    elif target_layer == "backbone.features[-1]":
        target_layers = model.backbone.features[-1]
    elif target_layer == "backbone.features[-3]":
        target_layers = model.backbone.features[-3]
    elif target_layer == "model.model.model[-2]":
        target_layers = model.model.model.model[-2]
    elif target_layer == "clip.vision_model.encoder.layers[-1].layer_norm1":
        target_layers = [model.clip.vision_model.encoder.layers[-1].layer_norm1]
    else:
        raise ValueError(f"Target layer {target_layer} not supported for model {model}")
    return target_layers