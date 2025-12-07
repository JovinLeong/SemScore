import cv2
import torch
import argparse
import torchvision
import numpy as np
from typing import List, Dict, Any
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM, XGradCAM, EigenGradCAM, EigenCAM, AblationCAM

from src.semscore_pipeline import SemScorePipeline
from src.processing_utils import (
    create_selected_semantic_mask, 
    semantic_spuriosity_score, 
    resize_for_memory_efficiency, 
    select_target_layer
)

METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "layercam": LayerCAM,
}

class FasterRCNNBoxScoreTarget:
    def __init__(self, labels, bounding_boxes):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        
    def __call__(self, model_output):
        target_boxes = self.bounding_boxes
        target_labels = self.labels
        
        # Get scores from FasterRCNN outputs
        boxes = model_output['boxes']
        scores = model_output['scores']
        labels = model_output['labels']
        
        # Match target boxes to detected boxes (using IoU)
        matched_scores = []
        for target_box, target_label in zip(target_boxes, target_labels):
            # Find matching box with same label
            best_iou = 0
            best_score = None
            
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                if label == target_label:
                    # Calculate IoU
                    tb = target_box.detach().cpu().numpy()
                    b = box.detach().cpu().numpy()
                    
                    # IoU calculation
                    x_left = max(tb[0], b[0])
                    y_top = max(tb[1], b[1])
                    x_right = min(tb[2], b[2])
                    y_bottom = min(tb[3], b[3])
                    
                    if x_right < x_left or y_bottom < y_top:
                        continue
                        
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    tb_area = (tb[2] - tb[0]) * (tb[3] - tb[1])
                    b_area = (b[2] - b[0]) * (b[3] - b[1])
                    union = tb_area + b_area - intersection
                    
                    iou = intersection / union
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_score = score
            
            if best_score is not None:
                matched_scores.append(best_score)
        
        if len(matched_scores) > 0:
            # Ensure the scores require gradients
            scores_tensor = torch.stack(matched_scores)
            if not scores_tensor.requires_grad:
                scores_tensor.requires_grad_(True)
            return scores_tensor.sum()
        return torch.tensor(0.0, device=boxes.device, requires_grad=True)

class RetinaNetBoxScoreTarget:
    def __init__(self, labels, bounding_boxes):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        
    def __call__(self, model_output):
        target_boxes = self.bounding_boxes
        target_labels = self.labels
        
        # Get scores from RetinaNet outputs
        boxes = model_output['boxes']
        scores = model_output['scores']
        labels = model_output['labels']
        
        # Match target boxes to detected boxes (using IoU)
        matched_scores = []
        for target_box, target_label in zip(target_boxes, target_labels):
            # Find matching box with same label
            best_iou = 0
            best_score = None
            
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                if label == target_label:
                    # Calculate IoU
                    tb = target_box.detach().cpu().numpy()
                    b = box.detach().cpu().numpy()
                    
                    # IoU calculation
                    x_left = max(tb[0], b[0])
                    y_top = max(tb[1], b[1])
                    x_right = min(tb[2], b[2])
                    y_bottom = min(tb[3], b[3])
                    
                    if x_right < x_left or y_bottom < y_top:
                        continue
                        
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    tb_area = (tb[2] - tb[0]) * (tb[3] - tb[1])
                    b_area = (b[2] - b[0]) * (b[3] - b[1])
                    union = tb_area + b_area - intersection
                    
                    iou = intersection / union
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_score = score
            
            if best_score is not None:
                matched_scores.append(best_score)
        
        if len(matched_scores) > 0:
            # Ensure the scores require gradients
            scores_tensor = torch.stack(matched_scores)
            if not scores_tensor.requires_grad:
                scores_tensor.requires_grad_(True)
            return scores_tensor.sum()
        return torch.tensor(0.0, device=boxes.device, requires_grad=True)

class FCOSBoxScoreTarget:
    def __init__(self, labels, bounding_boxes):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        
    def __call__(self, model_output):
        target_boxes = self.bounding_boxes
        target_labels = self.labels
        
        # Get scores from FCOS outputs
        boxes = model_output['boxes']
        scores = model_output['scores']
        labels = model_output['labels']
        
        # Match target boxes to detected boxes (using IoU)
        matched_scores = []
        for target_box, target_label in zip(target_boxes, target_labels):
            # Find matching box with same label
            best_iou = 0
            best_score = None
            
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                if label == target_label:
                    # Calculate IoU
                    tb = target_box.detach().cpu().numpy()
                    b = box.detach().cpu().numpy()
                    
                    # IoU calculation
                    x_left = max(tb[0], b[0])
                    y_top = max(tb[1], b[1])
                    x_right = min(tb[2], b[2])
                    y_bottom = min(tb[3], b[3])
                    
                    if x_right < x_left or y_bottom < y_top:
                        continue
                        
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    tb_area = (tb[2] - tb[0]) * (tb[3] - tb[1])
                    b_area = (b[2] - b[0]) * (b[3] - b[1])
                    union = tb_area + b_area - intersection
                    
                    iou = intersection / union
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_score = score
            
            if best_score is not None:
                matched_scores.append(best_score)
        
        if len(matched_scores) > 0:
            # Ensure the scores require gradients
            scores_tensor = torch.stack(matched_scores)
            if not scores_tensor.requires_grad:
                scores_tensor.requires_grad_(True)
            return scores_tensor.sum()
        return torch.tensor(0.0, device=boxes.device, requires_grad=True)

class SSDBoxScoreTarget:
    def __init__(self, labels, bounding_boxes):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        
    def __call__(self, model_output):
        target_boxes = self.bounding_boxes
        target_labels = self.labels
        
        # For SSD, the model_output here is before NMS processing
        # We need to convert the raw output to processed detections to match our format
        scores = model_output['scores']
        boxes = model_output['boxes']
        labels = model_output['labels']
        
        # Match target boxes to detected boxes
        matched_scores = []
        for target_box, target_label in zip(target_boxes, target_labels):
            # Find matching box with same label
            best_iou = 0
            best_score = None
            
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                if label == target_label:
                    # Calculate IoU
                    tb = target_box.detach().cpu().numpy()
                    b = box.detach().cpu().numpy()
                    
                    # IoU calculation
                    x_left = max(tb[0], b[0])
                    y_top = max(tb[1], b[1])
                    x_right = min(tb[2], b[2])
                    y_bottom = min(tb[3], b[3])
                    
                    if x_right < x_left or y_bottom < y_top:
                        continue
                        
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    tb_area = (tb[2] - tb[0]) * (tb[3] - tb[1])
                    b_area = (b[2] - b[0]) * (b[3] - b[1])
                    union = tb_area + b_area - intersection
                    
                    iou = intersection / union
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_score = score
            
            if best_score is not None:
                matched_scores.append(best_score)
        
        if len(matched_scores) > 0:
            # Ensure the scores require gradients
            scores_tensor = torch.stack(matched_scores)
            if not scores_tensor.requires_grad:
                scores_tensor.requires_grad_(True)
            return scores_tensor.sum()
        return torch.tensor(0.0, device=boxes.device, requires_grad=True)

class ObjectDetectionPipeline(SemScorePipeline):
    """Specialized pipeline for attention models."""
    
    def __init__(self, config_path, model_name,saliency_methods,
                 output_dir, n_samples=None, device='cuda'):
        super().__init__(
            config_path=config_path,
            model_name=model_name,
            saliency_methods=saliency_methods,
            output_dir=output_dir,
            n_samples=n_samples,
            device=device,
            pipeline_type='object_detection'
        )
        self.max_image_size = self.config['max_image_size']
        self.ratio_channels_to_ablate = self.config['ratio_channels_to_ablate']

    def load_model(self, model_settings, device = 'cuda'):
        model_name = model_settings['model']
        pretrained = self.config['pretrained']
        if model_name == "fasterrcnn_resnet50_fpn":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
            
        elif model_name == "retinanet_resnet50_fpn":
            model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=pretrained)
            
        elif model_name == "fcos_resnet50_fpn":
            model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=pretrained)
            
        elif model_name == "ssd300_vgg16":
            model = torchvision.models.detection.ssd300_vgg16(pretrained=pretrained)
            
        elif model_name == "ssdlite320_mobilenet_v3_large":
            model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=pretrained)
            
        elif model_name == "yolov5s":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=pretrained)
            
        elif model_name == "yolov5s6":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained=pretrained)
            
        elif model_name == "yolov5m":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=pretrained)
            
        elif model_name == "yolov5m6":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5m6', pretrained=pretrained)
            
        else:
            raise ValueError(f"Model {model_name} not supported yet")
            
        # Move model to device
        model = model.to(self.device)
        
        # Enable gradients for all model parameters
        for param in model.parameters():
            param.requires_grad = True
            
        # Set model to eval mode by default
        model.eval()
        return model

    def _initialize_cam(self, saliency_method_class: Any) -> Any:
        """Initialize CAM instance with appropriate parameters."""
        
        # Get target layers
        target_layers = [select_target_layer(self.model, self.model_settings['target_layer'])]
        cam_args = {
            'model': self.model,
            'target_layers': target_layers,
        }
        self.gradient_required = True

        # if str(saliency_method_class) in ("LayerCAM", "AblationCAM", "EigenCAM"):
        #     self.gradient_required = False

        # Special handling for AblationCAM
        if 'AblationCAM' in str(saliency_method_class):
            cam_args['ratio_channels_to_ablate'] = self.ratio_channels_to_ablate
        
        return saliency_method_class(**cam_args)
    
    def _detect_objects(self, input_tensor: torch.Tensor, min_confidence: float = 0.5) -> dict:
        """
        Perform object detection on the input image
        
        Args:
            input_tensor: Preprocessed image tensor
            min_confidence: Minimum confidence score for detections
            
        Returns:
            Dict with filtered detection results
        """
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
        # Extract results
        boxes = outputs[0]['boxes']
        labels = outputs[0]['labels']
        scores = outputs[0]['scores']
        
        # Filter by confidence
        indices = scores > min_confidence
        filtered_boxes = boxes[indices]
        filtered_labels = labels[indices]
        filtered_scores = scores[indices]
        
        return {
            'boxes': filtered_boxes,
            'labels': filtered_labels,
            'scores': filtered_scores,
            'raw_outputs': outputs[0]
        }

    def generate_gradcam_for_detections(
        self,
        input_tensor: torch.Tensor,
        detections: dict,
        saliency_method_class: Any,
        top_k: int = 3,
        visualize_all: bool = False
    ) -> dict:
        """
        Generate CAM for multiple detected objects
        
        Args:
            input_tensor: Preprocessed image tensor
            detections: Detection results from detect_objects
            saliency_method_class: CAM method class to use
            top_k: Number of top detections to process
            visualize_all: Whether to process all detections
            
        Returns:
            Dictionary with individual and combined results
        """
        if len(detections['boxes']) == 0:
            print("No objects detected")
            return {'individual': {}, 'combined': None, 'combined_with_boxes': None, 'grayscale_maps': []}
        
        # Get the dimensions of the input tensor (which may be resized)
        _, _, h, w = input_tensor.shape
        print(f"Input tensor dimensions: {h}x{w}")
        
        # Determine how many objects to process
        if visualize_all:
            num_objects = len(detections['boxes'])
            print(f"Generating CAM for all {num_objects} detected objects")
        else:
            num_objects = min(top_k, len(detections['boxes']))
            print(f"Generating CAM for top {num_objects} objects (of {len(detections['boxes'])} detected) with top_k: {top_k}")

        # Ensure input tensor requires gradients and is on the correct device
        input_tensor = input_tensor.to(self.device)
        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True)
        
        # Generate CAM for each selected object
        individual_results = {}
        grayscale_maps = []

        for i in range(num_objects):
            try:
                print(f"\nProcessing object {i+1}/{num_objects}")
                
                # Create a target for this object
                if self.model_settings['model'] == "fasterrcnn_resnet50_fpn":
                    target = FasterRCNNBoxScoreTarget(
                        labels=detections['labels'][i: i + 1],
                        bounding_boxes=detections['boxes'][i: i + 1]
                    )
                elif self.model_settings['model'] == "retinanet_resnet50_fpn":
                    target = RetinaNetBoxScoreTarget(
                        labels=detections['labels'][i: i + 1],
                        bounding_boxes=detections['boxes'][i: i + 1]
                    )
                elif self.model_settings['model'] == "fcos_resnet50_fpn":
                    target = FCOSBoxScoreTarget(
                        labels=detections['labels'][i: i + 1],
                        bounding_boxes=detections['boxes'][i: i + 1]
                    )
                elif self.model_settings['model'] == "ssd300_vgg16" or self.model_settings['model'] == "ssdlite320_mobilenet_v3_large":
                    target = SSDBoxScoreTarget(
                        labels=detections['labels'][i: i + 1],
                        bounding_boxes=detections['boxes'][i: i + 1]
                    )

                cam = self._initialize_cam(saliency_method_class)
                # Generate grayscale CAM with gradients enabled
                grayscale_cam = cam(input_tensor=input_tensor, targets=[target])
                    
                
                if grayscale_cam is None or len(grayscale_cam) == 0:
                    print(f"Warning: No CAM generated for object {i+1}")
                    continue
                    
                grayscale_map = grayscale_cam[0, :]  # Extract saliency for the first image
                
                # Resize grayscale map to match input tensor dimensions if needed
                if grayscale_map.shape != (h, w):
                    grayscale_map = cv2.resize(grayscale_map, (w, h))
                    print(f"Resized grayscale map to: {grayscale_map.shape}")
                
                # Store the grayscale map for later combination
                grayscale_maps.append(grayscale_map)
                
            except Exception as e:
                print(f"Error generating CAM for object {i+1}: {str(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
        
        # Combine saliency maps if requested and if we have any maps
        combined_result = np.maximum.reduce(grayscale_maps)        
        return {
            'individual': individual_results,
            'combined': combined_result,  # Pure overlay without boxes
            'grayscale_maps': grayscale_maps
        }

    def process_image(
        self,
        input_tensor,
        saliency_method_class,
        max_image_size,
        top_k,
        min_confidence,
        visualize_all=False
    ):
        """Process a single image with the given model and save results"""
        try:
            
            # Store original dimensions
            original_h, original_w = input_tensor.shape[:2]
            
            # Resize image if needed for memory efficiency
            input_tensor, rescale_boxes_fn = resize_for_memory_efficiency(input_tensor, input_tensor, max_image_size)

            # Detect objects
            detections = self._detect_objects(input_tensor, min_confidence=min_confidence)
            print(f"Number of detections: {len(detections['boxes'])}")
            
            # Rescale boxes if image was resized
            if max(original_h, original_w) > max_image_size:
                detections['boxes'] = rescale_boxes_fn(detections['boxes'])
            
            # Skip if no detections
            if len(detections['boxes']) == 0:
                return None, None
            
            # Generate CAM visualizations for detected objects
            print("\nGenerating CAM visualizations...")
            cam_results = self.generate_gradcam_for_detections(
                input_tensor, 
                detections, 
                saliency_method_class,
                top_k=top_k,
                visualize_all=visualize_all,
            )

            # Return combined CAM and labels
            return cam_results['combined'], detections['labels'].tolist()
                
        except Exception as e:
            print(f"\nError processing image: {e}")

    def get_sss(
        self,
        sample_idx: int,
        sample_image: torch.Tensor,
        sample_masks: torch.Tensor,
        sample_class_ids: List[int],
        sample_class_labels: List[str]
    ) -> Dict:
        """Default sample processing for attention pipeline."""
        
        score = {}
        
        # Preprocess image
        self.screen_count += 1
        input_tensor = self.preprocess_image(sample_image)

        # Iterate through all saliency methods
        for saliency_method_name, saliency_method_class in self.saliency_methods.items():
            # Set cam method
            score[saliency_method_name] = {}
            grayscale_cam, detected_classes = self.process_image(
                input_tensor=input_tensor,
                saliency_method_class=saliency_method_class,
                max_image_size=self.max_image_size,
                top_k=self.top_k,
                min_confidence=self.min_confidence
            )
            
            if not detected_classes:
                print("No valid classes to predict: ", sample_idx)
                continue
            
            # Create semantic mask            
            semantic_mask = create_selected_semantic_mask(sample_masks, np.unique(detected_classes)).astype(np.uint8)
            semantic_mask = cv2.resize(semantic_mask, (self.model_settings['image_size'], self.model_settings['image_size']))
            
            # Calculate semscore
            sss = semantic_spuriosity_score(grayscale_cam, semantic_mask, threshold=self.threshold, hard_threshold=self.hard_threshold)
        
            if np.isnan(sss):
                continue
            
            # Calculate scores
            score[saliency_method_name] = {
                "semantic_spuriosity_score": sss
            }
        
        return score

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="model_config/object_detection_config.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_name", nargs='+', type=str, default=["fasterrcnn", "retinanet", "fcos", "ssd300", "ssdlite320"])
    parser.add_argument("--output_dir", type=str, default="../outputs/object_detection")
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()
    
    for model_name in args.model_name:
        print(model_name)
        pipeline = ObjectDetectionPipeline(
            config_path=args.config,
            model_name=model_name,
            saliency_methods=METHODS,
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            device=args.device,
        )
        
        pipeline.run()

