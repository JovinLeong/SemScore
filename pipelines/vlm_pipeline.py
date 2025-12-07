import cv2
import torch
import argparse
import numpy as np
from typing import Dict, List, Any
from transformers import CLIPProcessor, CLIPModel
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import (
    GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
)

from src.semscore_pipeline import SemScorePipeline
from src.processing_utils import (
    create_selected_semantic_mask,
    semantic_spuriosity_score,
    make_reshape_transform,
    select_target_layer,
    determine_valid_classes
)

METHODS = {
    "gradcam": GradCAM,
    "scorecam": ScoreCAM,
    "gradcam++": GradCAMPlusPlus,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "layercam": LayerCAM,
    "fullgrad": FullGrad
}

class ImageClassifier(torch.nn.Module):
    def __init__(self, model="openai/clip-vit-large-patch14", processor="openai/clip-vit-large-patch14"):
        super(ImageClassifier, self).__init__()
        self.clip = CLIPModel.from_pretrained(model)
        self.processor = CLIPProcessor.from_pretrained(processor)
        self.labels = None
        
    def set_labels(self, labels):
        self.labels = labels

    def forward(self, x):
        text_inputs = self.processor(text=self.labels, return_tensors="pt", padding=True)

        outputs = self.clip(pixel_values=x, input_ids=text_inputs['input_ids'].to(self.clip.device),
                            attention_mask=text_inputs['attention_mask'].to(self.clip.device))

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        return probs

class VLMPipeline(SemScorePipeline):
    """Specialized pipeline for image classification models."""
    
    def __init__(self, config_path, model_name, saliency_methods,
                 output_dir, n_samples=None, device='cuda'):
        super().__init__(
            config_path=config_path,
            model_name=model_name,
            saliency_methods=saliency_methods,
            output_dir=output_dir,
            n_samples=n_samples,
            device=device,
            pipeline_type='vlm'
        )

    def load_model(self, model_settings, device='cuda'):
        """
        Load model based on settings.
        
        Args:
            model_settings: Dictionary containing model configuration
            device: Device to load model on
            
        Returns:
            Loaded model
        """
        try:
            model = ImageClassifier(model=model_settings['model'], processor=model_settings['processor']).to(torch.device(device)).eval()

        except:
            raise NotImplementedError(f"Loader '{model_settings['loader']}' is not supported yet.")
        
        return model

    def get_sss(
        self,
        sample_idx: int,
        sample_image: torch.Tensor,
        sample_masks: torch.Tensor,
        sample_class_ids: List[int],
        sample_class_labels: List[str]
    ) -> Dict:
        """Default sample processing for image classification pipeline."""
        
        score = {}
        
        # Preprocess image
        self.screen_count += 1
        input_tensor = self.preprocess_image(sample_image)
        
        if not sample_class_ids:
            return "No valid classes to predict"
        
        # Initial pass to get top_k predictions with all labels
        self.model.set_labels(sample_class_labels)
        predictions = self.model.forward(input_tensor)
        score["all_predictions"] = dict(zip(sample_class_labels, predictions[0].tolist()))
        all_predictions = dict(zip(sample_class_ids, predictions[0].tolist()))
        
        # Determine valid indices to score; filter coco stuff and low confidence
        valid_label_indices = determine_valid_classes(all_predictions, self.filter_coco_stuff, self.min_confidence)
        if len(valid_label_indices) == 0:
            return "No valid classes to predict"
        
        # Determine if valid_label_indices exceed top_k
        if len(valid_label_indices) > self.top_k:
            
            # If so, then set valid_label_indices to top_k labels
            pred_values_list = list(all_predictions.values())
            valid_entries = [(i, pred_values_list[i]) for i in valid_label_indices]
            valid_label_indices = [i for i, _ in sorted(valid_entries, key=lambda x: x[1], reverse=True)[:self.top_k]]

        # Iterate through all saliency methods
        for saliency_method_name, saliency_method_class in self.saliency_methods.items():
            
            score[saliency_method_name] = {}
            # Initialize CAM
            cam = self._initialize_cam(saliency_method_class)
                        
            for valid_label_index in valid_label_indices:
                
                targets = [ClassifierOutputTarget(valid_label_index)]
                cam.batch_size = 32
                grayscale_cam = cam(input_tensor=input_tensor,
                                    targets=targets,
                                    eigen_smooth=self.eigen_smooth,
                                    aug_smooth=self.aug_smooth)
                
                grayscale_cam = grayscale_cam[0, :]
                
                # Create semantic mask
                class_id = self.dataloader.class_object_key_to_index_key(sample_class_labels[valid_label_index])
                semantic_mask = create_selected_semantic_mask(sample_masks, [class_id]).astype(np.uint8)
                semantic_mask = cv2.resize(semantic_mask, (self.model_settings['image_size'], self.model_settings['image_size']))
                
                sss = semantic_spuriosity_score(grayscale_cam, semantic_mask, threshold=self.threshold, hard_threshold=self.hard_threshold)

                if np.isnan(sss):
                    sss = None

                # Calculate scores
                score[saliency_method_name][class_id] = {
                    "semantic_spuriosity_score": sss
                }
        
        return score

    def _initialize_cam(self, saliency_method_class: Any) -> Any:
        """Initialize CAM instance with appropriate parameters."""
        
        # Get target layers
        target_layers = select_target_layer(self.model, self.model_settings['target_layer'])
        
        # Create reshape transform if needed
        if 'height' in self.model_settings and 'width' in self.model_settings:
            reshape_transform = make_reshape_transform(
                self.model_settings['height'],
                self.model_settings['width'],
                hierarchical=self.model_settings['hierarchical']
            )
        else:
            reshape_transform = None
        
        # Initialize CAM with appropriate params
        cam_args = {
            'model': self.model,
            'target_layers': target_layers,
        }
        
        if reshape_transform is not None:
            cam_args['reshape_transform'] = reshape_transform
        
        # Special handling for AblationCAM
        if 'AblationCAM' in str(saliency_method_class):
            cam_args['ablation_layer'] = AblationLayerVit()
        
        return saliency_method_class(**cam_args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="model_config/vlm_config.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_name", nargs='+', type=str, default=["clip-vit-base", 'TinyCLIP-ViT-8M', 'TinyCLIP-ViT-40M', 'TinyCLIP-ViT-61M', 'fashion-clip', 'DFN', 'metaclip', 'plip', 'clip-rsicd', 'QuiltNet', 'pubmed-clip'])
    parser.add_argument("--output_dir", type=str, default="../outputs/vlm")
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()
    
    for model_name in args.model_name:

        pipeline = VLMPipeline(
            config_path=args.config,
            model_name=model_name,
            saliency_methods=METHODS,
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            device=args.device,
        )
        
        pipeline.run()
