import cv2
import timm
import torch
import argparse
import numpy as np
from typing import Dict, List, Any
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
    LayerCAM
)

from src.semscore_pipeline import SemScorePipeline
from src.processing_utils import (
    create_selected_semantic_mask,
    semantic_spuriosity_score,
    make_reshape_transform,
    select_target_layer
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
}

class ImageClassifierPipeline(SemScorePipeline):
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
            pipeline_type='image_classification'
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
        if model_settings['loader'] == 'torchhub':
            model = torch.hub.load(
                model_settings['repo'],
                model_settings['model'],
                pretrained=True
            ).to(torch.device(device)).eval()
        elif model_settings['loader'] == 'timm':
            model = timm.create_model(
                model_settings['model'],
                pretrained=True
            ).to(torch.device(device)).eval()
        else:
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
        
        # Iterate through all saliency methods
        for saliency_method_name, saliency_method_class in self.saliency_methods.items():
            score[saliency_method_name] = {}
            
            # Initialize CAM
            cam = self._initialize_cam(saliency_method_class)
            
            # Process each class
            for j in range(len(sample_class_ids)):
                cam.batch_size = 32
                grayscale_cam = cam(
                    input_tensor=input_tensor,
                    targets=[ClassifierOutputTarget(sample_class_ids[j])],
                    eigen_smooth=self.eigen_smooth,
                    aug_smooth=self.aug_smooth
                )
                
                grayscale_cam = grayscale_cam[0, :]
                
                # Create semantic mask
                class_id = int(sample_class_ids[j] + 1)
                semantic_mask = create_selected_semantic_mask(sample_masks, [class_id]).astype(np.uint8)
                semantic_mask = cv2.resize(semantic_mask, (self.model_settings['image_size'], self.model_settings['image_size']))
                
                # Calculate SSS
                sss = semantic_spuriosity_score(grayscale_cam, semantic_mask, threshold=self.threshold, hard_threshold=self.hard_threshold)
                
                if np.isnan(sss):
                    continue
                
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
    parser.add_argument("--config", type=str, default="model_config/image_classification_config.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_name", nargs='+', type=str, default=["vit_base_augreg"])
    parser.add_argument("--output_dir", type=str, default="../outputs/image_classification")
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()
    
    for model_name in args.model_name:

        pipeline = ImageClassifierPipeline(
            config_path=args.config,
            model_name=model_name,
            saliency_methods=METHODS,
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            device=args.device,
        )
        
        pipeline.run()
