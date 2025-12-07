import cv2
import timm
import torch
import argparse
import numpy as np
from typing import Dict, List
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from vit_explain.vit_rollout import VITAttentionRollout
from vit_explain.vit_grad_rollout import VITAttentionGradRollout
from daam.DAAM import DynamicAccumulatedAttentionMap

from src.semscore_pipeline import SemScorePipeline
from src.processing_utils import create_selected_semantic_mask, semantic_spuriosity_score, select_target_layer

METHODS = {
    "daam": DynamicAccumulatedAttentionMap,
    "attention_grad_rollout": VITAttentionGradRollout,
    "attention_rollout": VITAttentionRollout,
}

class AttentionPipeline(SemScorePipeline):
    """Specialized pipeline for attention models."""
    
    def __init__(
        self, 
        config_path, 
        model_name,
        saliency_methods, 
        output_dir, 
        n_samples=None, 
        device='cuda'
    ):
        super().__init__(
            config_path=config_path,
            model_name=model_name,
            saliency_methods=saliency_methods,
            output_dir=output_dir,
            n_samples=n_samples,
            device=device,
            pipeline_type='attention'
        )

        # Load runs and pipeline specific configs
        self.gpu_id = self.config['gpu_id']
        for block in self.model.blocks:
            block.attn.fused_attn = False

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
        """Default sample processing for attention pipeline."""
        
        score = {}
        
        # Preprocess image
        self.screen_count += 1
        input_tensor = self.preprocess_image(sample_image)

        # Iterate through all saliency methods
        for saliency_method_name, saliency_method_class in self.saliency_methods.items():
            score[saliency_method_name] = {}
            if saliency_method_name == "attention_rollout":
                
                # Create attention map
                attention_rollout = saliency_method_class(self.model, head_fusion=self.head_fusion, discard_ratio=self.discard_ratio)
                input_tensor.requires_grad = True
                attention_map = attention_rollout(input_tensor)
                attention_map = cv2.resize(attention_map, (self.model_settings['image_size'], self.model_settings['image_size']))
                
                # Create semantic mask
                corrected_class_ids = [class_id + 1 for class_id in sample_class_ids] # Add one to account for background
                semantic_mask = create_selected_semantic_mask(sample_masks, corrected_class_ids).astype(np.uint8)
                semantic_mask = cv2.resize(semantic_mask, (self.model_settings['image_size'], self.model_settings['image_size']))

                # Calculate scores
                sss = semantic_spuriosity_score(attention_map, semantic_mask, threshold=self.threshold, hard_threshold=self.hard_threshold)
                if np.isnan(sss):
                    continue
                
                score[saliency_method_name] = {
                    "semantic_spuriosity_score": sss
                }
            
            elif saliency_method_name == "attention_grad_rollout":
                grad_rollout = saliency_method_class(self.model, discard_ratio=self.discard_ratio)
                for j in range(len(sample_class_ids)):
                    
                    # Create attention map
                    class_id = sample_class_ids[j]
                    
                    attention_map = grad_rollout(input_tensor, [class_id])
                    attention_map = cv2.resize(attention_map, (self.model_settings['image_size'], self.model_settings['image_size']))

                    # Create semantic mask
                    semantic_mask = create_selected_semantic_mask(sample_masks, [class_id + 1]).astype(np.uint8) # Add one to account for background
                    semantic_mask = cv2.resize(semantic_mask, (self.model_settings['image_size'], self.model_settings['image_size']))
                    
                    # Calculate scores
                    sss = semantic_spuriosity_score(attention_map, semantic_mask, threshold=self.threshold, hard_threshold=self.hard_threshold)
                    if np.isnan(sss):
                        continue
                    
                    score[saliency_method_name][int(class_id)] = {
                        "semantic_spuriosity_score": sss
                    }
                        
            elif saliency_method_name == "daam":
                # Instantiate DAAM
                target_block_list = []
                target_layers = select_target_layer(self.model, self.model_settings['target_layer'])
                daam = saliency_method_class(
                    model=self.model, 
                    target_layers=target_layers, 
                    block_layers=target_block_list,
                    arch_name=self.model_settings['model'], 
                    norm=self.daam_norm, 
                    gpu_id=self.gpu_id)                    
                
                for j in range(len(sample_class_ids)):
                    
                    # Create attention map
                    class_id = sample_class_ids[j]
                    target_labels = [ClassifierOutputTarget(class_id)]
                    cam_list, predicted_label, target_label = daam(input_tensor=input_tensor, target_label=target_labels)
                    attention_map = cv2.resize(cam_list[-1], (self.model_settings['image_size'], self.model_settings['image_size']))
                    
                    # Create semantic mask
                    semantic_mask = create_selected_semantic_mask(sample_masks, [class_id + 1]).astype(np.uint8) # Add one to account for background
                    semantic_mask = cv2.resize(semantic_mask, (self.model_settings['image_size'], self.model_settings['image_size']))
                    
                    # Calculate scores
                    sss = semantic_spuriosity_score(attention_map, semantic_mask, threshold=self.threshold, hard_threshold=self.hard_threshold)
                    if np.isnan(sss):
                        continue
                    
                    score[saliency_method_name][int(class_id)] = {
                        "semantic_spuriosity_score": sss
                    }
        
        return score

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="model_config/attention_config.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_name", nargs='+', type=str, default=["vit_base_augreg"])
    parser.add_argument("--output_dir", type=str, default="../outputs/attention")
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()
    
    for model_name in args.model_name:

        pipeline = AttentionPipeline(
            config_path=args.config,
            model_name=model_name,
            saliency_methods=METHODS,
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            device=args.device,
        )
        
        pipeline.run()
