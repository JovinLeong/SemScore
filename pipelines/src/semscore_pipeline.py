"""
SemScore Pipeline - A unified pipeline for running semantic spuriosity score experiments

This module provides a flexible pipeline class that handles:
- Model loading and configuration
- Data loading and preprocessing
- Saliency method execution (CAM, attention, etc.)
- Score calculation (semantic spuriosity, semantic relevance)
- Output management and periodic saving
- Memory management and resource reporting
"""

import os
import gc
import cv2
import json
import time
import yaml
import torch
import traceback

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
from pytorch_grad_cam.utils.image import preprocess_image
from processing_utils import (get_timestamp,
    convert_to_serializable,    
    report_cpu,
    report_gpu,
)

class SemScorePipeline(ABC):
    """
    Unified pipeline for semantic spuriosity score experiments.
    
    This class provides a flexible framework for running experiments across different
    model types (image classifiers, VLMs, object detectors) and saliency methods
    (GradCAM, attention, etc.).
    """
    
    def __init__(
        self,
        config_path: str,
        model_name: str,
        saliency_methods: Dict[str, Any],
        output_dir: str,
        pipeline_type: str = "image_classification",
        n_samples: Optional[int] = None,
        device: str = "cuda",
        preprocess_fn: Optional[Callable] = None,
        process_sample_fn: Optional[Callable] = None,
        calculate_score_fn: Optional[Callable] = None,
    ):
        """
        Initialize the SemScore pipeline.
        
        Args:
            config_path: Path to the configuration YAML file
            model_name: Name of the model configuration to use
            dataloader: Data loader instance (ImageNetSDataLoader, COCODataLoader, etc.)
            model: Model instance (can be None if using model_loader_fn)
            saliency_methods: Dictionary of saliency method names to method classes
            output_dir: Directory to save output files
            pipeline_type: Type of pipeline ("image_classification", "vlm", "object_detection", "attention")
            n_samples: Number of samples to process (None for all)
            device: Device to run on ("cuda" or "cpu")
        """
        self.config_path = config_path
        self.model_name = model_name
        self.saliency_methods = saliency_methods
        self.output_dir = output_dir
        self.pipeline_type = pipeline_type
        self.device = device
        
        # Load config
        self.config, self.model_settings, self.dataset = self._load_config(config_path, model_name)
        
        # Load model
        self.model = self.load_model(self.model_settings, device=self.device)

        # Set dataloader to desired dataset
        self.dataloader = self.get_dataloader()

        # Set run data
        self.screen_count = 0
        self.now = time.time()

        # Set num samples
        self.n_samples = n_samples if n_samples is not None else self.dataloader.get_n_samples()
        
        self.threshold = self.config["threshold"]
        self.hard_threshold = self.config["hard_threshold"]
        self.min_confidence = self.config["min_confidence"]
        self.filter_coco_stuff = self.config["filter_coco_stuff"]
        self.eigen_smooth = self.config["eigen_smooth"]
        self.aug_smooth = self.config["aug_smooth"]
        self.top_k = self.config["top_k"]
        self.save_interval = self.config["save_interval"]
        self.report_memory = self.config["report_memory"]
        self.discard_ratio = self.config["discard_ratio"]
        self.head_fusion = self.config["head_fusion"]
        self.daam_norm = self.config["daam_norm"]
        
        # Custom preprocessing and scoring functions
        self.preprocess_fn = preprocess_fn
        self.process_sample_fn = process_sample_fn
        self.calculate_score_fn = calculate_score_fn
        
        # Initialize output structure
        self.start_time = time.time()
        self.outputs = self._initialize_outputs()
        
    @abstractmethod
    def load_model(self, model_settings, device):
        """Child classes must implement this."""
        raise NotImplementedError        

    def get_dataloader(self):
        dataset_name = self.dataset

        dataloader = None
        if dataset_name == "imagenet-s":
            from data_loaders import ImageNetSDataLoader
            dataloader = ImageNetSDataLoader()
        elif dataset_name == "coco-s":
            from data_loaders import COCODataLoader
            dataloader = COCODataLoader()
        else:
            raise ValueError(f"Unknown dataset name '{dataset_name}' in config")
        
        return dataloader
        
    def _load_config(self, config_path: str, model_name: str):
        """Load configuration from YAML with pipeline + model sections."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        pipeline_params = config["pipeline_params"]
        model_configs = config["models"]
        dataset = config["dataset"]
        if model_name not in model_configs:
            raise ValueError(f"Model '{model_name}' not found in YAML under 'model' key.")

        model_settings = model_configs[model_name]

        # Merge configs (model_settings override pipeline_params)
        merged_config = {**pipeline_params, **model_settings}
        return merged_config, model_settings, dataset
    
    def process_sample(
            self,
            sample_idx: int,
            sample_image: torch.Tensor,
            sample_masks: torch.Tensor,
            sample_class_ids: List[int],
            sample_class_labels: List[str]
        ) -> Dict:
            """
            Process a single sample.
            
            Args:
                sample_idx: Sample index
                sample_image: Raw image tensor
                sample_masks: Semantic masks
                sample_class_ids: List of class IDs in the sample
                sample_class_labels: List of class labels in the sample
                
            Returns:
                Dictionary of scores for this sample
            """
            # Check if the user has provided a custom function for processing a sample
            # Allows users to override the default sample processing logic by providing own function
            if self.process_sample_fn is not None:
                # If a custom function is provided, delegate the sample processing to it
                # Pass the sample index, image, masks, class IDs, class labels, and the pipeline instance itself
                # This allows the user-defined function to access the pipeline's configuration and helper methods
                return self.process_sample_fn(
                    sample_idx,         # Index of the current sample being processed
                    sample_image,       # Raw image tensor for this sample
                    sample_masks,       # Semantic masks corresponding to the image
                    sample_class_ids,   # List of class IDs present in this sample
                    sample_class_labels,# List of class labels corresponding to the class IDs
                    self                # Pass the pipeline instance to allow access to its methods/attributes
                )
            
            # Default processing for image classification
            return self.get_sss(
                sample_idx,
                sample_image,
                sample_masks,
                sample_class_ids,
                sample_class_labels
            )


    def _initialize_outputs(self) -> Dict:
        """Initialize the output dictionary structure."""
        metadata = {
            "n_samples": self.n_samples,
            "threshold": self.threshold,
            "hard_threshold": self.hard_threshold,
            "min_confidence": self.min_confidence,
            "filter_coco_stuff": self.filter_coco_stuff,
            "eigen_smooth": self.eigen_smooth,
            "aug_smooth": self.aug_smooth,
            "model_settings": self.model_settings,
            "time_taken": None,
            "pipeline_type": self.pipeline_type
        }
        
        # Add pipeline-specific metadata
        if self.pipeline_type == "vlm" or self.pipeline_type == "object_detection":
            metadata["top_k"] = self.top_k
        
        if self.pipeline_type == "attention":
            metadata["discard_ratio"] = self.discard_ratio
            metadata["head_fusion"] = self.head_fusion
            metadata["daam_norm"] = self.daam_norm
        
        return {
            "metadata": metadata,
            "scores": {}
        }
    
    def _save_outputs(self, suffix: str = ""):
        """
        Save outputs to JSON file.
        
        Args:
            suffix: Optional suffix for the filename (e.g., "_1000" for intermediate saves)
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate filename
        model_name_safe = self.model_settings['model'].replace('/', '_')
        timestamp = get_timestamp()
        filename = f"semscore_{model_name_safe}_{timestamp}_{suffix}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        self.outputs['metadata']['time_taken'] = int(time.time() - self.start_time)
        
        with open(filepath, 'w') as f:
            json.dump(self.outputs, f, indent=4, default=convert_to_serializable)
        
        print(f"Saved outputs to: {filepath}")
        return None
    
    def _report_memory(self):
        """Report CPU and GPU memory usage if enabled."""
        if self.report_memory:
            report_cpu()
            report_gpu()
        return None
    
    def _cleanup_memory(self):
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return None

    def preprocess_image(self, sample_image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            sample_image: Raw image tensor
            
        Returns:
            Preprocessed image tensor
        """
        if self.preprocess_fn is not None:
            return self.preprocess_fn(sample_image, self.model_settings)
        
        # Default preprocessing
        rgb_img = sample_image.permute(1, 2, 0).numpy()
        rgb_img = cv2.resize(rgb_img, (self.model_settings['image_size'], self.model_settings['image_size']))
        input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(self.device)
        
        return input_tensor
    
    @abstractmethod
    def get_sss(
        self,
        sample_idx: int,
        sample_image: torch.Tensor,
        sample_masks: torch.Tensor,
        sample_class_ids: List[int],
        sample_class_labels: List[str]
    ) -> Dict:
        """
        Abstract method for processing sample for desired pipeline.
        Must be implemented in the child class.
        """
        ...
    
    def run(self):
        """
        Execute the complete pipeline.
        
        This is the main entry point that runs the experiment across all samples.
        """
        print(f"Starting SemScore Pipeline: {self.pipeline_type}")
        print(f"Model: {self.model_settings['model']}")
        print(f"Processing {self.n_samples} samples")
        print(f"Output directory: {self.output_dir}")
        print(f"Saliency methods: {list(self.saliency_methods.keys())}")
        print("-" * 80)
        
        # Iterate through samples
        for i in range(self.n_samples):
            self._report_memory()
            
            print(f"\nSample {i}/{self.n_samples}")
            
            # Get sample data
            sample_image, sample_masks, sample_class_ids, sample_class_labels = self.dataloader.get_sample(i)
            
            # Check if sample is valid
            if not sample_class_ids or sample_image is None:
                self.outputs["scores"][i] = "No valid classes to predict"
                continue
            
            try:
                score = self.process_sample(
                    i,
                    sample_image,
                    sample_masks,
                    sample_class_ids,
                    sample_class_labels
                )
                
                self.outputs["scores"][i] = score
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                traceback.print_exc()
                self.outputs["scores"][i] = f"Error: {str(e)}"
            
            self._cleanup_memory()
            
            if i > 0 and i % self.save_interval == 0:
                self._save_outputs(suffix=f"_{i}")
        
        self.outputs['screening_rate'] = self.screen_count / self.n_samples
        self.outputs['time_taken'] = int(time.time() - self.now)

        self._save_outputs()
        
        print("\n" + "=" * 80)
        print(f"Pipeline completed!")
        print(f"Total time: {int(time.time() - self.start_time)} seconds")
        print(f"Processed {self.n_samples} samples")
        print("=" * 80)
        
        return self.outputs

class VLMPipeline(SemScorePipeline):
    """Specialized pipeline for Vision-Language Models (VLMs)."""
    
    def __init__(self, config_path, model_name, dataloader, model, saliency_methods,
                 output_dir, n_samples=None, device='cuda'):
        super().__init__(
            config_path=config_path,
            model_name=model_name,
            dataloader=dataloader,
            model=model,
            saliency_methods=saliency_methods,
            output_dir=output_dir,
            n_samples=n_samples,
            device=device,
            pipeline_type='vlm'
        )
