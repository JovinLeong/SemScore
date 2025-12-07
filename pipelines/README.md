# Pipelines

This directory contains all the pipelines used in the paper along with the base implementations of SemScore, custom dataloaders, and utility functions in `src`.

## Creating your own custom pipeline

### Dataloader

If your custom pipeline requires the use of an existing dataset for which a dataloader exists, then there is no need to define a custom dataloader. Else, you can access `src/data_loaders.py` and implement a custom dataloader class that extends the `DataLoader` base class to suit your dataset. Crucially, the custom dataloader should match the base class' interface to ensure that SemScore can correctly load data for processing.

### Pipeline

To create your own custom pipeline, create a new Python file similar to those of the predefined pipelines and extend the `SemScorePipeline` base class to appropriately load your models and run inferences with said models. The custom pipeline should match the base class' interface to ensure that the pipeline can properly run and generate scores which can align with the evaluation scripts.

To mirror the existing pipeline interface, your pipeline script should contain the same CLI commands that the existing pipeline scripts already implement.

### Configuration file

Create the appropriate configuration file similar to the existing configuration files in `model_config` and ensure that your models are correctly specified within the `model_config`.

## Predefined pipelines

The following pipelines were used with the specified configuration in the SemScore paper to obtain the experimental results; these pipelines can be extended for use with your own models or with custom tasks seeking to leverage SemScore.

### Image classification pipeline

To run the pipeline, you can either use the existing default configuration `image_classification_config.yaml` which was used in the SemScore paper or specify your own configuration as long as it contains all required fields.

To use models that are not covered by the existing configuration file, modify or extend `image_classification_pipeline.py` to accommodate your specific model.

Thereafter, once configuration is complete, run the pipeline with:

```bash
python image_classification_pipeline.py --config <path to config file> --model <name of target model>
```

e.g.

```bash
python image_classification_pipeline.py --config ./model_config/image_classification_config.yaml --model vit_base_augreg
```

### Object detection pipeline

To run the pipeline, you can either use the existing default configuration `object_detection_config.yaml` which was used in the SemScore paper or specify your own configuration as long as it contains all required fields.

To use models that are not covered by the existing configuration file, modify or extend `object_detection_pipeline.py` to accommodate your specific model.

Thereafter, once configuration is complete, run the pipeline with:

```bash
python object_detection_pipeline.py --config <path to config file> --model <name of target model>
```

e.g.

```bash
python object_detection_pipeline.py --config ./model_config/object_detection_config.yaml --model vit_base_augreg
```

### VLM pipeline

To run the pipeline, you can either use the existing default configuration `vlm_config.yaml` which was used in the SemScore paper or specify your own configuration as long as it contains all required fields.

To use models that are not covered by the existing configuration file, modify or extend `vlm_pipeline.py` to accommodate your specific model.

Thereafter, once configuration is complete, run the pipeline with:

```bash
python vlm_pipeline.py --config <path to config file> --model <name of target model>
```

e.g.

```bash
python vlm_pipeline.py --config ./model_config/vlm_config.yaml --model vit_base_augreg
```

### Attention pipeline

To run the pipeline, you can either use the existing default configuration `attention_config.yaml` which was used in the SemScore paper or specify your own configuration as long as it contains all required fields.

To use models that are not covered by the existing configuration file, modify or extend `attention_pipeline.py` to accommodate your specific model.

Thereafter, once configuration is complete, run the pipeline with:

```bash
python attention_pipeline.py --config <path to config file> --model <name of target model>
```

e.g.

```bash
python attention_pipeline.py --config ./model_config/attention_config.yaml --model vit_base_augreg
```
