# Data

This directory contains instructions to download the datasets used for model evaluation using SemScore. Instructions to download the desired datasets can be found in their respective sections.

## ImageNet-S

In `ImageNet-S`, run the following command to download the [ImageNet-S dataset](https://github.com/LUSSeg/ImageNet-S):

```bash
chmod +x get_data.sh
bash get_data.sh val 919
```

## ImageNet-1K Val

In `ImageNetVal`, to download the [ImageNet-1K validation dataset](https://huggingface.co/datasets/mlx-vision/imagenet-1k) install huggingface hub to your Python environment:

```bash
pip install -r requirements.txt
```

Then run the script to download and extract the ImageNet-1K validation dataset; modify directories if required.

```bash
python get_data.py
```

## sssegmentation

In `sssegmentation`, run the following commands to obtain the respective dataset through [sssegmentation](https://github.com/SegmentationBLWX/sssegmentation).

### Downloading ADE20k

```bash
bash prepare_datasets.sh ade20k
```

### Downloading VOCdevkit (PascalVOC)

```bash
bash prepare_datasets.sh pascalvoc
```

### Downloading COCOStuff10k

```bash
bash prepare_datasets.sh cocostuff10k
```
