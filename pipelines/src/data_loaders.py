
import os
import cv2
import json
import torch
import torchvision
import numpy as np
from scipy.io import loadmat
from constants import VOC_CLASSES, IMAGENET_CLASSES

class DataLoader:
    def __init__(self, data_dir='../data/sssegmentation'):
        self.data_dir = data_dir
        self.index = 0        
        self.to_tensor = torchvision.transforms.ToTensor()

    def load_class_label_mapping(self):
        ...

    def load_sample_details(self):
        ...
        
    def get_sample(self, index=None):
        ...
        
    def reset_index(self):
        self.index = 0
        return None
        
    def load_image_as_tensor(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.to_tensor(image)
    
    def get_n_samples(self):
        '''Get number of samples'''
        return len(self.sample_details)
    
    def get_n_classes(self):
        return len(self.class_index_key)
    
    def get_all_classes(self):
        return list(self.class_object_key.keys())
    
    def class_index_key_to_object_key(self, index_key):
        return self.class_index_key[index_key]
    
    def class_object_key_to_index_key(self, object_key):
        return self.class_object_key[object_key]
    
    def load_mask_as_tensor(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return torch.as_tensor(mask, dtype=torch.int32)
    
    def get_class_ids(self, masks, ignore=None):        
        '''Get class ids from masks'''
        class_ids = masks.unique()
        if ignore is not None:
            class_ids = class_ids[class_ids != ignore].tolist()
        return class_ids

class COCODataLoader(DataLoader):
    def __init__(self, data_dir='../data/sssegmentation'):
        super().__init__(data_dir)
        # Initialize class label mappings and directories
        self.annotations_dir = f'{data_dir}/COCOStuff10k/annotations'
        self.images_dir = f'{data_dir}/COCOStuff10k/images'
        self.class_label_mapping_dir = f'{data_dir}/COCOStuff10k_key_value_pair.json'
        self.load_class_label_mapping()
        self.load_sample_details()

    def load_class_label_mapping(self):
        # Access metadata from COCO dataset
        with open(self.class_label_mapping_dir, 'r') as file:
            class_label_mapping = json.load(file)
        
        self.class_index_key = {int(k): v for k, v in class_label_mapping['index_key'].items()}
        self.class_object_key = class_label_mapping['object_key']
        return None
    
    def load_sample_details(self):
        
        # Get sample names
        sample_names = [f.replace(".jpg", "") for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
        sample_details = {}
        
        # Iterate across samples
        for i in range(len(sample_names)):
            
            # Get paths
            sample_name = sample_names[i]
            image_path = os.path.join(self.images_dir, f"{sample_name}.jpg")
            annotation_path = os.path.join(self.annotations_dir, f"{sample_name}.mat")
            
            # Store details
            sample_details[i] = {
                'image_path': image_path,
                'annotation_path': annotation_path,
                'sample_name': sample_name
            }
        
        self.sample_details = sample_details
        return self.sample_details

    def get_sample(self, index=None):
        
        # Get current index if not specified
        if index is None:
            index = self.index
            
        # Get sample details
        sample_details = self.sample_details[index]
        
        # Get mask, image, class id, label
        sample_image = self.load_image_as_tensor(sample_details['image_path'])
        sample_annotations = loadmat(sample_details['annotation_path'])
        sample_masks = torch.as_tensor(sample_annotations['S'], dtype=torch.int32)
        sample_class_ids = self.get_class_ids(sample_masks, ignore=0)
        sample_class_labels = [self.class_index_key_to_object_key(class_id) for class_id in sample_class_ids]
        return sample_image, sample_masks, sample_class_ids, sample_class_labels

class PascalVOCDataLoader(DataLoader):
    def __init__(self, data_dir='../data/sssegmentation', image_set='train'):
        super().__init__(data_dir)
        
        # Initialize class label mappings and directories
        self.annotations_dir = f'{data_dir}/VOCdevkit/VOC2012/SegmentationClass'
        self.images_dir = f'{data_dir}/VOCdevkit/VOC2012/JPEGImages'
        self.image_sets_path = f'{data_dir}/VOCdevkit/VOC2012/ImageSets/Segmentation/{image_set}.txt'
        self.class_labels = VOC_CLASSES
        self.load_class_label_mapping()
        self.load_sample_details()

    def load_class_label_mapping(self):   
        self.class_index_key = {i: label for i, label in enumerate(self.class_labels)}
        self.class_object_key = {label: i for i, label in enumerate(self.class_labels)}
        return None
    
    def load_sample_details(self):
        # Get sample names        
        with open(self.image_sets_path, 'r') as f:
            sample_names = [line.strip() for line in f.readlines()]
        
        sample_details = {}
        
        # Iterate across samples
        for i in range(len(sample_names)):
            
            # Get paths
            sample_name = sample_names[i]
            image_path = os.path.join(self.images_dir, f"{sample_name}.jpg")
            annotation_path = os.path.join(self.annotations_dir, f"{sample_name}.png")
            
            # Store details
            sample_details[i] = {
                'image_path': image_path,
                'annotation_path': annotation_path,
                'sample_name': sample_name
            }
        
        self.sample_details = sample_details
        return self.sample_details

    def get_sample(self, index=None):
        # Get current index if not specified
        if index is None:
            index = self.index
            
        # Get sample details
        sample_details = self.sample_details[index]
        
        # Get mask, image, class id, label
        sample_image = self.load_image_as_tensor(sample_details['image_path'])
        sample_masks = self.load_mask_as_tensor(sample_details['annotation_path'])
        sample_class_ids = self.get_class_ids(sample_masks, ignore=255)
        sample_class_labels = [self.class_index_key_to_object_key(class_id) for class_id in sample_class_ids]
        return sample_image, sample_masks, sample_class_ids, sample_class_labels
        
class ADE20KDataLoader(DataLoader):
    def __init__(self, data_dir='../data/sssegmentation', dataset='training'):
        super().__init__(data_dir)
        
        # Initialize class label mappings and directories
        self.annotations_dir = f'{data_dir}/ADE20k/ADEChallengeData2016/annotations/{dataset}'
        self.images_dir = f'{data_dir}/ADE20k/ADEChallengeData2016/images/{dataset}'
        self.image_sets_path = f'{data_dir}/ADE20k/ADEChallengeData2016/images/{dataset}'
        self.class_label_mapping_dir = f'{data_dir}/ADE20k/ADEChallengeData2016/objectInfo150.txt'
        self.load_class_label_mapping()
        self.load_sample_details()

    def load_class_label_mapping(self):
        self.class_index_key = {}
        with open(self.class_label_mapping_dir, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                columns = line.strip().split('\t')
                if len(columns) >= 5:
                    idx = int(columns[0])
                    name = columns[4]
                    self.class_index_key[idx] = name
        self.class_object_key = {v: k for k, v in self.class_index_key.items()}
        return None
        
    def load_sample_details(self):
        
        # Get sample names        
        sample_names = [f.replace(".jpg", "") for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
        sample_details = {}
        
        # Iterate across samples
        for i in range(len(sample_names)):
            
            # Get paths
            sample_name = sample_names[i]
            image_path = os.path.join(self.images_dir, f"{sample_name}.jpg")
            annotation_path = os.path.join(self.annotations_dir, f"{sample_name}.png")
            
            # Store details
            sample_details[i] = {
                'image_path': image_path,
                'annotation_path': annotation_path,
                'sample_name': sample_name
            }
        
        self.sample_details = sample_details
        return self.sample_details

    def get_sample(self, index=None):
        # Get current index if not specified
        if index is None:
            index = self.index
            
        # Get sample details
        sample_details = self.sample_details[index]
        
        # Get mask, image, class id, label
        sample_image = self.load_image_as_tensor(sample_details['image_path'])
        sample_masks = self.load_mask_as_tensor(sample_details['annotation_path'])
        sample_class_ids = self.get_class_ids(sample_masks, ignore=0)
        sample_class_labels = [self.class_index_key_to_object_key(class_id) for class_id in sample_class_ids]
        return sample_image, sample_masks, sample_class_ids, sample_class_labels
        
class ImageNetSDataLoader(DataLoader):
    def __init__(self, data_dir='../data'):
        super().__init__(data_dir)
        # Initialize class label mappings and directories
        self.annotations_dir = f'{data_dir}/ImageNet-S/ImageNetS919/validation-segmentation'
        self.images_dir = f'{data_dir}/ImageNetVal/val'
        self.segment_label_mapping_dir = f'{data_dir}/ImageNetVal/ImageNetS_categories_im919_sort.json'
        self.model_label_mapping_dir = f'{data_dir}/ImageNetVal/map_clsloc_sort.json'
        self.load_class_label_mapping()
        self.load_sample_details()

    def load_class_label_mapping(self):
        # Access metadata from ImageNet1k dataset
        with open(self.model_label_mapping_dir, 'r') as file:
            self.class_object_key = json.load(file)
            
        with open(self.segment_label_mapping_dir, 'r') as file:
            segment_label_mapping_dir = json.load(file)
        
        self.class_index_key = {int(k): str(v) for k, v in segment_label_mapping_dir.items()}
        self.imagenet1k_class_index_key = {v['id']: k for k, v in self.class_object_key.items()}
        
        # Add 1 to account for background
        self.mask_mapping = {k: self.class_object_key[v]['id'] + 1 for k, v in self.class_index_key.items()}
        return None
    
    def imagenet1k_class_index_key_to_class_object(self, imagenet1k_index_key):
        return self.imagenet1k_class_index_key[imagenet1k_index_key]
    
    def imagenet1k_class_index_key_to_class_label(self, imagenet1k_index_key):
        return self.class_object_key[self.imagenet1k_class_index_key[imagenet1k_index_key]]['class_name']
    
    # Overrides to handle ImageNet-S
    def class_object_key_to_class_label(self, object_key):
        return self.class_object_key[object_key]['class_name']
        
    # Overrides to handle ImageNet-S
    def class_index_key_to_class_label(self, index_key):
        return self.class_object_key[self.class_index_key_to_object_key(index_key + 1)]['class_name']
    
    def map_segment_id_to_model_id(self, segment_id):
        class_code = self.class_index_key[segment_id]
        return self.class_object_key[class_code]['id']
            
    def map_segment_id_to_model_class(self, segment_id):
        class_code = self.class_index_key[segment_id]
        return self.class_object_key[class_code]['class_name']
    
    def map_sample_mask_id(self, sample_mask):
        
        # Mask for values in range and in the mapping and use vectorized lookup
        boolean_mask = (sample_mask > 0) & (sample_mask < 920)
        sample_mask[boolean_mask] = np.vectorize(self.mask_mapping.get)(sample_mask[boolean_mask])
        return sample_mask

    def load_sample_details(self):
        
        sample_names = []
        # Get sample names
        for dirpath, _, filenames in os.walk(self.annotations_dir):
            for filename in filenames:
                # Get relative path
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, self.annotations_dir)
                sample_names.append(rel_path.replace('\\', '/'))  # For Windows compatibility
        
        sample_names = [f.replace(".png", "") for f in sample_names if f.endswith('.png')]
        sample_details = {}
        
        for i in range(len(sample_names)):
            
            # Get paths
            sample_name = sample_names[i]
            image_path = os.path.join(self.images_dir, f"{sample_name}.JPEG")
            annotation_path = os.path.join(self.annotations_dir, f"{sample_name}.png")
            
            # Store details
            sample_details[i] = {
                'image_path': image_path,
                'annotation_path': annotation_path,
                'sample_name': sample_name
            }
        
        self.sample_details = sample_details
        return self.sample_details
    
    def get_sample(self, index=None):
        
        # Get current index if not specified
        if index is None:
            index = self.index
            
        # Get sample details
        sample_details = self.sample_details[index]
            
        # Get mask, image, class id, label
        try:
            # ImageNet-S masks need to be loaded differently
            sample_masks = np.array(cv2.cvtColor(cv2.imread(sample_details['annotation_path']), cv2.COLOR_BGR2RGB))
            sample_masks = sample_masks[:, :, 1] * 256 + sample_masks[:, :, 0]
            sample_masks = self.map_sample_mask_id(sample_masks)
            sample_image = self.load_image_as_tensor(sample_details['image_path'])

            sample_model_ids = [sample_model_id - 1 for sample_model_id in np.unique(sample_masks) if sample_model_id != 0]
            sample_model_labels = [IMAGENET_CLASSES[sample_model_id] for sample_model_id in sample_model_ids]
            return sample_image, sample_masks, sample_model_ids, sample_model_labels
        except Exception as e:
            print(f"File loading failed for {sample_details['annotation_path']}: ", e)
            return None, None, None, None
