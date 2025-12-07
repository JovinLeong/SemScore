import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict
from collections import defaultdict

sys.path.append('../pipelines/src')
from data_loaders import ImageNetSDataLoader

class SemScoreClassEvaluator:
    
    def __init__(self, config):
        self.config = config
        self.tasks = config.get("tasks", [])
        self.class_name_from_dataset = {
            'in1k': ImageNetSDataLoader().imagenet1k_class_index_key_to_class_label
        }

    def evaluate_all_tasks(self):
        for task in self.tasks:
            # Get desired task to run
            task_method = getattr(self, f"run_{task}")
            if callable(task_method):
                # Run desired task
                task_method()
                print(f"Completed: {task}!")
            else:
                raise ValueError (f"Unknown action: {task}")
        return None
    
    @staticmethod
    def make_columns(prefix: str, n: int) -> List:
        cols = ["method"]
        for i in range(1, n + 1):
            cols += [f"{prefix}{i}_class", f"{prefix}{i}"]
        return cols

    @staticmethod
    def get_timestamp():
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def get_per_image_scores(methods: List, data: Dict, results: Dict, output_key: str) -> Dict:
        
        results.update({output_key: {}})
        for method_id in range(len(methods)):
            if methods[method_id] == 'attention_rollout':
                raise NotImplementedError(f"Per image score evaluation for {methods[method_id]} is not yet implemented.")
            
            per_image_score = []
            for image_id in data['scores'].keys():
                if isinstance(data['scores'][image_id], str) or not bool(data['scores'][image_id][methods[method_id]]):
                    continue
                score = []
                for k, v in data['scores'][image_id][methods[method_id]].items():
                    score.append(v['semantic_spuriosity_score'])
                if len(score) > 0:
                    per_image_score.append(np.mean(score))
                    
            results[output_key].update({f"{methods[method_id]}_pia": np.mean(per_image_score)})
        return results
    
    @staticmethod
    def get_per_prediction_scores(methods: List, data: Dict, results: Dict, output_key: str) -> Dict:
        per_pred_score = []
        for method_id in range(len(methods)):
            for image_id in data['scores'].keys():
                if isinstance(data['scores'][image_id], str) or not bool(data['scores'][image_id][methods[method_id]]):
                    continue
                for k, v in data['scores'][image_id][methods[method_id]].items():
                    per_pred_score.append(v['semantic_spuriosity_score'])
                    
            results[output_key].update({f"{methods[method_id]}_ppa": np.mean(per_pred_score)})
        return results
    
    def run_vit_layer_analysis(self):
        filepaths = self.config['vit_layer_analysis']['input_jsons']
        methods = self.config['vit_layer_analysis']['methods']
        layers = self.config['vit_layer_analysis']['layers']
        model = filepaths[0].split("/")[-1].split(".")[0]

        results = {}
        for idx, filepath in enumerate(filepaths):
            with open(filepath, 'r') as f:    
                data = json.load(f)

            output_key = layers[idx]
            # Get per image scores and per prediction scores
            results = self.get_per_image_scores(methods, data, results, output_key)
            results = self.get_per_prediction_scores(methods, data, results, output_key)
        self.vit_layer_results = pd.DataFrame(results)
        self.vit_layer_results.index.name = "method"
        self.vit_layer_results.to_csv(f"{self.config['vit_layer_analysis']['output_filename']}_{model}_{self.get_timestamp()}.csv")

        return None        

    def run_threshold_analysis(self, threshold_method: str) -> None:
        if self.config[threshold_method]['task'] == 'object_detection':
            raise NotImplementedError(f"Threshold analysis for {self.config[threshold_method]['task']} is not yet implemented.")

        filepaths = self.config[threshold_method]['input_jsons']
        task = self.config[threshold_method]['task']
        result_key = self.config[threshold_method]['result_key']
        nested_key = self.config[threshold_method]['output_key']
        methods = self.config[task]
        results = {}

        for filepath in filepaths:
            # Load JSON
            with open(filepath, 'r') as f:    
                data = json.load(f)
            
            output_key = data[result_key][nested_key]
            # Get per image SSS with confidence thresholding 
            results = self.get_per_image_scores(methods, data, results, output_key)
            
            # Get per prediction SSS with confidence thresholding 
            results = self.get_per_prediction_scores(methods, data, results, output_key)
        self.threshold_results = pd.DataFrame(results)
        self.threshold_results.index.name = "method"
        self.threshold_results.to_csv(f"{self.config[threshold_method]['output_filename']}_{self.get_timestamp()}.csv")
        
        return None
        
    def run_conf_threshold_analysis(self):
        self.run_threshold_analysis('conf_threshold_analysis')
        return None
    
    def run_sss_threshold_analysis(self):
        self.run_threshold_analysis('sss_threshold_analysis')
        return None

    def get_avg_sss_scores_per_class(self, data: Dict, method: str) -> Dict:
        # Initialize accumulators
        score_totals = defaultdict(float)
        score_counts = defaultdict(int)

        # Iterate across data
        for k, v in data['scores'].items():
            if isinstance(v, str):
                continue
            for key, val in v[method].items():
                score_totals[key] += val['semantic_spuriosity_score']
                score_counts[key] += 1

        averaged_scores = {
            key: {'semantic_spuriosity_score': score_totals[key] / score_counts[key]}
            for key in score_totals
        }
        return averaged_scores

    def get_n_extreme_sss_classes(self, n: int, scores: Dict, method: str, dataset: str, highest: bool = True) -> List:
        # Sort based on semantic spuriosity score
        sorted_items = sorted(
            scores.items(),
            key=lambda x: x[1]['semantic_spuriosity_score'],
            reverse=not highest
        )[:n]

        # Interleave class labels and scores
        row = [method]
        for class_id, score in sorted_items:
            class_label = self.class_name_from_dataset[dataset](int(class_id))
            spuriosity = f"{score['semantic_spuriosity_score']:.4f}"
            row.extend([class_label, spuriosity])

        return row

    def run_top_n_classes(self):
        if self.config['top_n_classes']['task'] == 'object_detection':
            raise NotImplementedError(f"Getting top & bottom n classes for {self.config['top_n_classes']['task']} is not yet implemented.")
        
        # Obtain JSON with SSS containing class-level results and set configurations
        filepath = self.config['top_n_classes']['input_json']
        task = self.config['top_n_classes']['task']
        dataset = self.config['top_n_classes']['dataset']
        n = self.config['top_n_classes']['n_classes']
        
        # Create dataframe to store results
        self.top_n_classes = pd.DataFrame(columns=self.make_columns("top", n))
        self.bottom_n_classes = pd.DataFrame(columns=self.make_columns("bottom", n))
        self.map_class_index_key_to_class_label = self.class_name_from_dataset[dataset]

        os.makedirs(self.config['top_n_classes']['output_dir'], exist_ok=True) 
        filename = os.path.splitext(os.path.basename(filepath))[0]
        output_file = f"{self.config['top_n_classes']['output_dir']}/{filename}_{self.get_timestamp()}"

        with open(filepath, 'r') as f:    
            data = json.load(f)

        # Get top and bottom n number of classes with highest and lowest SSS scores
        methods = self.config[task]
        for method in methods:
            print(f"Getting top and bottom {n} classes for saliency method: {method}")
            if method == 'attention_rollout':
                raise NotImplementedError(f"Getting top & bottom n classes for {method} is not yet implemented.")
            averaged_scores = self.get_avg_sss_scores_per_class(data, method)
            h_row = self.get_n_extreme_sss_classes(n, averaged_scores, method, dataset, highest=True)
            l_row = self.get_n_extreme_sss_classes(n, averaged_scores, method, dataset, highest=False)

            # Appending scores for each method to the dataframe
            self.top_n_classes.loc[len(self.top_n_classes)] = h_row
            self.bottom_n_classes.loc[len(self.bottom_n_classes)] = l_row

        # Save dataframe to csvs
        self.top_n_classes.to_csv(f"{output_file}_top_{self.config['top_n_classes']['n_classes']}.csv", index=False)
        self.bottom_n_classes.to_csv(f"{output_file}_bottom_{self.config['top_n_classes']['n_classes']}.csv", index=False)
        return None

    def run_check_classes(self):
        if self.config['check_classes']['task'] == 'object_detection':
            raise NotImplementedError(f"Checking classes for {self.config['check_classes']['task']} is not yet implemented.")
        
        # Obtain JSON with SSS containing class-level results and set configurations
        filepath = self.config['check_classes']['input_json']
        task = self.config['check_classes']['task']
        os.makedirs(self.config['check_classes']['output_dir'], exist_ok=True) 
        filename = os.path.splitext(os.path.basename(filepath))[0]
        output_file = f"{self.config['check_classes']['output_dir']}/{filename}_{self.get_timestamp()}"
        
        # Create dataframe to store outputs
        columns = ['method'] + [f'class: {c}' for c in self.config['check_classes']['classes']]
        self.classes_to_check_df = pd.DataFrame(columns=columns)
        
        with open(filepath, 'r') as f:    
            data = json.load(f)

        methods = self.config[task]
        for method in methods:
            print(f"Checking classes for the following method: {method}")
            if method == 'attention_rollout':
                raise NotImplementedError(f"Checking classes for {method} is not yet implemented.")
            averaged_scores = self.get_avg_sss_scores_per_class(data, method)
            row = {'method': method}
            for class_id in self.config['check_classes']['classes']:
                row[f"class: {class_id}"] = averaged_scores[str(class_id)]['semantic_spuriosity_score']

            self.classes_to_check_df.loc[len(self.classes_to_check_df)] = row
        self.classes_to_check_df.to_csv(f'{output_file}_classes_to_check.csv', index=False)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/class_analysis.yaml')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    eval = SemScoreClassEvaluator(config)
    eval.evaluate_all_tasks()

