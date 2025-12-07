import os
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

def load_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

class SemScoreAggregator:
    def __init__(self, config):
        with open(config, "r") as f:
            self.config = yaml.safe_load(f)

    @staticmethod
    def get_timestamp():
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_scores_to_csv(self, df: pd.DataFrame, task_name: str, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{task_name}_aggregated_scores_{self.get_timestamp()}.csv")
        df.to_csv(output_path)
        print(f"Saved to: {output_path}")
        return None

    @staticmethod
    def get_aggregated_scores(results_paths: List[str], methods: List[str]) -> Dict[str, Any]:
        aggregated_scores_dict = {}
        for i in range(len(results_paths)):
            # Load JSON
            data = load_json(results_paths[i])
            model_name = data['metadata']['model_settings']['model']
            aggregated_scores_dict.update({model_name: {}})
            
            # Iterate across methods; rows
            for method_id in range(len(methods)):
                per_pred_score = []
                per_image_score = []
                method = methods[method_id]

                # Iterate across samples and collate scores
                for image_id in data['scores'].keys():
                    # Skip if no scores
                    if isinstance(data['scores'][image_id], str) or not bool(data['scores'][image_id][method]):
                        continue
                    image_scores = []
                    if method == 'attention_rollout':
                        image_scores.append(data['scores'][image_id][method]['semantic_spuriosity_score'])
                        per_pred_score.append(data['scores'][image_id][method]['semantic_spuriosity_score'])
                    else:
                        for k, v in data['scores'][image_id][method].items():
                            image_scores.append(v['semantic_spuriosity_score'])
                            per_pred_score.append(v['semantic_spuriosity_score'])
                    if len(image_scores) > 0:
                        per_image_score.append(np.mean(image_scores))
                
                aggregated_scores_dict[model_name].update({
                    f"{method}_pia": np.mean(per_image_score),
                    f"{method}_ppa": np.mean(per_pred_score)
                })
        return aggregated_scores_dict

    @staticmethod
    def get_object_detection_aggregated_scores(results_paths: List[str], methods: List[str]) -> Dict[str, Any]:
        aggregated_scores_dict = {}
        for i in range(len(results_paths)):
            data = load_json(results_paths[i])
            model_name = data['metadata']['model_settings']['model']
            aggregated_scores_dict.update({model_name: {}})
            
            # Iterate across methods; rows
            for method_id in range(len(methods)):
                method = methods[method_id]
                per_image_score = []
                # Iterate across samples and collate scores
                for image_id in data['scores'].keys():
                    # Skip if no scores
                    if isinstance(data['scores'][image_id], str) or not bool(data['scores'][image_id][method]):
                        continue
                    per_image_score.append(data['scores'][image_id][method]['semantic_spuriosity_score'])
                    
                aggregated_scores_dict[model_name].update({
                    f"{method}_pia": np.mean(per_image_score),
                })
        return aggregated_scores_dict

    def aggregate_sss(self) -> None:
        selected = self.config.get("selected_tasks", [])
        # Access methods and results_dir for each selected task
        for task in selected:
            task_info = self.config["tasks"].get(task, {})
            methods = task_info.get("methods", [])
            results_dir = task_info.get("results_dir", "")
            
            print(f"Task: {task}")
            print(f"Results dir: {results_dir}")
            print(f"Methods: {methods}")

            results = os.listdir(results_dir)
            results_paths = [f"{results_dir}/{result}" for result in results if result.endswith('.json')]
            
            if task == "object_detection":
                aggregated_scores_dict = self.get_object_detection_aggregated_scores(results_paths, methods)
            else:
                aggregated_scores_dict = self.get_aggregated_scores(results_paths, methods)
                
            df = pd.DataFrame(aggregated_scores_dict).T
            df.index.name = "model"
            self.save_scores_to_csv(df, task, self.config.get("output_dir", "."))
            
        return None
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/aggregated_scores_config.yaml')
    args = parser.parse_args()
    sss_aggregator = SemScoreAggregator(args.config)
    sss_aggregator.aggregate_sss()