import json
import os
import numpy as np
from strategies.base import BaseStrategy

class GreedyStrategy(BaseStrategy):
    """Greedy hint selection strategy with timeout"""
    
    def __init__(self, new_observe_size=32):
        self.new_observe_size = new_observe_size
        
    def run(self, dataset, output_path):
        """
        Run greedy strategy with timeout
        
        Args:
            dataset: Dataset object containing query data
            output_path: Path to save results
        """
        mask = np.zeros_like(dataset.init_mask)
        for i in range(dataset.matrix.shape[0]):
            same_hints = dataset.get_same_hints(i, 0)
            mask[i, same_hints] = 1
            
        exec_time = dataset.get_exec_time(mask)
        timeout_m = np.zeros(dataset.matrix.shape)
        min_observed = dataset.get_min_observed(dataset.matrix, mask)
        timeout = 0
        results = []
        explore_queries = set()
        
        while min_observed.sum() > dataset.opt_time + 20:
            exec_time = dataset.get_exec_time(mask) + timeout
            min_observed = dataset.get_min_observed(dataset.matrix, mask)
            
            results.append({
                "training_time": 0,
                "inference_time": 0,
                "exec_time": exec_time,
                "total_latency": np.sum(min_observed),
                "p50": np.median(min_observed),
                "p90": np.percentile(min_observed, 90),
                "p95": np.percentile(min_observed, 95),
                "p99": np.percentile(min_observed, 99),
                "explore_queries_cnt": len(explore_queries)
            })
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)
            
            cnt = 0
            selects = np.argsort(-min_observed)
            
            for i in range(len(selects)):
                if cnt >= self.new_observe_size:
                    break
                    
                file_i = selects[i]
                if mask[file_i].sum() == mask.shape[1]:
                    continue
                    
                while True:
                    hint_i = np.random.randint(dataset.matrix.shape[1])
                    # if all hints have been explored, break
                    if mask[file_i].sum() + timeout_m[file_i].sum() == mask.shape[1]:
                        break
                    if mask[file_i, hint_i] == 0:
                        if timeout_m[file_i, hint_i] == 1:
                            continue
                            
                        same_hints = dataset.get_same_hints(file_i, hint_i)
                        
                        if dataset.matrix[file_i, hint_i] >= min_observed[file_i]:
                            timeout_m[file_i, same_hints] = 1
                            timeout += min_observed[file_i]
                            break
                            
                        mask[file_i, same_hints] = 1
                        cnt += 1
                        explore_queries.add(file_i)
                        break 