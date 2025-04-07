import json
import os
import numpy as np
from strategies.base import BaseStrategy

class OracleStrategy(BaseStrategy):
    """Oracle strategy that knows the optimal hints"""
    
    def run(self, dataset, output_path):
        """
        Run oracle strategy
        
        Args:
            dataset: Dataset object containing query data
            output_path: Path to save results
        """
        mask = np.zeros_like(dataset.init_mask)
        for i in range(dataset.matrix.shape[0]):
            same_hints = dataset.get_same_hints(i, 0)
            mask[i, same_hints] = 1
            
        exec_time = dataset.get_exec_time(mask)
        oracle_min_indexes = np.argmin(dataset.matrix, axis=1)
        oracle_min = np.min(dataset.matrix, axis=1)
        min_observed = dataset.get_min_observed(dataset.matrix, mask)
        biggest_gain = min_observed - oracle_min
        biggest_gain_idx = np.argsort(-biggest_gain)
        explore_queries = set()
        
        results = []
        
        for row in biggest_gain_idx:
            if biggest_gain[row] == 0:
                break
                
            min_observed = dataset.get_min_observed(dataset.matrix, mask)
            exec_time = dataset.get_exec_time(mask)
            
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
            
            col = oracle_min_indexes[row]
            if mask[row, col] == 0:
                same_hints = dataset.get_same_hints(row, col)
                mask[row, same_hints] = 1
                explore_queries.add(row) 