import json
import os
import numpy as np
from strategies.base import BaseStrategy

class QOAdvisorStrategy(BaseStrategy):
    """Query Optimizer Advisor strategy that follows default hints"""
    
    def __init__(self, new_observe_size=8):
        self.new_observe_size = new_observe_size
        
    def run(self, dataset, output_path):
        """
        Run QO Advisor strategy
        
        Args:
            dataset: Dataset object containing query data
            output_path: Path to save results
        """
        mask = np.zeros_like(dataset.matrix)
        timeout_m = np.zeros_like(dataset.matrix)
        explored_m = np.zeros_like(dataset.matrix)
        
        # Initialize with first hint for each query
        for i in range(dataset.matrix.shape[0]):
            same_hints = dataset.get_same_hints(i, 0)
            mask[i, same_hints] = 1
            
        exec_time = dataset.get_exec_time(mask)
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
            
            # Try next hint for worst performing queries
            selects = np.argsort(-min_observed)
            cnt = 0
            
            for select in selects:
                if cnt >= self.new_observe_size:  # Fixed batch size of 8
                    break
                    
                # Find next unexplored hint
                for hint in range(dataset.matrix.shape[1]):
                    if mask[select, hint] == 0 and explored_m[select, hint] == 0:
                        same_hints = dataset.get_same_hints(select, hint)
                        
                        if dataset.matrix[select, hint] >= min_observed[select]:
                            timeout += min_observed[select]
                            explored_m[select, same_hints] = 1
                            timeout_m[select, same_hints] = min_observed[select]
                            continue
                        
                        mask[select, same_hints] = 1
                        explored_m[select, same_hints] = 1
                        cnt += 1
                        explore_queries.add(select)
                        break 