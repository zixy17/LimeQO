import json
import os
import numpy as np
import time
from strategies.base import BaseStrategy
from models.matrix_factorization import censored_als

class LimeQOStrategy(BaseStrategy):
    """LimeQO strategy using censored matrix factorization"""
    
    def __init__(self, rank=5, lambda_=0.2, alpha=1.0, beta=15.0, new_observe_size=32):
        self.rank = rank
        self.lambda_ = lambda_
        self.alpha = alpha
        self.beta = beta
        self.new_observe_size = new_observe_size
        
    def run(self, dataset, output_path):
        """
        Run LimeQO strategy
        
        Args:
            dataset: Dataset object containing query data
            output_path: Path to save results
        """
        mask = dataset.init_mask.copy()
        exec_time = dataset.get_exec_time(mask)
        timeout_m = np.zeros_like(dataset.matrix)
        explored_m = dataset.init_mask.copy()
        min_observed = dataset.get_min_observed(dataset.matrix, mask)
        timeout = 0
        results = []
        explore_queries = set()
        
        while min_observed.sum() > dataset.opt_time + 20:
            exec_time = dataset.get_exec_time(mask) + timeout
            min_observed = dataset.get_min_observed(dataset.matrix, mask)
            
            masked_m = dataset.matrix * mask
            log_m = np.log1p(masked_m)
            log_timeout_m = np.log1p(timeout_m)
            
            start_time = time.time()
            pred_m = censored_als(log_m, mask, log_timeout_m, self.rank, 50, self.lambda_)
            training_time = time.time() - start_time
            pred_m = np.expm1(pred_m)
            
            pred_m = pred_m * (1-mask)
            pred_m[pred_m == 0] = np.inf
            start_time = time.time()
            mc_select = np.argmin(pred_m, axis=1)
            inference_time = time.time() - start_time
            
            results.append({
                "training_time": training_time,
                "inference_time": inference_time,
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
            
            mc_min = np.min(pred_m, axis=1)
            improve = (min_observed - mc_min) / mc_min
            
            selects = np.argsort(-improve)
            cnt = 0
            for select in selects:
                if cnt >= self.new_observe_size:
                    break
                hint = mc_select[select]
                timeout_tolerance = min(self.alpha * min_observed[select], 
                                     self.beta * pred_m[select, hint])
                
                if (np.isinf(pred_m[select, hint]) or 
                    explored_m[select, hint] != 0 or 
                    pred_m[select, hint] >= timeout_tolerance):
                    continue
                
                same_hints = dataset.get_same_hints(select, hint)
                
                if dataset.matrix[select, hint] >= min_observed[select]:
                    explored_m[select, same_hints] = 1
                
                if dataset.matrix[select, hint] >= timeout_tolerance:
                    timeout_m[select, same_hints] = timeout_tolerance
                    timeout += timeout_tolerance
                    continue
                
                mask[select, same_hints] = 1
                explored_m[select, same_hints] = 1
                cnt += 1
                explore_queries.add(select)
            
            # Random exploration if needed
            while cnt < self.new_observe_size:
                min_observed = dataset.get_min_observed(dataset.matrix, mask)
                if min_observed.sum() <= dataset.opt_time + 50:
                    break
                    
                file_i = np.random.randint(mask.shape[0])
                hint_i = np.random.randint(mask.shape[1])
                
                if mask[file_i, hint_i] == 0 and explored_m[file_i, hint_i] == 0:
                    same_hints = dataset.get_same_hints(file_i, hint_i)
                    
                    if dataset.matrix[file_i, hint_i] >= min_observed[file_i]:
                        timeout += min_observed[file_i]
                        explored_m[file_i, same_hints] = 1
                        timeout_m[file_i, same_hints] = min_observed[file_i]
                        continue
                    
                    explored_m[file_i, same_hints] = 1
                    mask[file_i, same_hints] = 1
                    cnt += 1
                    explore_queries.add(file_i) 