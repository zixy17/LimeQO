import json
import os
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from strategies.base import BaseStrategy
from models.tcnn import TCNN, pad_and_stack

class LimeQOPlusStrategy(BaseStrategy):
    """LimeQO+ strategy using Tree-CNN and timeout mechanism"""
    
    def __init__(self, rank=5, alpha=1.0, beta=15.0, new_observe_size=32, device="cpu"):
        self.rank = rank
        self.alpha = alpha
        self.beta = beta
        self.new_observe_size = new_observe_size
        self.device = device
        
    def run(self, dataset, output_path):
        """
        Run LimeQO+ strategy
        
        Args:
            dataset: Dataset object containing query data
            output_path: Path to save results
        """
        mask = dataset.init_mask.copy()
        exec_time = dataset.get_exec_time(mask)
        timeout_m = np.zeros_like(dataset.matrix)
        explored_m = np.zeros_like(dataset.matrix)
        min_observed = dataset.get_min_observed(dataset.matrix, mask)
        timeout = 0
        results = []
        explore_queries = set()
        
        # Initialize TCNN model
        tcnn = TCNN(dataset.num_features, self.rank, 
                    dataset.matrix.shape[0], dataset.matrix.shape[1])
        tcnn = tcnn.to(self.device)
        
        while min_observed.sum() > dataset.opt_time + 10:
            exec_time = dataset.get_exec_time(mask) + timeout
            min_observed = dataset.get_min_observed(dataset.matrix, mask)
            
            # Prepare training data
            train_indices = []
            for i, pos in enumerate(dataset.all_matrix_pos):
                if mask[pos['row'], pos['cols'][0]] == 1:
                    train_indices.append(i)
            train_data = [dataset.get_plan_data(i) for i in train_indices]
            test_indices = [i for i in range(len(dataset.all_matrix_pos)) 
                          if i not in train_indices]
            test_data = [dataset.get_plan_data(i) for i in test_indices]
            
            ds = DataLoader(train_data, batch_size=32, shuffle=True, 
                          collate_fn=pad_and_stack)
            test_ds = DataLoader(test_data, batch_size=128, shuffle=True,
                               collate_fn=pad_and_stack)
            
            # Train model
            start_time = time.time()
            tcnn = self.train_model_censored(tcnn, ds, timeout_m)
            train_time = time.time() - start_time
            print("Training time:", train_time)
            
            # Make predictions
            start_time = time.time()
            pred_m = self.evaluate_model(tcnn, test_ds, dataset)
            inference_time = time.time() - start_time
            print("Inference time:", inference_time)
            
            results.append({
                "training_time": train_time,
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
            
            # Select queries to explore
            pred_m = pred_m * (1-mask)
            pred_m[pred_m == 0] = np.inf
            tcnn_select = np.argmin(pred_m, axis=1)
            tcnn_min = np.min(pred_m, axis=1)
            improve = (min_observed - tcnn_min) / tcnn_min
            
            selects = np.argsort(-improve)
            cnt = 0
            for select in selects:
                if cnt >= self.new_observe_size:
                    break
                hint = tcnn_select[select]
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
     
    def train_tcnn(self, tcnn, ds, device, epochs=100):
        """Train TCNN model"""
        begin_train_time = time.time()
        loss_fn = torch.nn.MSELoss()
        opt = torch.optim.Adam(tcnn.parameters())
        losses = []
        
        for epoch in range(epochs):
            tcnn.train()
            tloss = 0.0
            for (data, idx, label, _mposes) in ds:
                data = data.to(device)
                idx = idx.to(device)
                user_idxs = [pos['row'] for pos in _mposes]
                item_idxs = [pos['cols'][0] for pos in _mposes] 
                user_idxs = torch.tensor(user_idxs, device=device)
                item_idxs = torch.tensor(item_idxs, device=device)
                
                pred = tcnn((data, idx), user_idxs, item_idxs)
                label = label.to(device)
                
                loss = loss_fn(pred, label)
                tloss += loss.item()
                loss.backward()
                opt.step()
                opt.zero_grad()
            tloss /= len(ds)
            losses.append(tloss)
            
            # Stop condition
            if len(losses) > 10 and losses[-1] < 0.1:
                last_two = np.min(losses[-2:])
                if last_two > losses[-10] or (losses[-10] - last_two < 0.0001):
                    print("Stopped training from convergence condition at epoch", epoch)
                    break
            
            if epoch % 10 == 0:
                print("Epoch", epoch, "training loss:", tloss)
        end_train_time = time.time()
        print("Training time:", end_train_time - begin_train_time)
        # save_model(tcnn, model_path)
        return tcnn, end_train_time - begin_train_time
                    
    def train_model_censored(self, tcnn, ds, censored, epochs=200):
        """Train TCNN model with censoring"""
        opt = torch.optim.Adam(tcnn.parameters())
        losses = []
        
        for epoch in range(epochs):
            tcnn.train()
            tloss = 0.0
            for (data, idx, label, _mposes) in ds:
                data = data.to(self.device)
                idx = idx.to(self.device)
                user_idxs = [pos['row'] for pos in _mposes]
                item_idxs = [pos['cols'][0] for pos in _mposes] 
                user_idxs = torch.tensor(user_idxs, device=self.device)
                item_idxs = torch.tensor(item_idxs, device=self.device)
            
                pred = tcnn((data, idx), user_idxs, item_idxs)
                label = label.to(self.device)
            
                y_censored = torch.Tensor(len(label))
                for i, mpos in enumerate(_mposes):
                    y_censored[i] = censored[mpos["row"], mpos["cols"][0]]
                
                loss = self.censored_loss(pred, label, y_censored)
                tloss += loss.item()
                loss.backward()
                opt.step()
                opt.zero_grad()
            tloss /= len(ds)
            losses.append(tloss)
            
            # Stop condition
            if len(losses) > 10 and losses[-1] < 0.1:
                last_two = np.min(losses[-2:])
                if last_two > losses[-10] or (losses[-10] - last_two < 0.0001):
                    print("Stopped training from convergence condition at epoch", epoch)
                    break
            
            if epoch % 10 == 0:
                print("Epoch", epoch, "training loss:", tloss)
        # save_model(tcnn, model_path)
        return tcnn
    
    def evaluate_model(self, tcnn, test_ds, dataset):
        """Evaluate TCNN model and return predictions"""
        tcnn.eval()
        num_matrix_rows, num_matrix_cols = dataset.matrix.shape
        pred_m = np.zeros((num_matrix_rows, num_matrix_cols))
        
        with torch.no_grad():
            for batch in test_ds:
                data, idx, _, mposes = batch
                data = data.to(self.device)
                idx = idx.to(self.device)
                
                user_idxs = torch.tensor([pos['row'] for pos in mposes],
                                       device=self.device)
                item_idxs = torch.tensor([pos['cols'][0] for pos in mposes],
                                       device=self.device)
                
                pred = tcnn((data, idx), user_idxs, item_idxs)
                pred = dataset.latency_xform.inverse_transform(
                    pred.cpu().numpy()).flatten()
                
                for p, mpos in zip(pred, mposes):
                    for col in mpos["cols"]:
                        pred_m[mpos["row"], col] = p
                        
        return pred_m
    
    def censored_loss(self, y_pred, y, censored, threshold=0):
        """Compute loss with censoring"""
        mse_loss = torch.nn.functional.mse_loss(y_pred, y, reduction='none') + threshold
        loss = torch.where(torch.logical_or(censored == 0, y_pred < censored),
                         mse_loss, threshold)
        return loss.mean() 