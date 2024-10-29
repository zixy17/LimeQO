import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import argparse
from zipfile import ZipFile
import json
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import time

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='ceb', required=True)
argparser.add_argument('--device', type=str, default='cpu')
argparser.add_argument('--rank', type=int, default=5)

args = argparser.parse_args()

dataset = args.dataset
device = args.device
rank = args.rank

if torch.cuda.is_available() and device == "cuda":
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Device:", device)

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]
    
    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)

        if xroot == yroot:
            return
        
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        elif self.rank[xroot] > self.rank[yroot]:
            self.parent[yroot] = xroot
        else:
            self.parent[yroot] = xroot
            self.rank[xroot] += 1
    
    def get_elements_in_set(self, x):
        xroot = self.find(x)
        return [i for i in range(len(self.parent)) if self.find(i) == xroot]

def als(X, mask, rank, niters, lambda_):
    """
    Alternating Least Squares algorithm for matrix factorization
    X: matrix to factorize
    mask: binary mask of observed entries
    rank: rank of the factorization
    niters: number of iterations
    lambda_: regularization parameter
    """
    n, m = X.shape
    A = np.random.rand(n, rank)
    B = np.random.rand(m, rank)
    for _ in range(niters):
        target = X + (1 - mask) * (np.dot(A, B.T))
        A = np.linalg.solve(np.dot(B.T, B) + lambda_ * np.eye(rank), np.dot(target, B).T).T
        A[A < 0] = 0
        target = X + (1 - mask) * (np.dot(A, B.T))
        B = np.linalg.solve(np.dot(A.T, A) + lambda_ * np.eye(rank), np.dot(target.T, A).T).T
        B[B < 0] = 0
    
    return X + (1 - mask) * (np.dot(A, B.T))

def censored_als(X, mask, cutoffs, rank, niters, lambda_):
    """
    Alternating Least Squares algorithm for matrix factorization
    X: matrix to factorize
    mask: binary mask of observed entries
    rank: rank of the factorization
    niters: number of iterations
    lambda_: regularization parameter
    """
    n, m = X.shape
    A = np.random.rand(n, rank)
    B = np.random.rand(m, rank)
    for _ in range(niters):
        target = X + (1 - mask) * (np.dot(A, B.T))
        violations = (target < cutoffs) & (cutoffs > 0)
        target[violations] = cutoffs[violations]
        A = np.linalg.solve(np.dot(B.T, B) + lambda_ * np.eye(rank), np.dot(target, B).T).T
        A[A < 0] = 0
        target = X + (1 - mask) * (np.dot(A, B.T))
        violations = (target < cutoffs) & (cutoffs > 0)
        target[violations] = cutoffs[violations]
        B = np.linalg.solve(np.dot(A.T, A) + lambda_ * np.eye(rank), np.dot(target.T, A).T).T
        B[B < 0] = 0
    
    return X + (1 - mask) * (np.dot(A, B.T))

# define TCNN components
class TreeConvolution(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.__conv = nn.Conv1d(in_f, out_f, 3, stride=3)

    def forward(self, x):
        data, idxes = x
        first_dim = torch.arange(len(data))[:, None]
        selected_data = data[first_dim, idxes]
        result = self.__conv(selected_data.transpose(1, 2)).transpose(1, 2)

        return (result, idxes)

class TreeReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.__f = nn.ReLU()

    def forward(self, x):
        data, idxes = x
        return (self.__f(data), idxes)

class TreeLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.__f = nn.LayerNorm(dim)

    def forward(self, x):
        data, idxes = x
        return (self.__f(data), idxes)

class TreeDynamicPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        data, idxes = x
        return torch.max(data, dim=1).values

class TreeDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.__dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        data, idxes = x
        return (self.__dropout(data), idxes)



class TCNN(nn.Module):
    def __init__(self, num_features, rank, num_matrix_rows, num_matrix_cols):
        super(TCNN, self).__init__()
        self.tree_conv1 = TreeConvolution(num_features, 256)
        self.tree_ln1 = TreeLayerNorm(256)
        self.tree_relu1 = TreeReLU()
        self.tree_dropout1 = TreeDropout()
        
        self.tree_conv2 = TreeConvolution(256, 128)
        self.tree_ln2 = TreeLayerNorm(128)
        self.tree_relu2 = TreeReLU()
        self.tree_dropout2 = TreeDropout()
        
        # self.tree_conv3 = TreeConvolution(128, 64)
        # self.tree_ln3 = TreeLayerNorm(64)
        # self.tree_relu3 = TreeReLU()
        # self.tree_dropout3 = TreeDropout()
        
        self.tree_pool = TreeDynamicPooling()
        
        self.user_embeddings = nn.Embedding(num_matrix_rows, rank)
        self.item_embeddings = nn.Embedding(num_matrix_cols, rank)
        
        self.linear1 = nn.Linear(128 + 2*rank, 32)
        self.ln1 = nn.LayerNorm(32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
        
        self.linear2 = nn.Linear(32, 16)
        self.ln2 = nn.LayerNorm(16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)
        
        self.output = nn.Linear(16, 1)

    def forward(self, x, user_idx, item_idx):
        x = self.tree_dropout1(self.tree_relu1(self.tree_ln1(self.tree_conv1(x))))
        x = self.tree_dropout2(self.tree_relu2(self.tree_ln2(self.tree_conv2(x))))
        # x = self.tree_dropout3(self.tree_relu3(self.tree_ln3(self.tree_conv3(x))))
        
        x = self.tree_pool(x)
        
        user_embedding = self.user_embeddings(user_idx)
        item_embedding = self.item_embeddings(item_idx)
        x = torch.cat((x, user_embedding, item_embedding), dim=1)
        x = self.dropout1(self.relu1(self.ln1(self.linear1(x))))
        x = self.dropout2(self.relu2(self.ln2(self.linear2(x))))
        x = self.output(x)
        return x

def pad_and_stack(items):
    data = [x[0] for x in items]
    idxes = [x[1] for x in items]
    label = [x[2] for x in items]
    matrix_pos = [x[3] for x in items]
    data = pad_sequence(data, batch_first=True)
    idxes = pad_sequence(idxes, batch_first=True)
    return (data, idxes, torch.Tensor(np.array(label)), matrix_pos)

def train_tcnn(tcnn, ds, device, censored, epochs=100):
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
            
            y_censored = torch.Tensor(len(label))
            for i, mpos in enumerate(_mposes):
                y_censored[i] = censored[mpos["row"], mpos["cols"][0]]
            
            # loss = censored_loss(pred, label, y_censored)
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

def censored_loss(y_pred, y, censored, threshold=0):
    mse_loss = F.mse_loss(y_pred, y, reduction='none') + threshold
    loss = torch.where(torch.logical_or(censored == 0,y_pred < censored), mse_loss, threshold)
    return loss.mean()

def train_tcnn_censored(tcnn, ds, device, censored, epochs=100):
    begin_train_time = time.time()
    # loss_fn = torch.nn.MSELoss()
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
            
            y_censored = torch.Tensor(len(label))
            for i, mpos in enumerate(_mposes):
                y_censored[i] = censored[mpos["row"], mpos["cols"][0]]
            
            loss = censored_loss(pred, label, y_censored)
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

def evaluate_tcnn(tcnn, test_ds, device, ds, matrix, latency_xform):
    eval_begin_time = time.time()
    tcnn.eval()
    abs_errs = []
    num_matrix_rows, num_matrix_cols = matrix.shape
    pred_m = np.zeros((num_matrix_rows, num_matrix_cols))
    observed_m = np.zeros((num_matrix_rows, num_matrix_cols))

    for (data, idx, label, mposes) in test_ds:
        data = data.to(device)
        idx = idx.to(device)
        user_idxs = [pos['row'] for pos in mposes]
        item_idxs = [pos['cols'][0] for pos in mposes] 
        user_idxs = torch.tensor(user_idxs, device=device)
        item_idxs = torch.tensor(item_idxs, device=device)
            
        pred = tcnn((data, idx), user_idxs, item_idxs).detach().cpu().numpy()
        real_latency = latency_xform.inverse_transform(label).flatten()
        pred_latency = latency_xform.inverse_transform(pred).flatten()
        abs_errs.extend(np.abs(real_latency - pred_latency))
        
        # fill unobserved into matrix
        for pred_l, mpos in zip(pred_latency, mposes):
            for col in mpos["cols"]:
                pred_m[mpos["row"], col] = pred_l
    
    # fill observed training data into matrix as well
    for (_data, _idx, label, mposes) in ds:
        real_latency = latency_xform.inverse_transform(label).flatten()
        for real_l, mpos in zip(real_latency, mposes):
            for col in mpos["cols"]:
                pred_m[mpos["row"], col] = real_l
                # matrix[mpos["row"], col] = np.inf
                observed_m[mpos["row"], col] = real_l
            # print("row:", mpos["row"], "cols:", mpos["cols"])
            # print("real latency:", real_latency)

    eval_end_time = time.time()
    # find minimums of each row
    tcnn_selected  = np.argmin(pred_m, axis=1)[:, None]
    tcnn_actual_perf = np.take_along_axis(matrix, tcnn_selected, axis=1).flatten()
    
    # minimum observed latency
    observed_m[observed_m == 0] = np.inf
    min_observed = np.min(observed_m, axis=1)
    tcnn_total_latency = np.sum(min_observed)
    tcnn_p50 = np.percentile(min_observed, 50)
    tcnn_p90 = np.percentile(min_observed, 90)
    tcnn_p95 = np.percentile(min_observed, 95)
    tcnn_p99 = np.percentile(min_observed, 99)
    
    print("TCNN total latency:", tcnn_total_latency)  
    print("TCNN P90 latency:", tcnn_p90)
    
    return pred_m, eval_end_time - eval_begin_time


# Load the dataset
class Dataset():
    def __init__(self, dataset):
        matrix_path = f'dataset/{dataset}-matrix.csv'
        init_mask_path = f'dataset/init_{dataset}_mask.npy'
        self.zip_path = f'dataset/{dataset}.zip'
        
        self.matrix_df = pd.read_csv(matrix_path, index_col='filename')
        self.matrix = self.matrix_df.to_numpy()
        
        self.default_time = np.sum(self.matrix[:,0])
        print("default hint time:", self.default_time)
        self.opt_time = np.sum(np.min(self.matrix, axis=1))
        print("optimal hint time:", self.opt_time)
        
        self.init_mask = np.load(init_mask_path)
        self.ufs = {}
        self.load_plans()
        self.init_union_find()


    def load_plans(self):
        i = 0
        all_plans = []
        with ZipFile(self.zip_path, "r") as f:
            plans = [x for x in f.namelist() if x.endswith(".json") and "MACOSX" not in x]
            for fp in tqdm(plans):
                with f.open(fp) as pfd:
                    data = json.load(pfd)
                    data["plan_tree"] = data["plan"][0][0][0]["Plan"]
                    all_plans.append(data)
        self.all_plans = all_plans
        print("Loaded", len(all_plans), "query plans")

        all_filenames = set(x["filename"] for x in all_plans)
        filenames_to_idx = {fn: idx for idx, fn in enumerate(sorted(all_filenames))}
        self.all_filenames = all_filenames
        self.filenames_to_idx = filenames_to_idx
        for plan in all_plans:
            plan["matrix pos"] = {"row": filenames_to_idx[plan["filename"]],
                                "cols": plan["hint_list"]}


        all_matrix_pos = [x["matrix pos"] for x in all_plans]
        self.all_matrix_pos = all_matrix_pos

        num_matrix_rows = len(all_filenames)
        num_matrix_cols = max(max(plan["hint_list"]) for plan in all_plans) + 1
        assert min(min(plan["hint_list"]) for plan in all_plans) == 0
        matrix = np.zeros((num_matrix_rows, num_matrix_cols))
        for plan in all_plans:
            latency = np.median(plan["runtime_list"])
            row_idx = filenames_to_idx[plan["filename"]]
            for col_idx in plan["hint_list"]:
                matrix[row_idx, col_idx] = latency

        print("Rows:", num_matrix_rows, "Cols:", num_matrix_cols)

        # collect all operators for one-hot encoding
        def collect_node_types(tree):
            s = set()
            s.add(tree["Node Type"])
            if "Plans" in tree:
                for child in tree["Plans"]:
                    s |= collect_node_types(child)
            return s

        all_operators = set()
        for plan in tqdm(all_plans):
            all_operators |= collect_node_types(plan["plan_tree"])

        all_operators.add("Dummy")

        all_operators = sorted(all_operators)
        op_name_to_idx = {op: idx for idx, op in enumerate(all_operators)}
        print("Found", len(all_operators), "distinct operators:", all_operators)

        # Take the log of query latency
        log_xform = FunctionTransformer(np.log1p, inverse_func=np.expm1)

        y = np.array([np.median(x["runtime_list"]) for x in all_plans]).reshape(-1, 1)
        latency_xform = Pipeline([("log", log_xform), ("scale", MinMaxScaler())])
        self.latency_xform = latency_xform
        y = latency_xform.fit_transform(y)
        self.y = y

        # Log scale and min/max scale all features
        def root_to_features(tree):
            features = np.zeros(len(all_operators) + 3)
            features[op_name_to_idx[tree["Node Type"]]] = 1
            features[-3] = tree["Total Cost"]
            features[-2] = tree["Plan Rows"]
            features[-1] = tree["Plan Width"]
            return features


        def collect_features(tree):
            results = [root_to_features(tree)]
            if "Plans" in tree:
                for child in tree["Plans"]:
                    results.extend(collect_features(child))
            return results

        all_feature_values = []
        for plan in tqdm(all_plans):
            all_feature_values.extend(collect_features(plan["plan_tree"]))
            
        num_features = len(all_feature_values[0])
        print("Features per node:", num_features)
        self.num_features = num_features
        dummy_vec = np.zeros(num_features)
        dummy_vec[op_name_to_idx["Dummy"]] = 1.0
        all_feature_values.append(dummy_vec)


        all_feature_values = np.array(all_feature_values)
        feature_xform = Pipeline([("log", log_xform), ("scale", MinMaxScaler())])
        feature_xform.fit(all_feature_values)
        print("Feature scaler is ready")

        # Binarize for TCNN
        dummy = {"Node Type": "Dummy", "Total Cost": 0, "Plan Rows": 0, "Plan Width": 0}
        def binarize(tree):
            tree = tree.copy()
            if "Plans" not in tree:
                if tree["Node Type"] == "Dummy":
                    return tree

                tree["Plans"] = [
                    dummy, dummy
                ]
                return tree


            children = tree["Plans"]
            if len(children) == 2:
                tree["Plans"] = [
                    binarize(children[0]),
                    binarize(children[1])
                ]
                return tree

            if len(children) == 1:
                tree["Plans"] = [
                    binarize(children[0]),
                    dummy
                ]
                return tree

            raise "Tree with " + str(len(children)) + " children"

        # Give each node a pre-order index
        def add_indexes(tree, start_at=1):
            if tree["Node Type"] == "Dummy":
                tree["tcnn index"] = 0
                return start_at

            assert "tcnn index" not in tree
            tree["tcnn index"] = start_at

            start_at += 1
            if "Plans" not in tree:
                return start_at

            start_at = add_indexes(tree["Plans"][0], start_at = start_at)
            return add_indexes(tree["Plans"][1], start_at = start_at)


        trees = [x["plan_tree"] for x in all_plans]

        x = []
        for plan in tqdm(trees):
            plan = binarize(plan)
            add_indexes(plan)
            x.append(plan)
            
        # Sanity check: make sure the trees are binary and indexes are valid
        def verify_tree(tree, total_nodes):
            assert "tcnn index" in tree, f"node with type {tree['Node Type']} had no index"
            assert tree["tcnn index"] <= total_nodes, f"node with type {tree['Node Type']} has index {tree['tcnn index']} > {total_nodes}"
            if "Plans" in tree:
                assert len(tree["Plans"]) == 2
                verify_tree(tree["Plans"][0], total_nodes)
                verify_tree(tree["Plans"][1], total_nodes)

        def count_non_dummy_nodes(tree):
            my_count = 0 if tree["Node Type"] == "Dummy" else 1
            if "Plans" in tree:
                for plan in tree["Plans"]:
                    my_count += count_non_dummy_nodes(plan)
            return my_count

        for tree in tqdm(x):
            total_nodes = count_non_dummy_nodes(tree)
            verify_tree(tree, total_nodes)
            
        # Prepre data for 1D conv
        def flatten_tree(tree):
            if tree["Node Type"] == "Dummy":
                return []

            nodes = []
            nodes.append(root_to_features(tree))
            nodes.extend(flatten_tree(tree["Plans"][0]))
            nodes.extend(flatten_tree(tree["Plans"][1]))
            return nodes

        def tcnn_indexes(tree):
            if tree["Node Type"] == "Dummy":
                return []

            indexes = []
            indexes.append(tree["tcnn index"])
            indexes.append(tree["Plans"][0]["tcnn index"])
            indexes.append(tree["Plans"][1]["tcnn index"])

            indexes.extend(tcnn_indexes(tree["Plans"][0]))
            indexes.extend(tcnn_indexes(tree["Plans"][1]))
            return indexes

        x_data = []
        x_idxes = []

        for tree in tqdm(x):
            x_data.append(np.array([dummy_vec] + flatten_tree(tree)))
            x_idxes.append(np.array([0, 0, 0] + tcnn_indexes(tree)))

        print("Smallest plan:", min(x.shape[0] for x in x_idxes))
        print("Median plan:", np.median([x.shape[0] for x in x_idxes]))
        print("Largest plan:", max(x.shape[0] for x in x_idxes))

        for data, idxes in zip(x_data, x_idxes):
            assert np.max(idxes) < len(data), f"max index was {np.max(idxes)}, but data length was {len(data)}"

        self.x_data = [torch.Tensor(feature_xform.transform(x)) for x in x_data]
        self.x_idxes = [torch.IntTensor(x) for x in x_idxes]
    
    def init_union_find(self):
        for plan in self.all_plans:
            plan["matrix pos"] = {"row": self.filenames_to_idx[plan["filename"]],
                                "cols": plan["hint_list"]}
            i = plan["matrix pos"]["row"]
            hint_list = plan['hint_list']
            for j in range(len(hint_list)):
                col = hint_list[j]
                if i not in self.ufs:
                    self.ufs[i] = UnionFind(49)
                for k in range(j+1, len(hint_list)):
                    col2 = hint_list[k]
                    self.ufs[i].union(col, col2)
    
    def get_same_hints(self, q, hintset):
        return self.ufs[q].get_elements_in_set(hintset)

    def get_exec_time(self, mask):
        observed = self.matrix * mask
        exec_time = 0
        groups = []
        n,d = self.matrix.shape
        for i in range(n):
            groups.append(set())
        for i in range(n):
            for j in range(d):
                if mask[i, j] == 1:
                    group = self.ufs[i].find(j)
                    if group not in groups[i]:
                        groups[i].add(group)
                        exec_time += observed[i, j]
        
        return exec_time

    def get_p_49(mask):
        return mask.sum() / mask.size

    def get_min_observed(self, m, mask):
        R = m * mask
        R[R == 0] = np.inf
        min_observed_latency = np.min(R, axis=1)
        return min_observed_latency

    def oracle(self, output_path):
        mask = np.zeros_like(self.init_mask)
        for i in range(self.matrix.shape[0]):
            same_hints = self.get_same_hints(i, 0)
            mask[i, same_hints] = 1
        exec_time = self.get_exec_time(mask)
        oracle_min_indexes = np.argmin(self.matrix, axis=1)
        oracle_min = np.min(self.matrix, axis=1)
        min_observed = self.get_min_observed(self.matrix, mask)
        biggest_gain = min_observed - oracle_min
        biggest_gain_idx = np.argsort(-biggest_gain)
        
        results = []
        
        for row in biggest_gain_idx:
            if biggest_gain[row] == 0:
                break
            
            min_observed = self.get_min_observed(self.matrix, mask)
            exec_time = self.get_exec_time(mask)
            
            results.append({"training_time": 0,
                            "inference_time": 0,
                            "exec_time": exec_time, 
                            "total_latency": np.sum(min_observed), 
                            "p50": np.median(min_observed), 
                            "p90": np.percentile(min_observed, 90), 
                            "p95": np.percentile(min_observed, 95), 
                            "p99": np.percentile(min_observed, 99)})
            
            with open(output_path, "w") as file:
                json.dump(results, file, indent=4)
            
            # print("Total latency: ", np.sum(min_observed))
            # print("Execution time: ", exec_time)
            
            col = oracle_min_indexes[row]
            if mask[row, col] == 0:
                same_hints = self.get_same_hints(row, col)
                mask[row, same_hints] = 1
        
    def random_timeout(self, output_path, new_observe_size = 32):
        mask = np.zeros_like(self.init_mask)
        for i in range(self.matrix.shape[0]):
            same_hints = self.get_same_hints(i, 0)
            mask[i, same_hints] = 1
        exec_time = self.get_exec_time(mask)
        timeout_m = np.zeros(self.matrix.shape)
        min_observed = self.get_min_observed(self.matrix, mask)
        timeout = 0
        results = []
        
        while min_observed.sum() > self.opt_time + 20:
            exec_time = self.get_exec_time(mask) + timeout
            min_observed = self.get_min_observed(self.matrix, mask)
            
            results.append({"training_time": 0,
                            "inference_time": 0,
                            "exec_time": exec_time, 
                            "total_latency": np.sum(min_observed), 
                            "p50": np.median(min_observed), 
                            "p90": np.percentile(min_observed, 90), 
                            "p95": np.percentile(min_observed, 95), 
                            "p99": np.percentile(min_observed, 99)})
            
            with open(output_path, "w") as file:
                json.dump(results, file, indent=4)
            
            cnt = 0
            while cnt <= new_observe_size:
                file_i = np.random.randint(mask.shape[0])
                hint_i = np.random.randint(mask.shape[1])
                if mask[file_i, hint_i] == 0 \
                    and timeout_m[file_i, hint_i] == 0:
                        
                    same_hints = self.get_same_hints(file_i, hint_i)
                    
                    if self.matrix[file_i, hint_i] >= min_observed[file_i]:
                        timeout += min_observed[file_i]
                        timeout_m[file_i, same_hints] = 1
                        continue
                    
                    mask[file_i, same_hints] = 1
                    cnt += 1
    
    def greedy_timeout(self, output_path, new_observe_size = 32):
        mask = np.zeros_like(self.init_mask)
        for i in range(self.matrix.shape[0]):
            same_hints = self.get_same_hints(i, 0)
            mask[i, same_hints] = 1
        exec_time = self.get_exec_time(mask)
        timeout_m = np.zeros(self.matrix.shape)
        min_observed = self.get_min_observed(self.matrix, mask)
        timeout = 0
        results = []
        
        while min_observed.sum() > self.opt_time + 20:
            exec_time = self.get_exec_time(mask) + timeout
            min_observed = self.get_min_observed(self.matrix, mask)
            
            results.append({"training_time": 0,
                            "inference_time": 0,
                            "exec_time": exec_time, 
                            "total_latency": np.sum(min_observed), 
                            "p50": np.median(min_observed), 
                            "p90": np.percentile(min_observed, 90), 
                            "p95": np.percentile(min_observed, 95), 
                            "p99": np.percentile(min_observed, 99)})
            
            with open(output_path, "w") as file:
                json.dump(results, file, indent=4)
            
            cnt = 0
            selects = np.argsort(-min_observed)
            
            for i in range(len(selects)):
                if cnt >= new_observe_size:
                    break
                file_i = selects[i]
                
                if mask[file_i].sum() == mask.shape[1]:
                    continue
                
                while 1:
                    hint_i = np.random.randint(self.matrix.shape[1])
                    
                    if mask[file_i, hint_i] == 0:
                        if timeout_m[file_i, hint_i] == 1:
                            continue
                        
                        same_hints = self.get_same_hints(file_i, hint_i)
                        
                        if self.matrix[file_i, hint_i] >= min_observed[file_i]:
                            timeout_m[file_i, same_hints] = 1
                            # print("Timeout File {}, Hint {}, Real {}, Min observed {}".format(file_i, hint_i, matrix[file_i, hint_i], min_observed[file_i]))
                            timeout += min_observed[file_i]
                            break
                        
                        mask[file_i, same_hints] = 1
                        # print("File {}, Hint {}, Real {}, Min observed {}".format(file_i, hint_i, matrix[file_i, hint_i], min_observed[file_i]))
                        cnt += 1
                    break
    
    def censored_als_timeout(self, rank, lambda_, alpha, beta, output_path, new_observe_size=128):

        mask = self.init_mask.copy()
        exec_time = self.get_exec_time(mask)
        timeout_m = np.zeros_like(self.matrix)
        explored_m = self.init_mask.copy()
        min_observed = self.get_min_observed(self.matrix, mask)
        timeout = 0
        results = []
        
        while min_observed.sum() > self.opt_time + 50:
            exec_time = self.get_exec_time(mask) + timeout
            min_observed = self.get_min_observed(self.matrix, mask)
            
            masked_m = self.matrix * mask
            log_m = np.log1p(masked_m)
            log_timeout_m = np.log1p(timeout_m)
            
            start_time = time.time()
            pred_m = censored_als(log_m, mask, log_timeout_m, rank, 50, lambda_)
            training_time = time.time() - start_time
            pred_m = np.expm1(pred_m)
            mse = np.mean((pred_m - self.matrix) ** 2)
            pred_m = pred_m * (1-mask)
            pred_m[pred_m == 0] = np.inf
            start_time = time.time()
            mc_select = np.argmin(pred_m, axis=1)
            inference_time = time.time() - start_time
            
            results.append({"training_time": training_time,
                            "inference_time": inference_time,
                            "exec_time": exec_time, 
                            "total_latency": np.sum(min_observed), 
                            "p50": np.median(min_observed), 
                            "p90": np.percentile(min_observed, 90), 
                            "p95": np.percentile(min_observed, 95), 
                            "p99": np.percentile(min_observed, 99)})
            
            with open(output_path, "w") as file:
                json.dump(results, file, indent=4)
            
            # print("Total latency: ", np.sum(min_observed))
            # print("Execution time: ", exec_time)
            # print("MSE", mse)
            
            mc_min = np.min(pred_m, axis=1)
            improve = (min_observed - mc_min) / mc_min
            
            selects = np.argsort(-improve)
            cnt = 0
            for select in selects:
                if cnt >= new_observe_size:
                    break
                hint = mc_select[select]
                timeout_tolerance = min(alpha * min_observed[select], beta * pred_m[select, hint])
                # assert improve[select] == min_observed[select] - pred_m[select, hint]
                if np.isinf(pred_m[select, hint]) \
                    or explored_m[select, hint] != 0 \
                    or pred_m[select, hint] >= timeout_tolerance:
                    continue
                
                same_hints = self.get_same_hints(select, hint)
                
                if self.matrix[select, hint] >= min_observed[select]:
                    explored_m[select, same_hints] = 1
                
                if self.matrix[select, hint] >= timeout_tolerance:
                    timeout_m[select, same_hints] = timeout_tolerance
                    timeout += timeout_tolerance
                    # print("Timeout File {}, Hint {}, Real {}, Min observed {}".format(select, hint, matrix[select, hint], timeout_tolerance))
                    continue
                
                mask[select,same_hints] = 1
                explored_m[select, same_hints] = 1
                cnt += 1
                # print("File {}, Hint {}, Prediction {}, Min Observed {}, Actual {}".format(select, hint, pred_m[select, hint], min_observed[select], matrix[select, hint]))
            
            while cnt < new_observe_size:
                min_observed = self.get_min_observed(self.matrix, mask)
                if min_observed.sum() <= self.opt_time + 50:
                    break
                file_i = np.random.randint(mask.shape[0])
                hint_i = np.random.randint(mask.shape[1])
                if mask[file_i, hint_i] == 0 \
                    and explored_m[file_i, hint_i] == 0:
                        
                    same_hints = self.get_same_hints(file_i, hint_i)
                    
                    if self.matrix[file_i, hint_i] >= min_observed[file_i]:
                        timeout += min_observed[file_i]
                        explored_m[file_i, same_hints] = 1
                        timeout_m[file_i, same_hints] =  min_observed[file_i]
                        continue
                    
                    explored_m[file_i, same_hints] = 1
                    mask[file_i, same_hints] = 1
                    cnt += 1

    def tcnn_embedding_timeout_censored(self, rank, alpha, beta, output_path, new_observe_size = 128):
        num_features = self.num_features
        x_data = self.x_data
        x_idxes = self.x_idxes
        y = self.y
        all_matrix_pos = self.all_matrix_pos
        matrix = self.matrix
        
        tcnn = TCNN(num_features, rank = rank, num_matrix_rows = matrix.shape[0], num_matrix_cols = matrix.shape[1])
        tcnn = tcnn.to(device)

        all_data = list(zip(x_data, x_idxes, y, all_matrix_pos))
        # train, test = train_test_split(all_data, train_size=0.1)
        
        init_mask = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            init_mask[i, self.get_same_hints(i, 0)] = 1
        
        mask = init_mask.copy()
        exec_time = self.get_exec_time(mask)
        min_observed = self.get_min_observed(matrix, mask)
        timeout_m = np.zeros_like(matrix)
        explored_m = np.zeros_like(matrix)
        timeout = 0
        
        results = []
        
        while min_observed.sum() > self.opt_time + 50:
            exec_time = self.get_exec_time(mask) + timeout
            min_observed = self.get_min_observed(matrix, mask)
            
            train_indices = []
            for i, pos in enumerate(all_matrix_pos):
                if mask[pos['row'], pos['cols'][0]] == 1:
                    train_indices.append(i)
            train_data = [all_data[i] for i in train_indices]
            test_indices = [i for i in range(len(all_data)) if i not in train_indices]
            test_data = [all_data[i] for i in test_indices]
            
            print("Train size:", len(train_data), "Test size:", len(test_data))
            
            ds = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=pad_and_stack)
            test_ds = DataLoader(test_data, batch_size=128, shuffle=True, collate_fn=pad_and_stack)
            
            tcnn, train_time = train_tcnn_censored(tcnn, ds, device, timeout_m, epochs=200)
            tcnn_m, eval_time = evaluate_tcnn(tcnn, test_ds, device, ds, matrix, self.latency_xform)
            
            mse = np.mean((tcnn_m - matrix) ** 2)
            
            print("Execution Time {}, Timeout {}".format(exec_time, timeout))
            
            results.append({"training_time": train_time, 
                            "inference_time": eval_time, 
                            "exec_time": exec_time, 
                            "total_latency": np.sum(min_observed), 
                            "p50": np.median(min_observed), 
                            "p90": np.percentile(min_observed, 90), 
                            "p95": np.percentile(min_observed, 95), 
                            "p99": np.percentile(min_observed, 99)})
            
            with open(output_path, "w") as file:
                json.dump(results, file, indent=4)
            
            print("Total latency: ", np.sum(min_observed))
            print("Execution time: ", exec_time)
            
            tcnn_m = tcnn_m * (1-mask)
            tcnn_m[tcnn_m == 0] = np.inf
            tcnn_select = np.argmin(tcnn_m, axis=1)
            tcnn_min = np.min(tcnn_m, axis=1)
            improve = (min_observed - tcnn_min) / tcnn_min

            selects = np.argsort(-improve)
            cnt = 0
            for select in selects:
                if cnt >= new_observe_size:
                    break
                hint = tcnn_select[select]
                # assert improve[select] == min_observed[select] - pred_m[select, hint]
                
                timeout_tolerance = min(alpha * min_observed[select], beta * tcnn_m[select, hint])
                
                if np.isinf(tcnn_m[select, hint]) \
                    or explored_m[select, hint] != 0 \
                    or tcnn_m[select, hint] >= timeout_tolerance:
                    continue
                
                same_hints = self.get_same_hints(select, hint)
                
                if matrix[select, hint] >= min_observed[select]:
                    explored_m[select, same_hints] = 1
                
                
                if matrix[select, hint] >= timeout_tolerance:
                    timeout_m[select, same_hints] = timeout_tolerance
                    timeout += timeout_tolerance
                    print("Timeout File {}, Hint {}, Predicted {}, Real {}, Min observed {}".format(select, hint, tcnn_m[select, hint], matrix[select, hint], min_observed[select]))
                    continue
                
                mask[select,same_hints] = 1
                cnt += 1
                print("File {}, Hint {}, Prediction {}, Min Observed {}, Actual {}".format(select, hint, tcnn_m[select, hint], min_observed[select], matrix[select, hint]))
            
            while cnt < new_observe_size:
                min_observed = self.get_min_observed(matrix, mask)
                if min_observed.sum() <= self.opt_time + 50:
                    break
                file_i = np.random.randint(mask.shape[0])
                hint_i = np.random.randint(mask.shape[1])
                if mask[file_i, hint_i] == 0 \
                    and explored_m[file_i, hint_i] == 0:
                        
                    same_hints = self.get_same_hints(file_i, hint_i)
                    if matrix[file_i, hint_i] >= min_observed[file_i]:
                        timeout += min_observed[file_i]
                        explored_m[file_i, same_hints] = 1
                        timeout_m[file_i, same_hints] =  min_observed[file_i]
                        continue
                    
                    mask[file_i, same_hints] = 1
                    cnt += 1

def __main__():
    ds = Dataset(dataset)
    # # oracle
    print("Running oracle")
    output_path = f"experiment/{dataset}/oracle.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    open(output_path, 'w').close()
    ds.oracle(output_path)
    
    # # random
    print("Running random")
    for i in tqdm(range(1,21)):
        output_path = f"experiment/{dataset}/random/{i}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        open(output_path, 'w').close()
        ds.random_timeout(output_path)
        
    # # greedy
    print("Running greedy")
    for i in tqdm(range(1,21)):
        output_path = f"experiment/{dataset}/greedy/{i}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        open(output_path, 'w').close()
        ds.greedy_timeout(output_path)
        
    # limeqo
    print("Running limeqo")
    lambda_ = 0.2
    alpha = 1
    beta = 15
    new_observe_size = 16
    for run in tqdm(range(1,21)):
        output_path = f"experiment/{dataset}/limeqo/{run}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        open(output_path, 'w').close()
        ds.censored_als_timeout(rank, lambda_, alpha, beta, output_path, new_observe_size)

    # limeqo+
    print("Running limeqo+")
    for run in tqdm(range(1,6)):
        output_path = f"experiment/{dataset}/limeqo+/{run}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        open(output_path, 'w').close()
        ds.tcnn_embedding_timeout_censored(rank, alpha, beta, output_path, new_observe_size=32)

if __name__ == '__main__':
    __main__()