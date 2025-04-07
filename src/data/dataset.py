import numpy as np
import pandas as pd
import os
import json
from zipfile import ZipFile
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from utils.union_find import UnionFind
import torch

class Dataset:
    """
    Dataset class for loading and preprocessing query execution data
    """
    def __init__(self, dataset_name):
        """
        Initialize dataset
        
        Args:
            dataset_name: Name of the dataset to load
        """
        self.dataset_name = dataset_name
        self.matrix_path = f'dataset/{dataset_name}-matrix.csv'
        self.init_mask_path = f'dataset/init_{dataset_name}_mask.npy'
        self.zip_path = f'dataset/{dataset_name}.zip'
        
        # Load matrix and initial mask
        self.matrix_df = pd.read_csv(self.matrix_path, index_col='filename')
        self.matrix = self.matrix_df.to_numpy()
        self.init_mask = np.load(self.init_mask_path)
        
        # Calculate default and optimal times
        self.default_time = np.sum(self.matrix[:,0])
        self.opt_time = np.sum(np.min(self.matrix, axis=1))
        print(f"Default hint time: {self.default_time}")
        print(f"Optimal hint time: {self.opt_time}")
        
        # Initialize data structures
        self.ufs = {}  # Union-find sets for equivalent hints
        self.total_cost = np.zeros((self.matrix.shape[0], self.matrix.shape[1]))
        
        # Load and preprocess query plans
        self.load_plans()
        self.init_union_find()

    def load_plans(self):
        """Load and preprocess query execution plans"""
        all_plans = []
        with ZipFile(self.zip_path, "r") as f:
            plans = [x for x in f.namelist() if x.endswith(".json") and "MACOSX" not in x]
            for fp in tqdm(plans, desc="Loading plans"):
                with f.open(fp) as pfd:
                    data = json.load(pfd)
                    data["plan_tree"] = data["plan"][0][0][0]["Plan"]
                    all_plans.append(data)
        
        self.all_plans = all_plans
        print(f"Loaded {len(all_plans)} query plans")

        # Map filenames to indices
        all_filenames = set(x["filename"] for x in all_plans)
        self.filenames_to_idx = {fn: idx for idx, fn in enumerate(sorted(all_filenames))}
        self.all_filenames = all_filenames
        
        # Add matrix positions to plans
        for plan in all_plans:
            plan["matrix pos"] = {
                "row": self.filenames_to_idx[plan["filename"]],
                "cols": plan["hint_list"]
            }
        self.all_matrix_pos = [x["matrix pos"] for x in all_plans]

        # Collect all operators for feature encoding
        def collect_node_types(tree):
            s = {tree["Node Type"]}
            if "Plans" in tree:
                for child in tree["Plans"]:
                    s |= collect_node_types(child)
            return s

        all_operators = set()
        for plan in tqdm(all_plans, desc="Collecting operators"):
            all_operators |= collect_node_types(plan["plan_tree"])
        all_operators.add("Dummy")
        self.all_operators = sorted(all_operators)
        self.op_name_to_idx = {op: idx for idx, op in enumerate(self.all_operators)}
        print(f"Found {len(all_operators)} distinct operators")

        # Setup feature transformation pipeline
        log_xform = FunctionTransformer(np.log1p, inverse_func=np.expm1)
        self.latency_xform = Pipeline([
            ("log", log_xform),
            ("scale", MinMaxScaler())
        ])
        
        # Transform latencies
        y = np.array([np.median(x["runtime_list"]) for x in all_plans]).reshape(-1, 1)
        self.y = self.latency_xform.fit_transform(y)

        # Process plan features
        self.num_features = len(self.all_operators) + 3  # operators + cost/rows/width
        self.process_plan_features()

    def root_to_features(self, tree):
            """Convert a tree node to feature vector"""
            features = np.zeros(self.num_features)
            features[self.op_name_to_idx[tree["Node Type"]]] = 1
            features[-3] = tree["Total Cost"]
            features[-2] = tree["Plan Rows"]
            features[-1] = tree["Plan Width"]
            return features
    
    def process_plan_features(self):
        """Process and transform plan features"""
        def collect_features(tree):
            results = [self.root_to_features(tree)]
            if "Plans" in tree:
                for child in tree["Plans"]:
                    results.extend(collect_features(child))
            return results

        # Collect and transform features
        all_feature_values = []
        for plan in tqdm(self.all_plans, desc="Processing features"):
            all_feature_values.extend(collect_features(plan["plan_tree"]))
            
        # Add dummy feature vector
        dummy_vec = np.zeros(self.num_features)
        dummy_vec[self.op_name_to_idx["Dummy"]] = 1.0
        all_feature_values.append(dummy_vec)
        self.dummy_vec = dummy_vec

        # Setup feature transformation
        self.feature_xform = Pipeline([
            ("log", FunctionTransformer(np.log1p, inverse_func=np.expm1)),
            ("scale", MinMaxScaler())
        ])
        self.feature_xform.fit(np.array(all_feature_values))

        # Binarize trees for TCNN
        self.binarize_trees()

    def binarize_trees(self):
        """Convert trees to binary form for TCNN"""
        dummy = {"Node Type": "Dummy", "Total Cost": 0, "Plan Rows": 0, "Plan Width": 0}
        
        def binarize(tree):
            tree = tree.copy()
            if "Plans" not in tree:
                if tree["Node Type"] == "Dummy":
                    return tree
                tree["Plans"] = [dummy, dummy]
                return tree

            children = tree["Plans"]
            if len(children) == 2:
                tree["Plans"] = [binarize(children[0]), binarize(children[1])]
                return tree

            if len(children) == 1:
                tree["Plans"] = [binarize(children[0]), dummy]
                return tree

            raise ValueError(f"Tree with {len(children)} children")

        def add_indexes(tree, start_at=1):
            if tree["Node Type"] == "Dummy":
                tree["tcnn index"] = 0
                return start_at

            tree["tcnn index"] = start_at
            start_at += 1
            
            if "Plans" not in tree:
                return start_at

            start_at = add_indexes(tree["Plans"][0], start_at=start_at)
            return add_indexes(tree["Plans"][1], start_at=start_at)

        # Process all trees
        self.binary_trees = []
        for plan in tqdm(self.all_plans, desc="Binarizing trees"):
            tree = binarize(plan["plan_tree"])
            add_indexes(tree)
            self.binary_trees.append(tree)

    def init_union_find(self):
        """Initialize union-find data structures for equivalent hints"""
        for plan in self.all_plans:
            i = plan["matrix pos"]["row"]
            hint_list = plan['hint_list']
            for j in range(len(hint_list)):
                col = hint_list[j]
                if i not in self.ufs:
                    self.ufs[i] = UnionFind(49)  # Assuming max 49 hints
                for k in range(j+1, len(hint_list)):
                    col2 = hint_list[k]
                    self.ufs[i].union(col, col2)

    def get_same_hints(self, q, hintset):
        """Get all hints equivalent to the given hint for query q"""
        return self.ufs[q].get_elements_in_set(hintset)

    def get_exec_time(self, mask):
        """Calculate total execution time given observation mask"""
        observed = self.matrix * mask
        exec_time = 0
        groups = [set() for _ in range(self.matrix.shape[0])]
        
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                if mask[i, j] == 1:
                    group = self.ufs[i].find(j)
                    if group not in groups[i]:
                        groups[i].add(group)
                        exec_time += observed[i, j]
        
        return exec_time

    def get_min_observed(self, m, mask):
        """Get minimum observed latency for each query"""
        R = m * mask
        R[R == 0] = np.inf
        return np.min(R, axis=1)

    def get_plan_data(self, idx):
        """Get preprocessed plan data for TCNN"""
        plan = self.all_plans[idx]
        tree = self.binary_trees[idx]
        
        def flatten_tree(t):
            if t["Node Type"] == "Dummy":
                return []
            nodes = [self.root_to_features(t)]
            nodes.extend(flatten_tree(t["Plans"][0]))
            nodes.extend(flatten_tree(t["Plans"][1]))
            return nodes

        def tcnn_indexes(t):
            if t["Node Type"] == "Dummy":
                return []
            indexes = [t["tcnn index"], t["Plans"][0]["tcnn index"], t["Plans"][1]["tcnn index"]]
            indexes.extend(tcnn_indexes(t["Plans"][0]))
            indexes.extend(tcnn_indexes(t["Plans"][1]))
            return indexes

        # Get features and indexes
        features = np.array([self.dummy_vec] + flatten_tree(tree))
        indexes = np.array([0, 0, 0] + tcnn_indexes(tree))
        
        return (
            torch.Tensor(self.feature_xform.transform(features)),
            torch.IntTensor(indexes),
            self.y[idx],
            plan["matrix pos"]
        )
