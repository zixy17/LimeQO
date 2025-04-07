import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        x = self.tree_pool(x)
        
        user_embedding = self.user_embeddings(user_idx)
        item_embedding = self.item_embeddings(item_idx)
        x = torch.cat((x, user_embedding, item_embedding), dim=1)
        x = self.dropout1(self.relu1(self.ln1(self.linear1(x))))
        x = self.dropout2(self.relu2(self.ln2(self.linear2(x))))
        x = self.output(x)
        return x

def pad_and_stack(items):
    """Utility function to pad and stack batches of tree data"""
    data = [x[0] for x in items]
    idxes = [x[1] for x in items]
    label = [x[2] for x in items]
    matrix_pos = [x[3] for x in items]
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    idxes = torch.nn.utils.rnn.pad_sequence(idxes, batch_first=True)
    return (data, idxes, torch.Tensor(np.array(label)), matrix_pos) 