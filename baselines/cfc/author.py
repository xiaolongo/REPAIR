import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops
from typing import List, Union, Tuple


class GCN_model2(nn.Module):
    def __init__(self, in_feat, h_feat, out_feat, num_layers):
        super().__init__()
        self.Conv1 = GCNConv(in_feat, h_feat)
        self.Conv2 = GCNConv(h_feat, out_feat)
        self.clf1 = nn.Linear(out_feat, num_layers)
    
    def forward(self, seq, edge_index):
        res1 = self.Conv1(seq, edge_index)
        res2 = self.Conv2(res1, edge_index)
        res3 = self.clf1(res2)

        return res3

class labelConv(MessagePassing):
    # def __init__(self, in_channels, out_channels):
    #     super(labelConv, self).__init__(aggr='add')
    def label_propagate(self, feature, edge_index, train_indices, selected_unseen_indices, order=2, drop_rate=0.1):
        #feature = F.dropout(feature, args.dropout, training=training)
        x = feature
        edge_index, _ = add_self_loops(edge_index)
        train_mask = torch.zeros(x.shape[0], dtype=torch.bool).cuda()
        train_mask[train_indices] = True

        n = x.shape[0]
        drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)
        
        train_label = feature[train_mask,:]
        #y = torch.zeros([x.shape[0],x.shape[1]]).cuda()
        for i in range(order):
            masks = torch.bernoulli(1. - drop_rates).unsqueeze(1).cuda()
            x = masks * x
            # x = torch.spmm(edge_index, x)
            x = self.propagate(edge_index, x=x, edge_weight=None)

            x[train_mask, :] = train_label

        return x.detach()
