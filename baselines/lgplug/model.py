"""Full randomly initialized graph-text alignment for the agreed LG-Plug port."""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import CLIPTextConfig, CLIPTextModel


class GraphText(torch.nn.Module):
    def __init__(self, input_dim, cfg, token_ids):
        super().__init__()
        width = cfg["graph_width"]
        self.conv1 = GCNConv(input_dim, width)
        self.conv2 = GCNConv(width, width)
        self.slope = cfg["leaky_slope"]
        text_cfg = CLIPTextConfig(vocab_size=cfg["vocab_size"], hidden_size=cfg["text_width"],
                                  intermediate_size=cfg["text_ffn"], num_hidden_layers=cfg["text_layers"],
                                  num_attention_heads=cfg["text_heads"], max_position_embeddings=cfg["max_length"],
                                  hidden_act="quick_gelu", attention_dropout=0., **token_ids)
        self.text = CLIPTextModel(text_cfg)
        if cfg["gradient_checkpointing"]:
            self.text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.projection = torch.nn.Linear(cfg["text_width"], width, bias=False)
        self.log_scale = torch.nn.Parameter(torch.tensor(math.log(cfg["logit_scale_initial"])))
        self.max_scale = math.log(cfg["logit_scale_max"])

    def graph(self, x, edges):
        h = F.leaky_relu(self.conv1(x, edges), negative_slope=self.slope)
        return F.leaky_relu(self.conv2(h, edges), negative_slope=self.slope)

    def texts(self, tokens, mask):
        return self.projection(self.text(input_ids=tokens, attention_mask=mask).pooler_output)

    def scale(self):
        return self.log_scale.clamp(max=self.max_scale)


def alignment_split(n, edges, seed, validation_nodes, validation_edges):
    """Hold out text pairs and internal validation edges without class labels."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    val = order[:validation_nodes]
    train = order[validation_nodes:]
    if not len(train) or not len(val):
        raise ValueError("Alignment split must have training and validation nodes")
    pairs = np.asarray(edges).T
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]  # Input graph is symmetric.
    held = np.zeros(n, dtype=bool)
    held[val] = True
    train_edges = pairs[~held[pairs].any(1)]
    val_edges = pairs[held[pairs].all(1)]
    if not len(val_edges) or not len(train_edges):
        raise ValueError("Alignment split has no internal edges")
    val_edges = val_edges[rng.choice(len(val_edges), min(validation_edges, len(val_edges)), replace=False)]
    return {"train_nodes": train, "validation_nodes": val, "train_edges": train_edges, "validation_edges": val_edges}


def pack_rows(nodes, pairs):
    rows = np.unique(np.concatenate([nodes, pairs.ravel()]))
    return rows, np.searchsorted(rows, nodes), np.searchsorted(rows, pairs)
