import os
import torch
import torch.nn as nn

from src.models.graph_transformer import NUM_EDGES, D_EDGE

GRU_HIDDEN  = int(os.environ.get('FX_GRU_HIDDEN', 24))
GRU_DROPOUT = 0.4
SEQ_LEN     = int(os.environ.get('FX_SEQ_LEN', 20))

# 6 directed edges = 3 unordered pairs
PAIR_EDGES = [(0, 1), (2, 3), (4, 5)]


class EdgeGRU(nn.Module):
    """Temporal layer:
            inputs: edge_seq [batch, seq length rolling, edge count, size of edge vector]
            outputs: shift_pred [batch, 3 currency pairs]
    """
    def __init__(self):
        super().__init__()
        self.grus = nn.ModuleList([
            nn.GRU(input_size=D_EDGE, hidden_size=GRU_HIDDEN, batch_first=True)
            for _ in range(NUM_EDGES)
        ])
        self.in_drop  = nn.Dropout(GRU_DROPOUT)
        self.out_drop = nn.Dropout(GRU_DROPOUT)
        self.heads = nn.ModuleList([
            nn.Linear(2 * GRU_HIDDEN, 1)             # separates directions, directional weighted graph
            for _ in PAIR_EDGES
        ])

    def forward(self, edge_seq):
        final = [] # isolates each of 6 edges
        for k, gru in enumerate(self.grus):
            seq = self.in_drop(edge_seq[:, :, k, :])
            _, h_n = gru(seq)
            final.append(self.out_drop(h_n.squeeze(0)))

        preds = [] # 6 directed ->3 undirected edges, correlations are symmetric
        for head, (a, b) in zip(self.heads, PAIR_EDGES):
            pair_state = torch.cat([final[a], final[b]], dim=-1)
            preds.append(head(pair_state))
        return torch.cat(preds, dim=-1)
