import torch
import torch.nn as nn
 
from src.models.graph_transformer import NUM_EDGES, D_EDGE
 
   
 
GRU_HIDDEN  = 8 # heavy penalties due to likelihood of overfitting
GRU_DROPOUT = 0.4  
SEQ_LEN     = 20    
 
# the 6 directed edges = 3 unordered weighted edges.
PAIR_EDGES = [(0, 1), (2, 3), (4, 5)]
 
 
class EdgeGRU(nn.Module):
    """
    Temporal layer. Consumes a window of daily edge states from the
    graph transformer and predicts, per currency pair, the forward shift
    in the pair's relationship (change in 20d realized correlation).
 
    Input : edge_seq [B, SEQ_LEN, NUM_EDGES, D_EDGE]
    Output: shift_pred [B, 3]  — one scalar per pair
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
            nn.Linear(2 * GRU_HIDDEN, 1)            # both directions of a pair
            for _ in PAIR_EDGES
        ])
 
    def forward(self, edge_seq):
        B = edge_seq.shape[0]
        finals = []
        for k, gru in enumerate(self.grus):
            seq = self.in_drop(edge_seq[:, :, k, :])     # [B, SEQ_LEN, D_EDGE]
            _, h_n = gru(seq)                            # h_n: [1, B, GRU_HIDDEN]
            finals.append(self.out_drop(h_n.squeeze(0))) # [B, GRU_HIDDEN]
 
        preds = []
        for head, (a, b) in zip(self.heads, PAIR_EDGES):
            pair_state = torch.cat([finals[a], finals[b]], dim=-1)  # [B, 16]
            preds.append(head(pair_state))                          # [B, 1]
        return torch.cat(preds, dim=-1)                             # [B, 3]
 