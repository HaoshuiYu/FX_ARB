import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_EDGES   = 6
D_INPUT     = 75
D_MODEL     = 32
D_EDGE      = 64
NUM_HEADS   = 4
D_HEAD      = D_MODEL // NUM_HEADS  # 8
THRESHOLD   = 0.25
DROPOUT     = 0.2

TARGET_EDGES = [
    (0, 1),  # EURUSD -> GBPUSD
    (1, 0),  # GBPUSD -> EURUSD
    (0, 2),  # EURUSD -> USDJPY
    (2, 0),  # USDJPY -> EURUSD
    (1, 2),  # GBPUSD -> USDJPY
    (2, 1),  # USDJPY -> GBPUSD
]

class FXEdgeTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # preserve dimensionality compress down the line 
        self.edge_init = nn.ModuleList([
            nn.Linear(D_INPUT, D_EDGE)
            for _ in range(NUM_EDGES)
        ])

        # per-edge query projection: necessary for different context edges to enrich different target edges
        self.W_Q = nn.ModuleList([
            nn.Linear(D_EDGE, D_MODEL)
            for _ in range(NUM_EDGES)
        ])

        # Compress information post attention
        self.W_K = nn.Linear(D_INPUT, D_MODEL)
        self.W_V = nn.Linear(D_INPUT, D_MODEL)

        self.W_O = nn.ModuleList([
            nn.Linear(D_MODEL, D_EDGE)
            for _ in range(NUM_EDGES)
        ])

        # FFN per edge
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D_EDGE, D_EDGE * 2),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(D_EDGE * 2, D_EDGE)
            )
            for _ in range(NUM_EDGES)
        ])

        self.norm1 = nn.ModuleList([nn.LayerNorm(D_EDGE) for _ in range(NUM_EDGES)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(D_EDGE) for _ in range(NUM_EDGES)])

    def forward(self, x, nan_mask):
        """
        Forward pass
            x: input of [pct_change, lvl, rolling_lvl] of dimensions [25, 75]  
            nan_mask: masking proper, TRUE = masked
 
        Returns:
            edge_repr:   [6, 64]   one representation per directed edge
            attn_weights:[6, 4, 25] post-softmax post-threshold, for orthogonality penalty
        """
        # shared key/value projections — same for all edges
        K = self.W_K(x)
        V = self.W_V(x)
 
        # reshape into: [4, 25, 8]
        K_h = K.view(25, NUM_HEADS, D_HEAD).permute(1, 0, 2) 
        V_h = V.view(25, NUM_HEADS, D_HEAD).permute(1, 0, 2) 
 
        edge_repr_list  = []
        attn_weight_list = []
 
