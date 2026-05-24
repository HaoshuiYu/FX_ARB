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

        # compress raw edges into 
        self.input_proj = nn.Linear(D_INPUT, D_MODEL)

        # edge initialization — one per target edge
        self.edge_init = nn.ModuleList([
            nn.Linear(D_MODEL, D_EDGE)
            for _ in range(NUM_EDGES)
        ])

        # per-edge query projection — each edge asks its own question
        self.W_Q = nn.ModuleList([
            nn.Linear(D_EDGE, D_MODEL)
            for _ in range(NUM_EDGES)
        ])

        # shared key and value projections across all edges
        self.W_K = nn.Linear(D_MODEL, D_MODEL)
        self.W_V = nn.Linear(D_MODEL, D_MODEL)

        # output projection after head concatenation
        self.W_O = nn.ModuleList([
            nn.Linear(D_MODEL, D_EDGE)
            for _ in range(NUM_EDGES)
        ])

        # attention bias prior — 25 scalars per edge
        self.attn_bias = nn.Parameter(torch.zeros(NUM_EDGES, 25))

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