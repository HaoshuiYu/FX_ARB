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
        Only preserve 6 directed edges between 3 target nodes. Remaining nodes remain for context and aren't related to any except target nodes.
        """
    
        K = self.W_K(x)
        V = self.W_V(x)
 
        # reshape into: [4, 25, 8]
        K_h = K.view(25, NUM_HEADS, D_HEAD).permute(1, 0, 2) 
        V_h = V.view(25, NUM_HEADS, D_HEAD).permute(1, 0, 2) 
 
        edge_repr_list  = []
        attn_weight_list = []
 
        for idx, (i, j) in enumerate(TARGET_EDGES):
 
            # initialize edges: differencing captures directional relationality
            diff = x[i] - x[j]                        
            e    = self.edge_init[idx](diff)           # [75, 64] compression
 
            # independent Q weight per edge 
            q = self.W_Q[idx](e)                       
            q_h = q.view(NUM_HEADS, D_HEAD)            # [4, 8] each 
 
            # 4 heads, 25 x 8 (K) x 8 x 1 (Q)
            scores = torch.bmm(K_h, q_h.unsqueeze(2)).squeeze(2) / (D_HEAD ** 0.5)  # [4, 8] final output, concatted
 
            
            if nan_mask is not None:
                edge_mask = nan_mask.clone()
                edge_mask[i] = True # always mask self node
                edge_mask[j] = True
                scores = scores.masked_fill(edge_mask.unsqueeze(0), float('-inf'))
 
            # softmax over 25 nodes per head
            attn = F.softmax(scores, dim=-1) # [4, 25]
 
            
            threshold_vals = attn.max(dim=-1, keepdim=True).values * THRESHOLD
            attn = attn * (attn >= threshold_vals).float()
 
            
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)
 
            # store post-softmax post-threshold weights for orthogonality penalty
            attn_weight_list.append(attn)
 
            context_h = torch.bmm(attn.unsqueeze(1), V_h).squeeze(1)
            context   = context_h.reshape(D_MODEL)
            context = self.W_O[idx](context)           
            e = self.norm1[idx](e + context)
            e = self.norm2[idx](e + self.ffn[idx](e))
 
            edge_repr_list.append(e)
 
        edge_repr    = torch.stack(edge_repr_list,   dim=0)
        attn_weights = torch.stack(attn_weight_list, dim=0)
 
        return edge_repr, attn_weights
