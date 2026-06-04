import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_EDGES   = 6
D_INPUT     = 3   # per-node width: [return, level z-score, vol] — matches X.npy [T, 25, 3]
D_MODEL     = 32
D_EDGE      = 64
NUM_HEADS   = 4
D_HEAD      = D_MODEL // NUM_HEADS  # 8
THRESHOLD   = 0.25
SHARE_FFN   = True   # one FFN for all 6 edges; False = per-edge FFNs (+83k params)
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

        # learned attention prior: per-edge standing preference over the 25
        # nodes (adds node IDENTITY to attention — otherwise nodes are only
        # distinguishable by feature values). 150 params. Data can override
        # it daily; it's a soft prior, not a hard structure.
        self.attn_bias = nn.Parameter(torch.zeros(NUM_EDGES, 25))

        # shared K/V: one common description of the market all edges read
        self.W_K = nn.Linear(D_INPUT, D_MODEL)
        self.W_V = nn.Linear(D_INPUT, D_MODEL)

        self.W_O = nn.ModuleList([
            nn.Linear(D_MODEL, D_EDGE)
            for _ in range(NUM_EDGES)
        ])

        # FFN: with SHARE_FFN one network serves all 6 edges (−83k params on
        # ~5.6k training samples). Per-edge identity is preserved by
        # edge_init / W_Q / W_O / per-edge norms. Flip the flag to revert.
        def _make_ffn():
            return nn.Sequential(
                nn.Linear(D_EDGE, D_EDGE * 2),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(D_EDGE * 2, D_EDGE)
            )
        if SHARE_FFN:
            self.ffn = _make_ffn()
        else:
            self.ffn = nn.ModuleList([_make_ffn() for _ in range(NUM_EDGES)])

        self.norm1 = nn.ModuleList([nn.LayerNorm(D_EDGE) for _ in range(NUM_EDGES)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(D_EDGE) for _ in range(NUM_EDGES)])

    def forward(self, x, nan_mask):
        """
        Forward pass
            x: one day of features [25, 3] — 25 nodes x [pct_change, level z, vol]  
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
            e    = self.edge_init[idx](diff)           # [3] diff -> [64] edge state (expansion)
 
            # independent Q weight per edge 
            q = self.W_Q[idx](e)                       
            q_h = q.view(NUM_HEADS, D_HEAD)            # [4, 8] each 
 
            # 4 heads, 25 x 8 (K) x 8 x 1 (Q)
            scores = torch.bmm(K_h, q_h.unsqueeze(2)).squeeze(2) / (D_HEAD ** 0.5)  # [4, 25]
            scores = scores + self.attn_bias[idx].unsqueeze(0)   # learned node prior
 
            
            if nan_mask is not None:
                edge_mask = nan_mask.clone()
                edge_mask[i] = True # always mask self node
                edge_mask[j] = True
                # FIX: use the dtype's most-negative FINITE value, not -inf.
                # softmax over an all -inf row is 0/0 = NaN, which poisons the
                # backward pass. With a finite fill, an all-masked row softmaxes
                # to a uniform (finite) distribution instead of NaN.
                neg = torch.finfo(scores.dtype).min
                scores = scores.masked_fill(edge_mask.unsqueeze(0), neg)
 
            # softmax over 25 nodes per head
            attn = F.softmax(scores, dim=-1) # [4, 25]
 
            if nan_mask is not None:
                # FIX: force masked positions to exactly zero AFTER softmax.
                # - partially masked rows: cleans up the ~1e-38 residue the
                #   finite fill leaves behind, so masked nodes contribute 0
                # - fully masked rows: zeroes the uniform distribution, so the
                #   downstream renorm (sum + 1e-9) yields an all-zero row and
                #   the edge simply receives no context that day. No NaN.
                attn = attn.masked_fill(edge_mask.unsqueeze(0), 0.0)
 
            
            threshold_vals = attn.max(dim=-1, keepdim=True).values * THRESHOLD
            attn = attn * (attn >= threshold_vals).float()
 
            
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)
 
            # store post-softmax post-threshold weights for orthogonality penalty
            attn_weight_list.append(attn)
 
            context_h = torch.bmm(attn.unsqueeze(1), V_h).squeeze(1)
            context   = context_h.reshape(D_MODEL)
            context = self.W_O[idx](context)           
            e = self.norm1[idx](e + context)
            ffn = self.ffn if SHARE_FFN else self.ffn[idx]
            e = self.norm2[idx](e + ffn(e))
 
            edge_repr_list.append(e)
 
        edge_repr    = torch.stack(edge_repr_list,   dim=0)
        attn_weights = torch.stack(attn_weight_list, dim=0)
 
        return edge_repr, attn_weights

