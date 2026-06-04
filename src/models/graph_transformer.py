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
            scores = torch.bmm(K_h, q_h.unsqueeze(2)).squeeze(2) / (D_HEAD ** 0.5)  # [4, 8] final output, concatted
 
            
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
            e = self.norm2[idx](e + self.ffn[idx](e))
 
            edge_repr_list.append(e)
 
        edge_repr    = torch.stack(edge_repr_list,   dim=0)
        attn_weights = torch.stack(attn_weight_list, dim=0)
 
        return edge_repr, attn_weights


# ---------------------------------------------------------------------------
# guards against NaN explosion which would silently occur. Manual implementation...
if __name__ == '__main__':
    torch.manual_seed(0)

    def _check(label, nan_mask):
        model = FXEdgeTransformer()
        x = torch.randn(25, D_INPUT, requires_grad=True)
        edge_repr, attn = model(x, nan_mask)
        edge_repr.pow(2).mean().backward()      # NaNs usually detonate here
        ok = (torch.isfinite(edge_repr).all()
              and torch.isfinite(attn).all()
              and torch.isfinite(x.grad).all())
        print(f"[{label}] outputs/attn/grads finite: {bool(ok)}")
        return attn, bool(ok)

    attn_w, ok1 = _check("all 25 masked (worst case)", torch.ones(25, dtype=torch.bool))
    ok1 &= bool((attn_w.sum(-1).abs() < 1e-6).all())          # dead rows sum to 0

    partial = torch.zeros(25, dtype=torch.bool)
    partial[[5, 9, 14, 20, 22, 23, 24]] = True
    attn_p, ok2 = _check("7 of 25 masked (realistic)", partial)
    ok2 &= bool((attn_p[:, :, torch.where(partial)[0]] == 0).all())  # masked = exactly 0
    ok2 &= bool(((attn_p.sum(-1) - 1).abs() < 1e-4).all())           # live rows sum to 1

    _, ok3 = _check("no mask (sanity)", None)

    print("ALL CHECKS PASS" if (ok1 and ok2 and ok3) else "FAILED — do not train")
