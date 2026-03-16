import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Transformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, context_len):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # Positional embeddings
        self.pos_embedding = nn.Embedding(context_len, hidden_dim)

        # TODO: Add projections for the queries, keys, values, and matrices.
        # These should map from a matrix of size `hidden_dim` to a matrix
        # of size `hidden_dim`.
        # STUDENT START ---------------------------------
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # STUDENT END -----------------------------------

        # TODO: Add two projects for the MLP: W_up and W_down.
        # Make sure that the output dimension of W_up is 4 times the
        # size of the original hidden_dim, and that W_down puts it back
        # into the original dimensionality.
        # STUDENT START ---------------------------------
        self.W_up = nn.Linear(hidden_dim, hidden_dim * 4) 
        self.W_down = nn.Linear(hidden_dim * 4, hidden_dim)
        # STUDENT END -----------------------------------
        
        # Gamma and Beta for Attention output and MLP output
        self.gamma_attn = nn.Parameter(torch.ones(hidden_dim))
        self.beta_attn = nn.Parameter(torch.zeros(hidden_dim))
        self.gamma_mlp = nn.Parameter(torch.ones(hidden_dim))
        self.beta_mlp = nn.Parameter(torch.zeros(hidden_dim))

    def layer_norm(self, x, gamma, beta):
        # TODO: implement layer norm.
        # For now, this just returns x. Delete this line,
        # and replace it with this functionality:
        # x_hat = gamma * (x - mu) / sigma + beta
        # STUDENT START ------------------------------
        # return x
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-5)
        return gamma * (x - mean) / std + beta
        # STUDENT END --------------------------------

    def forward(self, x):
        B, T = x.size()     # (Batch size, sequence length)
        
        # 1. token embeddings + positional encodings
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)

        # TODO: 2. Implement self-attention. Q, K, and V should
        # be of dimension (B, T).
        # STUDENT START --------------------------
        residual_1 = h # Store for residual

        Q = self.W_q(h)
        K = self.W_k(h)
        V = self.W_v(h)

        # Attention Scores: Q @ K.T / sqrt(d_k)
        # (B, T, H) @ (B, H, T) -> (B, T, T)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.hidden_dim)

        # Causal Masking (Decoder only attends to past)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, float('-inf'))

        # Softmax and multiply by values
        attn_weights = F.softmax(attn_scores, dim=-1)
        # (B, T, T) @ (B, T, H) -> (B, T, H)
        a = torch.bmm(attn_weights, V)
        
        # Output projection
        a = self.W_o(a)

        # Add residual and layer norm
        a = self.layer_norm(residual_1 + a, self.gamma_attn, self.beta_attn)
        # STUDENT END ------------------------------

        # TODO: 3. Implement the MLP.
        # STUDENT START -------------------------
        residual_2 = a # Store for residual
        
        # MLP: h = ReLU(aW_up + b1)W_down + b2
        # Note: Linear layers in PyTorch handle the bias terms b1 and b2 internally,
        # so you don't need to manually add them.
        mlp_out = self.W_down(F.relu(self.W_up(a)))
        # STUDENT END ----------------------------

        # TODO: 4. Add residual & layer norm. 
        # The current `self.layer_norm` method simply returns its input without
        # actually normalizing them; you will need to implement `self.layer_norm`
        # for this to work properly.
        # STUDENT START ----------------------------
        output = self.layer_norm(residual_2 + mlp_out, self.gamma_mlp, self.beta_mlp)
        # STUDENT END ------------------------------

        return output


class MultiHeadTransformer(Transformer):
    def __init__(self, vocab_size, hidden_dim, context_len, num_heads=4):
        super().__init__(vocab_size, hidden_dim, context_len)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "Hidden dim must be divisible by num_heads"

        # In MHA, we usually implement Q, K, V as one large layer and split them, 
        # or separate layers per head. Here we keep the single layers but shape them differently in forward.
        # Efficient implementation: Use the existing linear layers but view as (B, T, Heads, Head_Dim)
        
    def forward(self, x):
        B, T = x.size()
        
        # Embeddings
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)

        # TODO: Implement multi-head attention. This should be nearly identical
        # to your implementation in `Transformer`, except Q, K, and V should
        # be of size (B, T, num_heads, head_dim); your previous implementation
        # had them at size (B, T).
        # STUDENT START ------------------------------------
        residual_1 = h

        # Calculate Q, K, V for all heads at once
        # Reshape to (B, T, Num_Heads, Head_Dim) then Transpose to (B, Num_Heads, T, Head_Dim)
        Q = self.W_q(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scores: (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
        a_heads = attn_weights @ V

        # Concatenate heads
        # Transpose back: (B, T, H, D) -> reshape to (B, T, hidden_dim)
        a_concat = a_heads.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)

        # Output projection
        a = self.W_o(a_concat)

        # Add residual and layer norm
        a = self.layer_norm(residual_1 + a, self.gamma_attn, self.beta_attn)

        # MLP block (same as in single-head attention)
        residual_2 = a
        mlp_out = self.W_down(F.relu(self.W_up(a)))
        output = self.layer_norm(residual_2 + mlp_out, self.gamma_mlp, self.beta_mlp)
        # STUDENT END --------------------------------------------------------

        return output