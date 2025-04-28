import torch

class RelativePositionEncoding(torch.nn.Module):
    def __init__(self, model_dim: int, max_distance: int = 32, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_distance = max_distance
        self.rel_embeddings = torch.nn.Embedding(2 * max_distance + 1, model_dim)
    
    def get_relative_positions(self, seq_len: int, device: torch.device):
        positions = torch.arange(seq_len, device=device)

        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)

        rel_pos = torch.clamp(rel_pos, -self.max_distance, self.max_distance) + self.max_distance

        return rel_pos
    
    def forward(self, seq_len: int, device: torch.device):
        rel_pos = self.get_relative_positions(seq_len, device)
        rel_pos_embed = self.rel_embeddings(rel_pos)
        return rel_pos_embed

class RelativeAttention(torch.nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1, max_distance: int = 32, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert self.head_dim * num_heads == model_dim

        self.query = torch.nn.Linear(model_dim, model_dim)
        self.key = torch.nn.Linear(model_dim, model_dim)
        self.value = torch.nn.Linear(model_dim, model_dim)

        self.output = torch.nn.Linear(model_dim, model_dim)

        self.dropout = torch.nn.Dropout(dropout)

        self.rel_pos_encoding = RelativePositionEncoding(self.head_dim, max_distance)

        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, _ = x.shape
        device = x.device

        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [batch, heads, seq, head_dim]
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)   # [batch, heads, seq, head_dim]
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [batch, heads, seq, head_dim]

        rel_pos_embed = self.rel_pos_encoding(seq_len, device)

        # Content-content attention: standard dot-product attention
        content_content = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, heads, seq, seq]

        # Content-position attention: how each query attends to relative positions
        # Reshape q for broadcasting: [batch, heads, seq, 1, head_dim]
        q_expanded = q.unsqueeze(-2)
        # Reshape rel_pos_embed for broadcasting: [1, 1, seq, seq, head_dim]
        rel_pos_expanded = rel_pos_embed.unsqueeze(0).unsqueeze(0)
        # Calculate content-position attention
        content_position = torch.sum(q_expanded * rel_pos_expanded, dim=-1) * self.scale  # [batch, heads, seq, seq]

        # Combine content-content and content-position attention
        attention_logits = content_content + content_position

        # Apply mask if provided
        if mask is not None:
            # Expand mask for batch and heads: [batch, 1, 1, seq]
            mask = mask.unsqueeze(1).unsqueeze(2)
            attention_logits = attention_logits.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attention_weights = torch.nn.functional.softmax(attention_logits, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)  # [batch, heads, seq, head_dim]
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)
        output = self.output(context)
        
        return output
    
class TransformerLayerWithRelativeAttention(torch.nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1, max_distance: int = 32):
        super().__init__()
        
        # Relative self-attention
        self.self_attn = RelativeAttention(model_dim, num_heads, dropout, max_distance)
        
        # Feed-forward network
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(model_dim, 4 * model_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(4 * model_dim, model_dim)
        )
        
        # Layer normalization
        self.norm1 = torch.nn.LayerNorm(model_dim)
        self.norm2 = torch.nn.LayerNorm(model_dim)
        
        # Dropout
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        mlp_output = self.mlp(x)
        x = x + self.dropout(mlp_output)
        x = self.norm2(x)
        
        return x