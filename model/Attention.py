import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerSelfAttention2D(nn.Module):
    """
    Pure Transformer Self-Attention adapted for 2D feature maps
    Identical to transformer attention but processes spatial locations
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        x: [B, C, H, W] -> [B, C, H, W]
        """
        B, C, H, W = x.shape
        residual = x
        
        # Reshape to sequence: [B, H*W, C]
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        
        # Apply layer norm
        x = self.layer_norm(x)
        
        # Generate Q, K, V
        Q = self.w_q(x).view(B, H * W, self.num_heads, self.d_k).transpose(1, 2)  # [B, heads, H*W, d_k]
        K = self.w_k(x).view(B, H * W, self.num_heads, self.d_k).transpose(1, 2)  # [B, heads, H*W, d_k]
        V = self.w_v(x).view(B, H * W, self.num_heads, self.d_k).transpose(1, 2)  # [B, heads, H*W, d_k]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, heads, H*W, H*W]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, heads, H*W, d_k]
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, H * W, self.d_model)  # [B, H*W, C]
        
        # Output projection
        output = self.w_o(attn_output)  # [B, H*W, C]
        
        # Reshape back to [B, C, H, W]
        output = output.permute(0, 2, 1).view(B, C, H, W)
        
        # Residual connection
        return output + residual


class TransformerCrossAttention2D(nn.Module):
    """
    Enhanced Cross-Attention for FPN feature fusion
    Query attends to Key-Value with proper spatial handling
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_query = nn.LayerNorm(d_model)
        self.layer_norm_kv = nn.LayerNorm(d_model)
        
        # Add positional encoding for better spatial understanding
        self.register_buffer('pe_cache', None)
        
    def _get_positional_encoding(self, H, W, device):
        """Generate or retrieve cached positional encoding"""
        if self.pe_cache is None or self.pe_cache.shape != (1, self.d_model, H, W):
            pe = torch.zeros(1, self.d_model, H, W, device=device)
            
            position_h = torch.arange(0, H, dtype=torch.float32, device=device).unsqueeze(1)
            position_w = torch.arange(0, W, dtype=torch.float32, device=device).unsqueeze(0)
            
            div_term = torch.exp(torch.arange(0, self.d_model // 2, 2, dtype=torch.float32, device=device) * 
                               (-math.log(10000.0) / (self.d_model // 2)))
            
            pe[0, 0::4, :, :] = torch.sin(position_h * div_term.view(-1, 1, 1))[:self.d_model//4]
            pe[0, 1::4, :, :] = torch.cos(position_h * div_term.view(-1, 1, 1))[:self.d_model//4]
            pe[0, 2::4, :, :] = torch.sin(position_w * div_term.view(-1, 1, 1))[:self.d_model//4]
            pe[0, 3::4, :, :] = torch.cos(position_w * div_term.view(-1, 1, 1))[:self.d_model//4]
            
            self.pe_cache = pe
        
        return self.pe_cache
    
    def forward(self, query_feat, key_value_feat):
        """
        query_feat: [B, C, H, W] - the feature to be enhanced
        key_value_feat: [B, C, H', W'] - the feature to attend to
        Returns: [B, C, H, W] - enhanced query feature
        """
        B, C, H, W = query_feat.shape
        _, _, H_kv, W_kv = key_value_feat.shape
        
        # Add positional encoding
        pe_q = self._get_positional_encoding(H, W, query_feat.device)
        pe_kv = self._get_positional_encoding(H_kv, W_kv, key_value_feat.device)
        
        query_with_pe = query_feat + pe_q
        kv_with_pe = key_value_feat + pe_kv
        
        # Reshape to sequences
        query = query_with_pe.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        key_value = kv_with_pe.view(B, C, H_kv * W_kv).permute(0, 2, 1)  # [B, H'*W', C]
        
        # Apply layer norms
        query = self.layer_norm_query(query)
        key_value = self.layer_norm_kv(key_value)
        
        # Generate Q from query, K,V from key_value
        Q = self.w_q(query).view(B, H * W, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key_value).view(B, H_kv * W_kv, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(key_value).view(B, H_kv * W_kv, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, H * W, self.d_model)
        
        # Output projection
        output = self.w_o(attn_output)
        output = output.permute(0, 2, 1).view(B, C, H, W)
        
        # Residual connection
        return output + query_feat


class WindowedTransformerAttention(nn.Module):
    """
    Windowed Self-Attention (like Swin Transformer)
    More efficient for large feature maps
    """
    def __init__(self, d_model, num_heads=8, window_size=7, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        # Pad if necessary
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        _, _, H_pad, W_pad = x.shape
        
        # Partition into windows
        x = x.view(B, C, H_pad // self.window_size, self.window_size, 
                   W_pad // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, num_h, num_w, ws, ws, C]
        x = x.view(B * (H_pad // self.window_size) * (W_pad // self.window_size), 
                   self.window_size * self.window_size, C)  # [B*num_windows, ws*ws, C]
        
        # Apply transformer attention within each window
        x = self.layer_norm(x)
        
        Q = self.w_q(x).view(x.shape[0], x.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(x.shape[0], x.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(x.shape[0], x.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            x.shape[0], x.shape[1], self.d_model)
        
        output = self.w_o(attn_output)
        
        # Reshape back
        output = output.view(B, H_pad // self.window_size, W_pad // self.window_size,
                           self.window_size, self.window_size, C)
        output = output.permute(0, 5, 1, 3, 2, 4).contiguous()
        output = output.view(B, C, H_pad, W_pad)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :H, :W]
        
        return output + residual


class MultiScaleCrossAttention(nn.Module):
    """
    Advanced multi-scale cross-attention for FPN fusion
    Each scale can attend to all other scales with adaptive weights
    """
    def __init__(self, d_model, num_heads=8, num_scales=5, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_scales = num_scales
        
        # Cross-attention modules for each scale interaction
        self.cross_attentions = nn.ModuleList([
            TransformerCrossAttention2D(d_model, num_heads, dropout)
            for _ in range(num_scales)
        ])
        
        # Learnable attention weights for combining multi-scale features
        self.scale_weights = nn.Parameter(torch.ones(num_scales, num_scales))
        
        # Feature refinement after fusion
        self.refine_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GroupNorm(32, d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, 1),
        )
        
    def forward(self, features):
        """
        features: List of [B, C, H_i, W_i] feature maps
        Returns: List of enhanced feature maps
        """
        if len(features) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} features, got {len(features)}")
        
        enhanced_features = []
        
        # Normalize attention weights
        norm_weights = F.softmax(self.scale_weights, dim=1)
        
        for i, query_feat in enumerate(features):
            enhanced_feat = query_feat
            
            # Attend to all other scales
            for j, key_feat in enumerate(features):
                if i != j:
                    # Get the attention weight for this scale interaction
                    weight = norm_weights[i, j]
                    
                    # Apply cross-attention
                    cross_attended = self.cross_attentions[i](query_feat, key_feat)
                    enhanced_feat = enhanced_feat + weight * cross_attended
            
            # Refine the fused feature
            enhanced_feat = self.refine_conv(enhanced_feat)
            enhanced_features.append(enhanced_feat)
        
        return enhanced_features


class PositionalEncoding2D(nn.Module):
    """
    2D Positional encoding for transformer attention
    """
    def __init__(self, d_model, max_h=200, max_w=250):
        super().__init__()
        
        pe = torch.zeros(max_h, max_w, d_model)
        
        position_h = torch.arange(0, max_h, dtype=torch.float).unsqueeze(1).unsqueeze(2)
        position_w = torch.arange(0, max_w, dtype=torch.float).unsqueeze(0).unsqueeze(2)
        
        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * 
                           (-math.log(10000.0) / (d_model // 2)))
        
        pe[:, :, 0::4] = torch.sin(position_h * div_term)
        pe[:, :, 1::4] = torch.cos(position_h * div_term)
        pe[:, :, 2::4] = torch.sin(position_w * div_term)
        pe[:, :, 3::4] = torch.cos(position_w * div_term)
        
        self.register_buffer('pe', pe.permute(2, 0, 1).unsqueeze(0))  # [1, d_model, max_h, max_w]
    
    def forward(self, x):
        B, C, H, W = x.shape
        return x + self.pe[:, :, :H, :W]