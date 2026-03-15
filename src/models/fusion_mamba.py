import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.models.audio_stream import Audio_Encoder
from src.models.video_stream import Video_Encoder

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Error: 'mamba-ssm' not installed. Please run: pip install mamba-ssm")
    Mamba = None

# ==========================================
#  Fusion Sub-Modules (FiLM, LinearAttn, CrossAttn)
# ==========================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM)
    Audio = Gamma(Video) * Audio + Beta(Video)
    """
    def __init__(self, dim):
        super(FiLMLayer, self).__init__()
        self.dim = dim
        self.fc_gamma = nn.Linear(dim, dim)
        self.fc_beta = nn.Linear(dim, dim)

    def forward(self, audio, video):
        # audio: [B, T, D], video: [B, T, D]
        gamma = self.fc_gamma(video)
        beta = self.fc_beta(video)
        return audio * gamma + beta

class CrossAttentionLayer(nn.Module):
    """
    Standard Cross Attention
    Query = Audio, Key/Value = Video
    """
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio, video):
        # audio (Query): [B, T, D]
        # video (Key, Value): [B, T, D]
        attn_output, _ = self.multihead_attn(query=audio, key=video, value=video)
        # Residual + Norm
        return self.norm(audio + self.dropout(attn_output))

class LinearAttentionLayer(nn.Module):
    """
    Linear Attention (Efficient Attention) based on Katharopoulos et al.
    Uses elu() + 1 as kernel function to avoid Softmax(N^2) complexity.
    Query = Audio, Key/Value = Video
    """
    def __init__(self, dim, dropout=0.1):
        super(LinearAttentionLayer, self).__init__()
        self.dim = dim
        self.scale = 1 / math.sqrt(dim)
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio, video):
        # audio: [B, T, D], video: [B, T, D]
        Q = self.q_proj(audio)
        K = self.k_proj(video)
        V = self.v_proj(video)

        # Apply kernel function (elu + 1) to make them positive
        Q = F.elu(Q) + 1.0
        K = F.elu(K) + 1.0

        # Efficient Attention Formula: (Q * (K^T * V)) / (Q * K^T * 1)
        # Assuming casual masking is NOT needed for cross-modality fusion here, 
        # or we treat it as global context. 
        # For simplicity in linear attention, we compute over the sequence dimension T.
        
        # K: [B, T, D], V: [B, T, D] -> KV: [B, D, D]
        KV = torch.einsum("btd,bte->bde", K, V)
        
        # Compute denominator factor: sum of keys
        K_sum = K.sum(dim=1) # [B, D]
        
        # Numerator: [B, T, D] * [B, D, D] -> [B, T, D]
        attn_num = torch.einsum("btd,bde->bte", Q, KV)
        
        # Denominator: [B, T, D] * [B, D] -> [B, T, D] (element-wise broadcast)
        attn_denom = torch.einsum("btd,bd->btd", Q, K_sum) + 1e-6
        
        attn_out = attn_num / attn_denom
        attn_out = self.out_proj(attn_out)
        
        return self.norm(audio + self.dropout(attn_out))

class MambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mamba = Mamba(
            d_model=dim, 
            d_state=64,  
            d_conv=4,    
            expand=2     
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.mamba(x)) # Residual Connection

class FusionMamba(nn.Module):
    def __init__(self, embedding_size, hidden_dim=400, num_layers=1, fusion_type=None):
        super(FusionMamba, self).__init__()
        self.fusion_type = fusion_type
        self.embedding_size = embedding_size
        
        self.audio_encoder = Audio_Encoder()
        self.video_encoder = Video_Encoder(self.embedding_size)
        
        self.audio_feat_dim = 2056 
        self.video_feat_dim = self.embedding_size
        self.input_dim = self.audio_feat_dim + self.video_feat_dim # 2568

        if self.fusion_type is not None:
            # Projection Layers (Pre-Fusion)
            # For fair comparison with HG-mamba, we project audio/video features to the same hidden_dim 400
            self.input_proj = None  # No need for input projection since fusion layer will handle it
            self.audio_proj = nn.Linear(self.audio_feat_dim, hidden_dim)
            self.video_proj = nn.Linear(self.video_feat_dim, hidden_dim)

            # Fusion Module Selection
            if self.fusion_type == 'film':
                self.fusion_layer = FiLMLayer(hidden_dim)
            elif self.fusion_type == 'linear_attn':
                self.fusion_layer = LinearAttentionLayer(hidden_dim)
            elif self.fusion_type == 'cross_attn':
                self.fusion_layer = CrossAttentionLayer(hidden_dim)
            else:
                self.fusion_layer = None  # No explicit fusion layer, just concatenation

        else:      
            # Mamba expects a single input, so we will concatenate audio/video features and then project to hidden_dim
            self.input_proj = nn.Linear(self.input_dim, 400)
            self.audio_proj = None
            self.video_proj = None
            self.fusion_layer = None
        
        # Mamba Core (1 Layer, d_model=400, d_state=64)
        self.mamba_layer = MambaBlock(dim=400)

        # 3FC Output Layers (Exactly as LSTM version)
        self.fc1 = nn.Linear(400, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 600)
        self.mag_mask = nn.Linear(600, 257)

        self.skip_proj = nn.Linear(self.audio_feat_dim, 400)
        self.fusion_norm = nn.LayerNorm(400)

    def forward(self, mag_spec, face_embed, norm=False, inference=False):
        # Feature Extraction
        audio_encoded = self.audio_encoder(mag_spec)
        audio_encoded = audio_encoded.permute(0, 2, 1, 3)
        B, T, _, _ = audio_encoded.size()
        audio_encoded = audio_encoded.reshape(audio_encoded.size(0), audio_encoded.size(1), -1)
        video_encoded = self.video_encoder(face_embed) 
        
        # Store original audio characteristics for Residual use
        identity_audio = audio_encoded.clone()

        if norm:
            audio_encoded = self.normalize(audio_encoded)
            video_encoded = self.normalize(video_encoded)                 
        
        # Fusion Strategy
        if self.fusion_layer is not None:
            audio_vec = self.audio_proj(audio_encoded) # [B, T, 400]
            video_vec = self.video_proj(video_encoded) # [B, T, 400]
            x = self.fusion_layer(audio_vec, video_vec) # [B, T, 400]
        else:
            # Fusion (B, T, 2568)
            fusion = torch.cat((audio_encoded, video_encoded), dim=2)
            x = self.input_proj(fusion) # [B, T, 400]

        mamba_out = self.mamba_layer(x) # (B, T, 400)
        skip_feat = self.skip_proj(identity_audio)
        mamba_out = self.fusion_norm(mamba_out + skip_feat)

        # 4. 3FC Layers
        mamba_out = F.relu(self.fc1(mamba_out))
        mamba_out = F.relu(self.fc2(mamba_out))
        mamba_out = F.relu(self.fc3(mamba_out))
        
        # 5. Mask Generation
        mag_mask = self.mag_mask(mamba_out)
        mag_mask = torch.sigmoid(mag_mask)
        
        mag_spec_est = mag_mask * mag_spec

        return mag_spec_est
    
    def normalize(self, x, norm='l2'):
        if norm == 'l2':
            return self.l2_normalize(x)
        elif norm == 'z_score':
            return self.z_score_normalization(x)
        else:
            return x
    
    def l2_normalize(self, x):
        return x / (torch.norm(x, p=2, dim=2, keepdim=True) + 1e-8)
    
    def z_score_normalization(self, x):
        mean = x.mean(dim=2, keepdim=True)  
        std = x.std(dim=2, keepdim=True) + 1e-6  
        return (x - mean) / std