import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mamba_ssm import Mamba
from src.models.audio_stream import Audio_Encoder
from src.models.video_stream import Video_Encoder

class GraphInputProcessor(nn.Module):
    def __init__(self, audio_dim, visual_dim, hidden_dim, max_len=1000):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        
        # Positional Encoding
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, audio_feat, visual_feat):
        # audio_feat: [B, T, A_Dim]
        # visual_feat: [B, T, V_Dim]
        
        a_emb = self.audio_proj(audio_feat) 
        v_emb = self.visual_proj(visual_feat)
        
        seq_len = audio_feat.size(1)
        # Add PE (Handle case where seq_len > max_len roughly or just slice)
        if seq_len > self.pe.size(0):
            # Extend PE dynamically if needed, or assume max_len is enough
            pe = self.pe[:self.pe.size(0), :].unsqueeze(0) # Fallback to max
        else:
            pe = self.pe[:seq_len, :].unsqueeze(0)
        a_emb = a_emb + pe
        v_emb = v_emb + pe
        
        # Concatenate nodes: [B, 2T, H]
        # Nodes: 0~T-1 are Audio, T~2T-1 are Visual
        graph_nodes = torch.cat([a_emb, v_emb], dim=1) 
        return graph_nodes

class ResGCNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gcn_linear = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, raw_adj):
        # x: [B, N, H]
        # raw_adj: [N, N] (Binary)
        
        # Get Adjacency Matrix
        if raw_adj.dim() == 2:
            adj = raw_adj.unsqueeze(0) 
        else:
            adj = raw_adj

        # Graph Convolution (Batch Matmul)
        # [B, N, N] x [B, N, H] -> [B, N, H]
        support = torch.matmul(adj, x) 
        
        # Linear Transform
        out = self.gcn_linear(support)
        out = self.dropout(out)
        
        # Residual + Norm
        out = self.norm(out + x)
        out = self.relu(out) 
        return out


class ResGCNBackbone(nn.Module):
    def __init__(self, hidden_dim, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            ResGCNLayer(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x


class FusionGCN_Sequential(nn.Module):
    def __init__(self, embedding_size, hidden_dim=400, sequential_type='mamba'):
        super(FusionGCN_Sequential, self).__init__()
        self.embedding_size = embedding_size
        self.sequential_type = sequential_type.lower() if sequential_type else 'none'
        
        self.audio_encoder = Audio_Encoder()
        self.video_encoder = Video_Encoder(self.embedding_size)
        
        # Feature Dims
        self.audio_feat_dim = 2056 
        self.visual_feat_dim = self.embedding_size 
        
        self.input_proc = GraphInputProcessor(self.audio_feat_dim, self.visual_feat_dim, hidden_dim)
        self.backbone = ResGCNBackbone(hidden_dim, num_layers=4)

        # Optional Temporal Sequential Module after GCN
        if self.sequential_type == 'lstm':
            self.temporal_core = nn.LSTM(
                input_size=hidden_dim, hidden_size=hidden_dim,
                num_layers=1, batch_first=True, bidirectional=False, bias=True
            )
            self.rnn_norm = nn.LayerNorm(hidden_dim)
        elif self.sequential_type == 'mamba':
            if Mamba is None:
                raise ImportError("Mamba not installed.")
            self.temporal_core = Mamba(
                d_model=hidden_dim, d_state=64, d_conv=4, expand=2
            )
            self.mamba_norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"Unknown sequential_type: {sequential_type}")

        # 3FC Layers 
        self.fc1 = nn.Linear(hidden_dim, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 600)

        # Output Layer for Magnitude Mask 
        self.mag_mask = nn.Linear(600, 257) 

        self.skip_proj = nn.Linear(self.audio_feat_dim, 400)
        self.fusion_norm = nn.LayerNorm(400)

    def build_adjacency_matrix(self, T, device, max_hop=3):
        # Returns a BINARY mask of allowed connections (Causal & Cross-modal)
        # Normalization is now handled dynamically in the Pruning module
        num_nodes = 2 * T
        adj = torch.eye(num_nodes, device=device)
        
        # Cross-Modal
        for t in range(T):
            adj[t, t+T] = 1.0
            adj[t+T, t] = 1.0
            
        # Multi-Hop Temporal & Cross-Modal
        for t in range(T):
            for h in range(1, max_hop + 1): # h = 1, 2, 3
                prev_t = t - h
                next_t = t + h
                
                # --- Audio ---
                if prev_t >= 0:
                    adj[t, prev_t] = 1.0; adj[prev_t, t] = 1.0
                if next_t < T:
                    adj[t, next_t] = 1.0; adj[next_t, t] = 1.0
                
                # --- Visual ---
                if prev_t >= 0:
                    adj[t+T, prev_t+T] = 1.0; adj[prev_t+T, t+T] = 1.0
                if next_t < T:
                    adj[t+T, next_t+T] = 1.0; adj[next_t+T, t+T] = 1.0
                
                # --- Cross-Modal ---
                if prev_t >= 0:
                    adj[t, prev_t+T] = 1.0 # V(t-h) -> A(t)
                if next_t < T:
                    adj[t, next_t+T] = 1.0 # V(t+h) -> A(t)

        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        norm_adj = adj / degree 

        return norm_adj # Just 0s and 1s

    def forward(self, mag_spec, face_embed, dnsmos=None, inference=False):
        """
        mag_spec: [B, 257, T]
        """
        # --- Feature Extraction ---
        audio_encoded = self.audio_encoder(mag_spec) # [B, 8, T, 257]
        audio_encoded = audio_encoded.permute(0, 2, 1, 3) # [B, T, 8, 257]
        B, T, _, _ = audio_encoded.size()
        audio_encoded = audio_encoded.reshape(B, T, -1) # [B, T, 2056]
        
        # store original audio characteristics for Residual use
        identity_audio = audio_encoded.clone()

        video_encoded = self.video_encoder(face_embed) # [B, T, 512]
        
        audio_encoded = F.layer_norm(audio_encoded, audio_encoded.shape[2:])
        video_encoded = F.layer_norm(video_encoded, video_encoded.shape[2:])

        # --- Graph Processing ---
        nodes = self.input_proc(audio_encoded, video_encoded) # [B, 2T, H]
        
        # Build Static Heterogeneous Graph 
        raw_adj = self.build_adjacency_matrix(T, nodes.device)
        
        # Pass through Backbone 
        features = self.backbone(nodes, raw_adj) # [B, 2T, H]
        
        # Extract Audio Nodes for Temporal Refinement
        audio_nodes = features[:, :T, :] # [B, T, H]


        # Switch Temporal Refinement
        if self.sequential_type == 'lstm':
            core_out, _ = self.temporal_core(audio_nodes)
            audio_nodes = self.rnn_norm(audio_nodes + core_out) # Residual + Norm
            
        elif self.sequential_type == 'mamba':
            core_out = self.temporal_core(audio_nodes)
            audio_nodes = self.mamba_norm(audio_nodes + core_out) # Residual + Norm


        # --- 3FC Layers ---
        skip_feat = self.skip_proj(identity_audio)
        out = self.fusion_norm(audio_nodes + skip_feat)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out)) # [B, T, 600]

        # --- Predictions ---
        mag_mask = self.mag_mask(out)
        mag_mask = torch.sigmoid(mag_mask) # [B, T, 257]
        mag_spec_est = mag_mask * mag_spec
        return mag_spec_est
