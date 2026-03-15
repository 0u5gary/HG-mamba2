import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.audio_stream import Audio_Encoder
from src.models.video_stream import Video_Encoder



class Fusion(nn.Module):
    def __init__(self, embedding_size):
        super(Fusion, self).__init__()
        self.embedding_size = embedding_size
        self.audio_encoder = Audio_Encoder()
        self.video_encoder = Video_Encoder(self.embedding_size)
        self.input_dim = 257*8 + self.embedding_size
        
        # for fair comparison with other models, we will keep the same hidden dimension to 400
        self.input_proj = nn.Linear(self.input_dim, 400)

        self.lstm = nn.LSTM(400, 400, num_layers=1, bias=True, batch_first=True, bidirectional=False) # requires input of dim (N, frames, input_dim)
        self.fc1 = nn.Linear(400, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 600)
        self.mag_mask = nn.Linear(600, 257)
        
        # for fair comparison with other models, we add a skip connection from the input of the LSTM to the output
        self.skip_proj = nn.Linear(2056, 400)
        self.fusion_norm = nn.LayerNorm(400)


    def forward(self, mag_spec, face_embed, norm = False):
        
        
        audio_encoded = self.audio_encoder(mag_spec) # dim: (batch_size, 8, 500, 257) N, C, frames, bins
        
        audio_encoded = audio_encoded.permute(0, 2, 1, 3) # dim: (batch_size, 500, 8, 257) N, frames, C, bins
        audio_encoded = audio_encoded.reshape(audio_encoded.size(0), audio_encoded.size(1), -1) # dim: (batch_size, 500, 2056) N, frames, C*bins

        
        video_encoded = self.video_encoder(face_embed) # dim: (batch_size, 500, 512) N, frames, embed_dim
        
        # store original audio characteristics for Residual use
        identity_audio = audio_encoded.clone()

        if norm:
            audio_encoded = self.normalize(audio_encoded)  # Normalize audio
            video_encoded = self.normalize(video_encoded)  # Normalize video

        fusion = torch.cat((audio_encoded, video_encoded), dim=2) # dim: (batch_size, 500, 2568) N, frames, input_dim
        fusion = self.input_proj(fusion) # dim: (batch_size, 500, 400) N, frames, hidden_dim

        # Residual connection
        lstm_out, _ = self.lstm(fusion)
        skip_feat = self.skip_proj(identity_audio)
        lstm_out = self.fusion_norm(lstm_out + skip_feat)

        lstm_out = F.relu(self.fc1(lstm_out))
        lstm_out = F.relu(self.fc2(lstm_out))
        lstm_out = F.relu(self.fc3(lstm_out))
        
        mag_mask = self.mag_mask(lstm_out)

        # add sigmoid to limit values between 0 and 1
        mag_mask = torch.sigmoid(mag_mask)
        
        mag_spec_est = mag_mask * mag_spec # dim: (batch_size, 500, 257) N, frames, bins
        
        
        return mag_spec_est
    
    def normalize(self, x, norm='l2'):
        if norm == 'l2':
            return self.l2_normalize(x)
        elif norm == 'z_score':
            return self.z_score_normalization(x)
        else:
            return x
    
    def l2_normalize(self, x):
        return x / (torch.norm(x, p=2, dim=2, keepdim=True ) + 1e-8)
    
    def z_score_normalization(self, x):
        mean = x.mean(dim=2, keepdim=True)  
        std = x.std(dim=2, keepdim=True) + 1e-6  
        return (x - mean) / std
    

            
        
        
        
        
        