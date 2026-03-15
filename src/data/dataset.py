import os
import numpy as np
import torch
from torch.utils import data
import pandas as pd
import librosa
import hashlib
from src.utils.utils import crop_pad_audio
from src.utils.augment_visual import augment_visual
import src.config as config

SAMPLING_RATE = 16000

class LRS3(data.Dataset):
    def __init__(self, split, data_path, visual_encoder, embedding_size, all_data=None, condition=None, snr=None):
        self.split = split
        self.data_path = data_path
        if all_data is None:
            split_path = os.path.join(config.PROJECT_ROOT, "data/lrs3_split.csv")
            all_data = pd.read_csv(split_path)
        self.data = all_data[all_data["split"] == self.split].reset_index(drop=True)
        self.visual_encoder = visual_encoder
        self.embedding_size = embedding_size

        self.embedding_path_dict = {
            "TalkNet": "LRS3_TalkNet_Features",
            "AVHuBERT": "LRS3_AVHuBERT_Features",
        }
        self.device = "cuda"
        self.condition = condition
        self.snr = snr     

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Audio processing
        clean_fp = row['clean_path']
        id = row['id']
        if self.split == "train" or self.split == "val":
            mixed_audio_fp = config.PROJECT_ROOT + f"/LRS3_Mixed/{self.split}/{id}.wav"
        elif self.split == "test" and self.condition and self.snr is not None:
            mixed_audio_fp = config.PROJECT_ROOT + f"/LRS3_Mixed/{self.split}/{self.condition}/{self.snr}/{id}.wav"
    
        if not os.path.exists(mixed_audio_fp):
            return None

        clean_audio = crop_pad_audio(clean_fp, SAMPLING_RATE, 5)
        mixed_audio = crop_pad_audio(mixed_audio_fp, SAMPLING_RATE, 5)
        clean_audio = librosa.util.normalize(clean_audio)
        mixed_audio = librosa.util.normalize(mixed_audio)
        clean_audio, mixed_audio = torch.tensor(clean_audio), torch.tensor(mixed_audio)
        
        if_speaker_fp = "None"
        
        # Visual Features processing
        face_embed = torch.zeros((125, self.embedding_size))
        if "_" in self.visual_encoder:     
            if "addition" in self.visual_encoder:
                face_embed = self._add_embeddings(id, self.visual_encoder)
            elif "concatenate" in self.visual_encoder:
                face_embed = self._concatenate_embeddings(id, self.visual_encoder)
        
            if face_embed is None:
                return None
        else:
            id = id.replace("_","/")
            fe_fp = config.PROJECT_ROOT + f"/Visual_Feature/{self.embedding_path_dict[self.visual_encoder]}/{self.split}/{id}.npy"
            
            if not os.path.exists(fe_fp):
                return None

            face_embed = np.load(fe_fp, mmap_mode="r")
            face_embed = self._crop_pad_face_embeddings(face_embed)
            face_embed = torch.tensor(face_embed)
            
            if face_embed == None:
                return None

            if self.split != "test":
                face_embed = augment_visual(face_embed, self.visual_encoder)[0]

        return {
            "face_embed": face_embed,
            "input_audio": clean_audio,
            "mixed_audio": mixed_audio,
            "audio_fp": clean_fp,
            "interfering_speaker_fp": if_speaker_fp
        }
    
    def _concatenate_embeddings(self, id, combined_features):
        
        fe1, fe2 = combined_features.split("_")[0], combined_features.split("_")[1]
        id = id.replace("_","/")
        fe_fp1 = config.PROJECT_ROOT + f"/Visual_Feature/{self.embedding_path_dict[fe1]}/{self.split}/{id}.npy"
        fe_fp2 = config.PROJECT_ROOT + f"/Visual_Feature/{self.embedding_path_dict[fe2]}/{self.split}/{id}.npy"            
        
        if not os.path.exists(fe_fp1) or not os.path.exists(fe_fp2):
            print(f"[Skip] Missing visual feature: {fe_fp1} or {fe_fp2}") 
            return None

        face_embed1 = np.load(fe_fp1, mmap_mode="r")
        face_embed2 = np.load(fe_fp2, mmap_mode="r")
        
        face_embed1, face_embed2 = self._crop_pad_face_embeddings(face_embed1), self._crop_pad_face_embeddings(face_embed2)
        
        face_embed1, face_embed2 = torch.tensor(face_embed1), torch.tensor(face_embed2)
        
        if self.split != "test":
            face_embed1 = augment_visual(face_embed1, fe1)[0]
            face_embed2 = augment_visual(face_embed2, fe2)[0]

        return torch.cat((face_embed1, face_embed2), dim=1)
    
    def _add_embeddings(self, id, combined_features):
        
        fe1, fe2 = combined_features.split("_")[0], combined_features.split("_")[1]
        id = id.replace("_","/")
        fe_fp1 = config.PROJECT_ROOT + f"/Visual_Feature/{self.embedding_path_dict[fe1]}/{self.split}/{id}.npy"
        fe_fp2 = config.PROJECT_ROOT + f"/Visual_Feature/{self.embedding_path_dict[fe2]}/{self.split}/{id}.npy"            
        
        if not os.path.exists(fe_fp1) or not os.path.exists(fe_fp2):
            print(f"[Skip] Missing visual feature: {fe_fp1} or {fe_fp2}")
            return None

        face_embed1 = np.load(fe_fp1, mmap_mode="r")
        face_embed2 = np.load(fe_fp2, mmap_mode="r")
        
        face_embed1, face_embed2 = self._crop_pad_face_embeddings(face_embed1), self._crop_pad_face_embeddings(face_embed2)
        
        face_embed1, face_embed2 = torch.tensor(face_embed1), torch.tensor(face_embed2)
        
        if self.split != "test":
            face_embed1 = augment_visual(face_embed1, fe1)[0]
            face_embed2 = augment_visual(face_embed2, fe2)[0]

        
        return face_embed1 + face_embed2
      
    def normalize(self, x, norm='l2'):
        if norm == 'l2':
            return self.l2_normalize(x)
        elif norm == 'z_score':
            return self.z_score_normalization(x)
        else:
            return
    
    def l2_normalize(self, x):
        return x / torch.norm(x, p=2, dim=1, keepdim=True)
    
    def z_score_normalization(self, x):
        mean = x.mean(dim=1, keepdim=True)  
        std = x.std(dim=1, keepdim=True) + 1e-6  
        return (x - mean) / std
      
    def _crop_pad_face_embeddings(self, fe):
        
        if len(fe) < 25*5:
            fe = np.pad(fe, ((0, 25*5 - len(fe)), (0,0)))
        fe = fe[:25*5]
        
        return fe
