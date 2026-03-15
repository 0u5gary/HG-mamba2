import pytorch_lightning as pl
from torch.utils import data
from src.data.dataset import LRS3
import torch
import pandas as pd
import src.config as config

SPLIT_FILE_PATH = config.PROJECT_ROOT + "/data/lrs3_split.csv"

class LRS3DataModule(pl.LightningDataModule):

    def __init__(self, data_path, visual_encoder, embedding_size, batch_size=4, num_workers=2):
        super(LRS3DataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.visual_encoder = visual_encoder
        self.embedding_size = embedding_size
        self.data_path = data_path

    def setup(self, test_condition=None, test_snr=None):
        all_data = pd.read_csv(SPLIT_FILE_PATH)
        self.train_dataset = LRS3("train", self.data_path, self.visual_encoder, self.embedding_size, all_data)
        self.val_dataset = LRS3("val", self.data_path, self.visual_encoder, self.embedding_size, all_data)
        if test_condition and test_snr:
            self.test_dataset = LRS3("test", self.data_path, self.visual_encoder, self.embedding_size, all_data, test_condition, test_snr)


    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn_custom, persistent_workers=True, prefetch_factor=2)

    def val_dataloader(self, small=True):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn_custom, persistent_workers=True, prefetch_factor=2)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn_custom, persistent_workers=True, prefetch_factor=2)
    
    def collate_fn_custom(self, batch):
        # Remove None entries from the batch (e.g., due to failed data loading)
        batch = [x for x in batch if x is not None]
        batch = [x for x in batch if x['face_embed'] is not None]
        if len(batch) == 0:
            return None

        return {
            'face_embed': torch.stack([x['face_embed'] for x in batch]), 
            'input_audio': torch.stack([x['input_audio'] for x in batch]), 
            'mixed_audio': torch.stack([x['mixed_audio'] for x in batch]), 
            'audio_fp': [x['audio_fp'] for x in batch],
            'interfering_speaker_fp': [x['interfering_speaker_fp'] for x in batch],
        }
