import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
import src.config as config
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms.functional import center_crop

sys.path.append(config.TALKNET_PATH)
from talkNet import talkNet

# ================= settings =================
LRS3_CSV_PATH = config.PROJECT_ROOT + "/data/lrs3_split.csv"

#TARGET_SPLIT = "train" 
#TARGET_SPLIT = "val"
TARGET_SPLIT = "test"

OUTPUT_DIR_NAME = "LRS3_TalkNet_Features"
# Need modify on line 127 if your corpus is stored at a different location
# ============================================

class LRS3VideoDataset(Dataset):
    def __init__(self, csv_path, split):
        super().__init__()
        self.split = split
        print(f"Loading CSV from {csv_path}...")
        df = pd.read_csv(csv_path)
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.video_paths = self.df['video_path'].tolist()  
        print(f"[{split}] Found {len(self.video_paths)} videos.")

    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, index):
        fp = self.video_paths[index]
        
        try:
            # read_video return: (Video[T, H, W, C], Audio, Info)
            video_tensor_orig, _, info = torchvision.io.read_video(fp, pts_unit='sec')
            if video_tensor_orig.shape[0] == 0:
                print(f"Warning: Empty video {fp}")
                return None, fp

            # truncate or pad to 5 seconds
            video_tensor = fixed_5s_clip(video_tensor_orig)
            assert video_tensor.shape[0] == 125, f"Video {fp} has {video_tensor.shape[0]} frames (expected 125)"
            
            # RGB to Grayscale
            rgb_to_grayscale_weight = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)
            rgb_to_grayscale_weight = rgb_to_grayscale_weight.view(1, 1, 1, 3)
            
            # Sum over channel dimension and center crop to 112x112
            gray_video = torch.sum(video_tensor * rgb_to_grayscale_weight, dim=-1, keepdim=False)
            gray_video = center_crop(gray_video, [112, 112])
            
            return gray_video, fp
            
        except Exception as e:
            print(f"Error reading {fp}: {e}")
            return None, fp


class TalkNetBatchedPreprocessing:
    def __init__(
            self,
            model_path,
            csv_path,
            split,
            batch_size: int = 20, 
            num_workers: int = 8,
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ) -> None:
        super().__init__()
        self.split = split
        self.ds = LRS3VideoDataset(csv_path, split=self.split)
        
        # DataLoader
        self.dataloader = DataLoader(
            self.ds, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            drop_last=False, 
            shuffle=False,
            collate_fn=self.collate_fn_skip_none
        )
        
        # Load TalkNet Model
        print(f"Loading TalkNet model from {model_path} on {device}...")
        self.model = talkNet().to(device)
        self.model.loadParameters(model_path)
        self.model.eval()
        self.device = device

    def collate_fn_skip_none(self, batch):
        # filter out None entries (failed video reads)
        batch = list(filter(lambda x: x[0] is not None, batch))
        if len(batch) == 0:
            return None, None
        return torch.utils.data.dataloader.default_collate(batch)

    def extract_features(self):
        failed_text_path = os.path.join(config.PROJECT_ROOT, f"failed_LRS3_TalkNet_feat_{self.split}.txt")
        
        with torch.no_grad(): 
            with open(failed_text_path, 'w') as failed_txt:
                for video_tensors, fps in tqdm(self.dataloader, desc=f"Extracting {self.split}"):
                    
                    if video_tensors is None:
                        continue

                    video_tensors = video_tensors.to(self.device)
                    B, T, H, W = video_tensors.shape
                    processed_video = video_tensors.view(B*T, 1, 1, H, W)
                    processed_video = (processed_video / 255.0 - 0.4161) / 0.1688
                    video_embed = self.model.model.visualFrontend(processed_video)
                    video_embed = video_embed.view(B, T, -1) 
                    video_embed_np = video_embed.cpu().numpy()

                    for i, fp in enumerate(fps):
                        try:
                            # Need modify the output if you want to save the features to a different location
                            if "lrs3" in fp:
                                output_ft_path = config.PROJECT_ROOT + "/Visual_Feature/" + OUTPUT_DIR_NAME + "/" + self.split
                                if "pretrain" in fp:
                                    temp = "lrs3/pretrain"
                                elif "trainval" in fp:
                                    temp = "lrs3/trainval"
                                elif "test" in fp:
                                    temp = "lrs3/test"
                                temp = "/share/corpus/" + temp
                                output_ft_path = fp.replace(temp, output_ft_path)
                            output_ft_path = os.path.splitext(output_ft_path)[0] + ".npy"
                            
                            print(f"Saving feature to {output_ft_path}...")
                            os.makedirs(os.path.dirname(output_ft_path), exist_ok=True)
                            feat = video_embed_np[i]
                            assert feat.shape == (125, 512), f"Feature shape mismatch: {feat.shape}"
                            np.save(output_ft_path, feat)
                            
                        except Exception as save_err:
                            print(f"Error saving {fp}: {save_err}")
                            failed_txt.write(f"{fp}\n")


def fixed_5s_clip(video, frame_rate=25):
    """
    truncate or pad video to exactly 5 seconds (125 frames at 25 fps).
    """
    T, H, W, C = video.shape
    target_frames = 5 * frame_rate 

    if T >= target_frames:
        # truncate to first 125 frames
        return video[:target_frames, :, :, :]
    else:
        # pad with zeros to reach 125 frames       
        pad_length = target_frames - T
        padding = (0, 0, 0, 0, 0, 0, 0, pad_length)     
        return F.pad(video, padding, mode='constant', value=0)

def main():
    model_path = os.path.join(config.TALKNET_PATH, "pretrain_AVA.model")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    process = TalkNetBatchedPreprocessing(
        model_path=model_path,
        csv_path=LRS3_CSV_PATH,
        split=TARGET_SPLIT,
        batch_size=32,   
        num_workers=8    
    )
    
    print("Start extracting features...")
    process.extract_features()
    print("Extraction finished.")

if __name__ == "__main__":
    os.environ["GLOG_minloglevel"] ="2"
    main()
