import os
import torch
import numpy as np
import pandas as pd
import sys
import src.config as config
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, get_worker_info
import torchvision
from pathlib import Path

sys.path.append(config.AVHUBERT_PATH)
sys.path.append(config.VSRIW_PATH)
sys.path.append("/your/path/to/this_project/src/data/Visual_Encoder/AV_HuBERT/fairseq")

# AV-HuBERT dependencies
import utils as avhubert_utils
from argparse import Namespace
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import Dictionary
from omegaconf import OmegaConf

# ============================================
# [HOTFIX] Fairseq Dictionary Monkey Patch
# Purpose: Fairseq strictly requires a dictionary file during task setup, 
#          even when we only need to extract the Visual Encoder skeleton.
# Action:  Intercepts `Dictionary.load`. If the file is missing, it returns 
#          a dummy dictionary to bypass the check and prevent runtime crashes.
# ============================================
_original_dictionary_load = Dictionary.load

def _dummy_dictionary_load(filename):
    if not os.path.exists(filename):
        d = Dictionary()
        d.add_symbol("dummy_symbol")
        return d
    return _original_dictionary_load(filename)

Dictionary.load = _dummy_dictionary_load
# ============================================

# VSRiW dependencies
from mediapipe.python.solutions.face_detection import FaceDetection, FaceKeyPoint
from pipelines.detectors.mediapipe.video_process import VideoProcess

# ================= settings =================
LRS3_CSV_PATH = config.PROJECT_ROOT + "/data/lrs3_split.csv"
#TARGET_SPLIT = "train"
#TARGET_SPLIT = "val"
TARGET_SPLIT = "test"

OUTPUT_DIR_NAME = "LRS3_AVHuBERT_Features"
AVHUBERT_CKPT_PATH = os.path.join(config.AVHUBERT_PATH, "conf/finetune/base_lrs3_433h.pt")
# Need modify on line 274 if your corpus is stored at a different location
# ============================================

class VideoTransform:
    def __init__(self, speed_rate):
        self.transform = avhubert_utils.Compose([
            avhubert_utils.Normalize(0.0, 255.0),
            avhubert_utils.CenterCrop((88, 88)),
            avhubert_utils.Normalize(0.421, 0.165)])
    def __call__(self, sample):
        return self.transform(sample)

class LRS3VideoDataset(Dataset):
    def __init__(self, csv_path, split, face_track: bool = True) -> None:
        super().__init__()
        print(f"Loading CSV from {csv_path}...")
        df = pd.read_csv(csv_path)
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.video_paths = self.df['video_path'].tolist()
        print(f"[{split}] Found {len(self.video_paths)} videos.")
        
        self.video_processor = VideoProcess()
        self.video_transform = VideoTransform(speed_rate=1)
        self.face_track = face_track
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, index):
        fp = self.video_paths[index]
        try:
            try:
                video_tensor = torchvision.io.read_video(fp, pts_unit='sec')[0].numpy()
            except Exception as e:
                 return None, fp, f"ReadVideoError: {str(e)}"
            if video_tensor.size == 0: return None, fp, "EmptyVideo"
            
            # Truncate or pad to 5 seconds (125 frames at 25fps)
            T, H, W, C = video_tensor.shape
            target_frames = 125
            if T >= target_frames:
                video_tensor = video_tensor[:target_frames]
            else:
                pad_width = ((0, target_frames - T), (0, 0), (0, 0), (0, 0))
                video_tensor = np.pad(video_tensor, pad_width, mode='constant')

            landmarks = self._process_landmarks(video_tensor)
            if landmarks is None: return None, fp, "NoFaceDetected"

            video_tensor = self.video_processor(video_tensor, landmarks)
            video_tensor = self.video_transform(torch.tensor(video_tensor))
            return video_tensor, fp, "Success"
        except Exception as e:
            return None, fp, f"UnknownException: {str(e)}"
        
    def _initialize_detector(self):
        self.short_range_detector = FaceDetection(min_detection_confidence=0.5, model_selection=0)
        self.long_range_detector = FaceDetection(min_detection_confidence=0.5, model_selection=1)

    def _process_landmarks(self, video_tensor):
        if self.face_track:
            landmarks = self._detect(video_tensor, self.long_range_detector)
            if all(l is None for l in landmarks):
                landmarks = self._detect(video_tensor, self.short_range_detector)
                if all(l is None for l in landmarks): return None
            return landmarks
        return None
    
    def _detect(self, video_frames, detector):
        landmarks = []
        for frame in video_frames:
            results = detector.process(frame)
            if not results.detections:
                landmarks.append(None)
                continue
            face_points = []
            for idx, detected_faces in enumerate(results.detections):
                bboxC = detected_faces.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                lmx = [
                    [int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(0).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(0).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(1).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(1).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(2).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(2).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(3).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(3).value].y * ih)],
                ]
                face_points.append(lmx)
            if face_points:
                landmarks.append(np.array(face_points[0]))
            else:
                landmarks.append(None)
        return landmarks

def video_dataset_worker_init(worker_id):
    info = get_worker_info()
    info.dataset._initialize_detector()

class AVHuBERTBatchedPreprocessing:
    def __init__(
            self,
            csv_path,
            split,
            batch_size = 16, 
            num_workers = 8,
            face_track: bool = True, 
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ) -> None:
        super().__init__()
        self.split = split
        self.ds = LRS3VideoDataset(csv_path, split=self.split, face_track=face_track)
        
        self.dataloader = DataLoader(
            self.ds, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            drop_last=False, 
            worker_init_fn=video_dataset_worker_init,
            collate_fn=self.collate_fn_robust
        )
        self.device = device
        
        print(f"Loading AV-HuBERT from {AVHUBERT_CKPT_PATH}...")
        if not os.path.exists(AVHUBERT_CKPT_PATH):
            raise FileNotFoundError(f"Checkpoint not found: {AVHUBERT_CKPT_PATH}")
        state = checkpoint_utils.load_checkpoint_to_cpu(AVHUBERT_CKPT_PATH)
        cfg = state["cfg"]
        
        # Force the dimension to be set to 768 to conform to the AV-HuBERT
        OmegaConf.set_struct(cfg.model, False)
        cfg.model.encoder_embed_dim = 768
        cfg.model.decoder_embed_dim = 768
        OmegaConf.set_struct(cfg.model, True)

        # build model skeleton
        utils.import_user_module(Namespace(user_dir=config.AVHUBERT_PATH))
        task = tasks.setup_task(cfg.task)
        model = task.build_model(cfg.model)

        # only extract Visual Encoder's weights
        visual_encoder = model.encoder.w2v_model.feature_extractor_video
        
        # search Key with prefix "encoder.w2v_model.feature_extractor_video." and remove the prefix to load into visual_encoder
        prefix = "encoder.w2v_model.feature_extractor_video."
        visual_state_dict = {} 
        for key, value in state["model"].items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                visual_state_dict[new_key] = value
        
        if len(visual_state_dict) == 0:
            raise RuntimeError("No matching keys found for Visual Encoder in the checkpoint. Please check the checkpoint structure.")

        # Load the filtered state dict into the visual encoder
        visual_encoder.load_state_dict(visual_state_dict, strict=True)
        print(f"Successfully loaded Visual Encoder weights ({len(visual_state_dict)} keys).")
        
        self.model = visual_encoder.to(self.device)
        self.model.eval()

    def collate_fn_skip_none(self, batch):
        batch = list(filter(lambda x: x[0] is not None, batch))
        if len(batch) == 0: return None, None
        return torch.utils.data.dataloader.default_collate(batch)

    def collate_fn_robust(self, batch):
        """
        Returns:
            batched_tensors: Tensor (B, ...) or None (if all samples failed)
            valid_fps: List[str] or None
            failed_fps: List[str] with reason, e.g. [(fp, reason), ...]
        """
        valid_batch = []
        failed_fps = []

        for video_tensor, fp, reason in batch:
            if video_tensor is None:
                failed_fps.append((fp, reason))
            else:
                valid_batch.append((video_tensor, fp))

        if len(valid_batch) > 0:
            tensors, valid_fps = zip(*valid_batch)
            batched_tensors = torch.stack(tensors)
        else:
            batched_tensors = None
            valid_fps = None

        return batched_tensors, valid_fps, failed_fps

    def extract_features(self):
        failed_text_path = os.path.join(config.PROJECT_ROOT, f"failed_avhubert_feat_{self.split}.txt")
        
        with open(failed_text_path, 'w') as failed_txt:
            for video_tensors, valid_fps, failed_info in tqdm(self.dataloader, desc=f"Extracting AVHuBERT {self.split}"):
                
                if failed_info:
                    for fp, reason in failed_info:
                        failed_txt.write(f"{fp},{reason}\n")
                    failed_txt.flush()
                if video_tensors is None: continue
                
                try:
                    video_tensors = video_tensors.to(self.device)
                    # Input: (B, T, H, W) -> (B, 1, T, H, W)
                    if video_tensors.ndim == 4: 
                        video_tensors = video_tensors.unsqueeze(1) 
                    elif video_tensors.ndim == 5 and video_tensors.shape[-1] == 1:
                         video_tensors = video_tensors.permute(0, 4, 1, 2, 3) 
                    
                    with torch.no_grad():
                        enc_feats: torch.Tensor = self.model(video_tensors)
                        if enc_feats.ndim == 3:
                             if enc_feats.shape[1] == 768:
                                 enc_feats = enc_feats.transpose(1, 2)

                    features_np = enc_feats.detach().cpu().numpy()
                    
                    for i, fp in enumerate(valid_fps):
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
                        os.makedirs(os.path.dirname(output_ft_path), exist_ok=True)
                        np.save(output_ft_path, features_np[i])

                except Exception as e:
                    print(f"Inference Error batch start {valid_fps[0]}: {e}")
                    for fp in valid_fps:
                        failed_txt.write(f"{fp},InferenceError: {e}\n")
                    failed_txt.flush()

def main():
    if not os.path.exists(AVHUBERT_CKPT_PATH):
        print(f"Error: Model not found at {AVHUBERT_CKPT_PATH}")
        return

    process = AVHuBERTBatchedPreprocessing(
        csv_path=LRS3_CSV_PATH,
        split=TARGET_SPLIT,
        batch_size=32, 
        num_workers=8,
        face_track=True
    )
    
    print("Start extracting AV-HuBERT features...")
    process.extract_features()
    print("Extraction finished.")

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["GLOG_minloglevel"] ="1"
    main()
