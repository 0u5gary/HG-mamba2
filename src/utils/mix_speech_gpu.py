import random
import soundfile as sf
import hashlib
from src.utils.utils import crop_pad_audio
from pathlib import Path
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
from concurrent.futures import ProcessPoolExecutor
import src.config as config

# ================= settings =================
#SPLIT = "train" 
SPLIT = "val"

LRS3_CSV_PATH = config.PROJECT_ROOT + "/data/lrs3_split.csv"
DNS_CSV_PATH = config.PROJECT_ROOT + "/data/dns_noise_split.csv"

OUTPUT_DIR = os.path.join(config.PROJECT_ROOT, "LRS3_Mixed", SPLIT)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# probility settings for mixing strategies
PROB_DNS_NOISE = 0.34    
PROB_1_SPEAKER = 0.33    
PROB_3_SPEAKER = 0.33    
# ============================================

print(f"Loading LRS3 split from {LRS3_CSV_PATH}...")
lrs3_df = pd.read_csv(LRS3_CSV_PATH)
target_files_df = lrs3_df[(lrs3_df["split"] == SPLIT) & (lrs3_df["clean_path"].notna())]
target_files = target_files_df["clean_path"].tolist()
print(f"[{SPLIT}] in LRS3 (Target & Interference Source): {len(target_files)}")

print(f"Loading DNS split from {DNS_CSV_PATH}...")
dns_df = pd.read_csv(DNS_CSV_PATH)
dns_noises = dns_df[dns_df["split"] == SPLIT]["path"].tolist()
print(f"[{SPLIT}] in DNS: {len(dns_noises)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_random_interference(target_fp, all_files, n_speakers=1):
    """
    get n_speakers random interference file paths from all_files, excluding target_fp itself
    """
    interferers = []
    max_attempts = 20
    
    sample_seed = (hash(target_fp) + n_speakers) % (2**32)
    rng = np.random.default_rng(sample_seed)

    while len(interferers) < n_speakers:
        candidate = rng.choice(all_files)
        if candidate != target_fp and candidate not in interferers:
            interferers.append(candidate)
            
    return interferers

def get_random_dns_noise(target_fp, noise_list):
    """ get a random noise file path from noise_list """
    sample_seed = hash(target_fp) % (2**32)
    rng = np.random.default_rng(sample_seed)
    return rng.choice(noise_list)

def load_and_sum_audio(file_paths, orig_sr=16000, crop_length=5):
    """
    Load multiple audio files, crop/pad them to the same length, normalize, and sum them together.
    """
    summed_audio = None
    
    for fp in file_paths:
        audio_np = crop_pad_audio(fp, orig_sr, crop_length)
        audio_np = librosa.util.normalize(audio_np) 
        
        audio_tensor = torch.tensor(audio_np, device=device, dtype=torch.float32)
        
        if summed_audio is None:
            summed_audio = audio_tensor
        else:
            summed_audio += audio_tensor
            
    return summed_audio

def mix_hybrid(target_fp, noise_source, noise_type, snr_lower, snr_upper):
    """ 
    mix target with noise/interference according to the specified noise_type and SNR range.
    target_fp: file path of the target clean speech
    noise_source: according to noise_type, either a single noise file path (for 'dns') or a list of interference file paths (for '1spk' or '3spk')
    noise_type: 'dns', '1spk', '3spk'
    """
    stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    
    with torch.cuda.stream(stream):  
        target_np = crop_pad_audio(target_fp, 16000, 5)
        target_tensor = torch.tensor(target_np, device=device, dtype=torch.float32)

        if isinstance(noise_source, str):
            # Case A: DNS
            noise_np = crop_pad_audio(noise_source, 16000, 5)
            noise_np = librosa.util.normalize(noise_np)
            noise_tensor = torch.tensor(noise_np, device=device, dtype=torch.float32)
        elif isinstance(noise_source, list):
            # Case B: LRS3 1 or 3 speakers
            noise_tensor = load_and_sum_audio(noise_source, 16000, 5)
        
        dB_snr = torch.tensor(np.random.uniform(snr_lower, snr_upper), device=device)
        snr_factor = 10**(dB_snr / 10)
        
        target_power = torch.mean(target_tensor**2)
        noise_power = torch.mean(noise_tensor**2)
        
        # avoid divide by zero
        if noise_power == 0:
            return target_tensor

        scale_factor = torch.sqrt(target_power / (snr_factor * noise_power + 1e-8))
        mixed_audio = target_tensor + noise_tensor * scale_factor
        
        # avoid Clipping, use Peak Normalization
        max_val = torch.max(torch.abs(mixed_audio))
        if max_val > 1.0:
            mixed_audio = mixed_audio / max_val

    return mixed_audio

def process_file(target_fp):
    try:
        filename_part = target_fp.split(os.sep)
        filename = filename_part[-2] + "_" + filename_part[-1]
        # print(f"Processing: {filename}")
        output_filename = f"{filename}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        if os.path.exists(output_path):
            return 

        # mix strategy selection
        rand_val = np.random.random() # 0.0 ~ 1.0
        
        if rand_val < PROB_DNS_NOISE:
            noise_source = get_random_dns_noise(target_fp, dns_noises)
            noise_type = 'dns'
        elif rand_val < (PROB_DNS_NOISE + PROB_1_SPEAKER):
            noise_source = get_random_interference(target_fp, target_files, n_speakers=1)
            noise_type = '1spk'
        else:
            noise_source = get_random_interference(target_fp, target_files, n_speakers=3)
            noise_type = '3spk'

        lower_snr = getattr(config, 'TRAINING_LOWER_SNR', -10)
        upper_snr = getattr(config, 'TRAINING_UPPER_SNR', 10)

        mixed_tensor = mix_hybrid(target_fp, noise_source, noise_type, lower_snr, upper_snr)
        
        mixed_audio = mixed_tensor.cpu().numpy()
        sf.write(output_path, mixed_audio, 16000, format="wav")
        
    except Exception as e:
        print(f"Error processing {target_fp}: {e}")

def main():
    num_workers = min(os.cpu_count(), 8) 
    
    print(f"Processing: Split={SPLIT}")
    print(f"Strategy Distribution: DNS={PROB_DNS_NOISE}, 1-Spk={PROB_1_SPEAKER}, 3-Spk={PROB_3_SPEAKER}")
    print(f"Output dir: {OUTPUT_DIR}")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_file, target_files), total=len(target_files)))

    print("Mix Done.")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True) 
    os.environ["GLOG_minloglevel"] = "3"
    
    if not target_files:
        print("[Warning]: Found no target files to process.")
    else:
        main()