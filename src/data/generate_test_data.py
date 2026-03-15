import pandas as pd
import numpy as np
from src.utils.utils import crop_pad_audio
import librosa
import hashlib
import os
import soundfile as sf
from tqdm import tqdm
import argparse
import src.config as config

# ================= settings =================
# CSV paths
LRS3_CSV_PATH = os.path.join(config.PROJECT_ROOT, "data/lrs3_split.csv")
DNS_CSV_PATH = os.path.join(config.PROJECT_ROOT, "data/dns_noise_split.csv")

# output directory
OUTPUT_ROOT = os.path.join(config.PROJECT_ROOT, "LRS3_Mixed/test")

# SAMPLE_N: Set to an integer (e.g., 1000) to generate only a portion; None to the entire.
SAMPLE_N = None 
# ============================================

def load_data():
    """Load test split"""
    print(f"Loading LRS3 split from {LRS3_CSV_PATH}...")
    lrs3_df = pd.read_csv(LRS3_CSV_PATH)
    test_clean_df = lrs3_df[lrs3_df["split"] == "test"].reset_index(drop=True)
    
    print(f"Loading DNS split from {DNS_CSV_PATH}...")
    dns_df = pd.read_csv(DNS_CSV_PATH)
    test_noise_df = dns_df[dns_df["split"] == "test"].reset_index(drop=True)
    
    return test_clean_df, test_noise_df

def mix_per_snr(target_speaker, all_interference, dB_snr):
    """ Mixing target speaker with interference (noise or other speakers) 
    based on the specified SNR."""
    
    if dB_snr == "mixed":
        dB_snr = np.random.uniform(-10, 10)
    else:
        dB_snr = int(dB_snr)
        
    snr_factor = 10**(dB_snr / 10)
    
    target_speaker_power = np.mean(target_speaker**2)
    all_interference_power = np.mean(all_interference**2)

    # Add a small epsilon to avoid division by zero
    if all_interference_power > 0:
        scale_factor = np.sqrt(target_speaker_power / (snr_factor * all_interference_power + 1e-8))
        mixed_audio = target_speaker + all_interference * scale_factor
    else:
        mixed_audio = target_speaker

    mixed_audio = librosa.util.normalize(mixed_audio)
    return mixed_audio

def generate_noise_condition(target_fp, noise_df, snr, orig_sr=16000, crop_length=5):
    """ Condition: Noise Only (LRS3 Clean + DNS Noise) """
    target_audio = crop_pad_audio(target_fp, orig_sr, crop_length)
    target_audio = librosa.util.normalize(target_audio)

    # generate seed based on target_fp
    seed = int(hashlib.md5(target_fp.encode()).hexdigest(), 16) % (10 ** 8)
    
    noise_fp = noise_df.sample(1, random_state=seed)["path"].values[0]
    noise_audio = crop_pad_audio(noise_fp, orig_sr, crop_length)
    noise_audio = librosa.util.normalize(noise_audio)
    
    mixed_audio = mix_per_snr(target_audio, noise_audio, snr)
    return mixed_audio

def generate_one_interfering_speaker_condition(target_fp, all_test_files, snr, orig_sr=16000, crop_length=5):
    """ Condition: 1 Interferer (LRS3 Clean + 1 Other LRS3) """
    target_audio = crop_pad_audio(target_fp, orig_sr, crop_length)
    target_audio = librosa.util.normalize(target_audio)
    
    # generate seed based on target_fp
    seed = int(hashlib.md5(target_fp.encode()).hexdigest(), 16) % (10 ** 8)
    
    # Randomly select 2 candidates and exclude the target
    candidates = all_test_files.sample(2, random_state=seed)["clean_path"].values
    interferer_fp = candidates[0] if candidates[0] != target_fp else candidates[1]
    interfering_audio = crop_pad_audio(interferer_fp, orig_sr, crop_length)
    interfering_audio = librosa.util.normalize(interfering_audio)
    
    mixed_audio = mix_per_snr(target_audio, interfering_audio, snr)
    return mixed_audio

def generate_three_interfering_speakers_condition(target_fp, all_test_files, snr, orig_sr=16000, crop_length=5):
    """ Condition: 3 Interferers (LRS3 Clean + 3 Other LRS3) """
    target_audio = crop_pad_audio(target_fp, orig_sr, crop_length)
    target_audio = librosa.util.normalize(target_audio)
    
    # generate seed based on target_fp
    seed = int(hashlib.md5(target_fp.encode()).hexdigest(), 16) % (10 ** 8)
    
    # Randomly select 4 candidates and exclude the target
    candidates = all_test_files.sample(4, random_state=seed)["clean_path"].values
    interferers = [fp for fp in candidates if fp != target_fp][:3]
    interfering_audios = [crop_pad_audio(fp, orig_sr, crop_length) for fp in interferers]
    interfering_audios = [librosa.util.normalize(audio) for audio in interfering_audios]
    combined_interference = np.sum(interfering_audios, axis=0)
    
    mixed_audio = mix_per_snr(target_audio, combined_interference, snr)
    return mixed_audio

def main(condition, snr):
    test_clean_df, test_noise_df = load_data()
    
    target_df = test_clean_df
    if SAMPLE_N is not None and len(target_df) > SAMPLE_N:
        target_df = target_df.sample(SAMPLE_N, random_state=42, replace=False)
        print(f"Sampling {SAMPLE_N} files for testing.")
    
    print(f"Generating {len(target_df)} files for condition: {condition}, SNR: {snr}")

    for i, row in tqdm(target_df.iterrows(), total=len(target_df)):
        target_fp = row['clean_path']
        file_id = row['id'] 
        
        try:
            if condition == "noise_only":
                mixed_audio = generate_noise_condition(target_fp, test_noise_df, snr)
            elif condition == "one_interfering_speaker":
                mixed_audio = generate_one_interfering_speaker_condition(target_fp, test_clean_df, snr)
            elif condition == "three_interfering_speakers":
                mixed_audio = generate_three_interfering_speakers_condition(target_fp, test_clean_df, snr)
            else:
                print(f"Unknown condition: {condition}")
                return

            # Output structure: LRS3_Mixed_Test / condition / snr / Sxxxx.wav
            output_dir = os.path.join(OUTPUT_ROOT, condition, str(snr))
            os.makedirs(output_dir, exist_ok=True)     
            output_path = os.path.join(output_dir, f"{file_id}.wav")
            sf.write(output_path, mixed_audio, 16000)
            
        except Exception as e:
            print(f"Error processing {target_fp}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LRS3 test data under various conditions.")
    parser.add_argument("--condition", type=str, required=True,
                        help='Condition(s), comma-separated: noise_only, one_interfering_speaker, three_interfering_speakers')
    parser.add_argument("--snr", type=str, required=True,
                        help='SNR value(s), comma-separated: 0, 5, 10, mixed')

    args = parser.parse_args()

    conditions = [c.strip() for c in args.condition.split(",")]
    snrs = [s.strip() for s in args.snr.split(",")]

    for condition in conditions:
        for snr in snrs:
            main(condition, snr)
