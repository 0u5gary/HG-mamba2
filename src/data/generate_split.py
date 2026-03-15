"""
Generate CSV files for LRS3 and DNS-Challenge dataset splits.
 
LRS3 Splitting Rules:
- Test Set: All assigned to 'test'.
- Pretrain Set: All assigned to 'train'.
- Trainval Set: All assigned to 'val'.
DNS-Challenge Noise Dataset Splitting Rules:
- Randomly split noise files into Train (80%), Val (10%), and Test (10%).

The generated CSV contains two columns: 'path' and 'split'.
"""

import os
import glob
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import src.config as config

# ================= setting paths =================
LRS3_ROOT = config.LRS3_FOLDER_PATH
DNS_NOISE_ROOT = config.DNS_FOLDER_PATH

# output csv path
OUTPUT_DIR = os.path.join(config.PROJECT_ROOT, "data")
# =================================================

def generate_lrs3_split(root_dir, output_path):
    print(f"Processing LRS3 dataset: {root_dir} ...")
    
    data = []
    
    # dataset structure: .../{subset}/{speaker_id}/{file}.wav
    target_subsets = ['pretrain', 'trainval', 'test']
    
    for subset in target_subsets:
        subset_root = os.path.join(root_dir, subset)
        
        if not os.path.exists(subset_root):
            print(f"[Warning] Folder not found: {subset_root}. Skipping {subset} subset.")
            continue
            
        # Get all speaker IDs (subfolders) in the current subset
        speaker_ids = [d for d in os.listdir(subset_root) if os.path.isdir(os.path.join(subset_root, d))]
        
        if not speaker_ids:
            print(f"[Warning] No speaker folders found in {subset}.")
            continue
            
        print(f"Found {len(speaker_ids)} speakers in {subset} subset.")
        
        speaker_split_map = {}
        if subset == 'pretrain':
            for spk in speaker_ids:
                speaker_split_map[spk] = 'train'
        elif subset == 'trainval':
            for spk in speaker_ids:
                speaker_split_map[spk] = 'val'
        elif subset == 'test':
            for spk in speaker_ids:
                speaker_split_map[spk] = 'test'

        file_count = 0
        for spk_id in speaker_ids:
            spk_path = os.path.join(subset_root, spk_id)
            current_split = speaker_split_map[spk_id]
            
            mp4_files = glob.glob(os.path.join(spk_path, "*.mp4"))
            
            for mp4_path in mp4_files:
                # get wav path (need modify)
                wav_path = mp4_path.replace("/corpus", "/nas165/0u5gary").replace("/lrs3","/lrs3_audio").replace(".mp4", ".wav")
                
                if not os.path.exists(wav_path):
                    print(f"[Skip] Missing video for {wav_path}")
                    continue
                
                filename = os.path.basename(wav_path)
                file_id = os.path.splitext(filename)[0]
                
                # use unique file ID: "SpeakerID_FileID"
                unique_id = f"{spk_id}_{file_id}"
                
                entry = {
                    'id': unique_id,           
                    'split': current_split,    
                    'clean_path': wav_path,    
                    'video_path': mp4_path,
                    'subset_source': subset
                }
                
                data.append(entry)
                file_count += 1

    if not data:
        print("[Error] Generation failed. Please check the dataset structure and paths.")
        return

    df = pd.DataFrame(data)
    df = df[['id', 'split', 'clean_path', 'video_path', 'subset_source']]
    df.to_csv(output_path, index=False)
    
    counts = df['split'].value_counts()

    print(f"\nLRS3 processing completed.")
    print(f"Split: Train={counts.get('train', 0)}, Val={counts.get('val', 0)}, Test={counts.get('test', 0)}")
    print(f"Save csv file path: {output_path}\n")

def generate_dns_noise_split(root_dir, output_path):
    print(f"Processing DNS Noise dataset: {root_dir} ...")
    
    noise_files = glob.glob(os.path.join(root_dir, '**/*.wav'), recursive=True)
    
    if not noise_files:
        print("[Error] No DNS noise files found (.wav).")
        return

    noise_files.sort()  # Ensure consistent order before shuffling
    print(f"Total found {len(noise_files)} noise files")

    random.seed(42)
    random.shuffle(noise_files)

    # 80% Train, 10% Val, 10% Test
    total = len(noise_files)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    data = []
    for i, f in enumerate(noise_files):
        if i < train_end:
            split = 'train'
        elif i < val_end:
            split = 'val'
        else:
            split = 'test'
        
        data.append({'path': f, 'split': split})

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    counts = df['split'].value_counts()
    print(f"\nDNS Noise processing completed.")
    print(f"Split: Train={counts.get('train', 0)}, Val={counts.get('val', 0)}, Test={counts.get('test', 0)}")
    print(f"Save csv file path: {output_path}\n")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    lrs3_csv_path = os.path.join(OUTPUT_DIR, "lrs3_split.csv")
    dns_csv_path = os.path.join(OUTPUT_DIR, "dns_noise_split_b.csv")
    
    generate_lrs3_split(LRS3_ROOT, lrs3_csv_path)
    generate_dns_noise_split(DNS_NOISE_ROOT, dns_csv_path)
    
    print("All CSV files generated successfully!")