import os
import glob
from pathlib import Path
from tqdm import tqdm
import subprocess
from concurrent.futures import ProcessPoolExecutor

# ================= settings =================
# LRS3 splits (include pretrain, trainval, test)
SPLIT = "test"
LRS3_ROOT = "/your/path/to/lrs3/" + SPLIT + "/"

# Audio output directory settings
OUTPUT_SAME_DIR = False 
OUTPUT_ROOT = "/your/path/to/lrs3_audio_test/" + SPLIT + "/"
# ===========================================

def extract_audio(video_path):
    try:
        if OUTPUT_SAME_DIR:
            # /path/to/video.mp4 -> /path/to/video.wav
            output_path = str(Path(video_path).with_suffix('.wav'))
        else:
            # /LRS3/trainval/id/vid.mp4 -> OUTPUT_ROOT/trainval/id/vid.wav
            rel_path = os.path.relpath(video_path, LRS3_ROOT)
            output_path = os.path.join(OUTPUT_ROOT, rel_path)
            output_path = str(Path(output_path).with_suffix('.wav'))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if os.path.exists(output_path):
            return # Skip if exists

        # use ffmpeg to extract audio (mono, 16kHz)
        # -y: overwrite, -vn: disable video, -ac 1: mono, -ar 16000: 16k rate
        cmd = [
            'ffmpeg', 
            '-i', video_path, 
            '-vn', 
            '-ac', '1', 
            '-ar', '16000', 
            '-f', 'wav', 
            output_path,
            '-loglevel', 'error', 
            '-y'
        ]
        subprocess.run(cmd, check=True)
        
    except Exception as e:
        print(f"Error extracting {video_path}: {e}")

def main():
    print(f"Searching for mp4 files in: {LRS3_ROOT} ...")
    
    mp4_files = glob.glob(os.path.join(LRS3_ROOT, '**/*.mp4'), recursive=True)
    print(f"Found {len(mp4_files)} mp4 files, starting audio extraction...")

    num_workers = min(os.cpu_count(), 16)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(extract_audio, mp4_files), total=len(mp4_files)))

    print("Audio extraction completed!")

if __name__ == "__main__":
    main()