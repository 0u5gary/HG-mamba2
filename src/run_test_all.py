import subprocess
import os
import glob
import re
import pandas as pd
import sys
import src.config as config

# ================= setting =================
# 1. test visual Encoder 
ENCODERS = [
    # "AVHuBERT",
    # "TalkNet",
    "AVHuBERT_TalkNet_concatenate", # main test model
]

# 2. test Condition
CONDITIONS = [
    "noise_only",
    "one_interfering_speaker",
    "three_interfering_speakers"
]

# 3. test SNR
SNRS = ["10",
        "5", 
        "0", 
        "mixed"]

# 4. path & script
CKPT_ROOT = config.PROJECT_ROOT + "/checkpoints"    # Checkpoint dir
TEST_SCRIPT = "test.py"                             # script
OUTPUT_CSV = "gcn_mamba(max10).csv"                      # output CSV file name
FUSION_METHOD = "gcn_mamba"                              # fusion method
# ===========================================

def find_best_checkpoint(encoder_name):
    """
        Search for best-epoch=*.ckpt in possible directories for the given encoder_name.
    """
    possible_dir_names = [
        #f"{encoder_name}_lstm_LRS3",
        f"{encoder_name}_gcn_mamba_LRS3",
        #f"{encoder_name}_mamba_LRS3",
        #f"{encoder_name}_mamba_film_LRS3",
        #f"{encoder_name}_mamba_linear_LRS3",
        #f"{encoder_name}_mamba_cross_LRS3",
        #encoder_name
    ]
    
    for dir_name in possible_dir_names:
        # search for best-epoch=*.ckpt
        search_path = os.path.join(CKPT_ROOT, dir_name, "best-epoch=*.ckpt")
        ckpts = glob.glob(search_path)
        
        if ckpts:
            # if multiple checkpoints found, sort by epoch number (assuming format best-epoch=XX.ckpt)
            best_ckpt = sorted(ckpts)[-1]
            return best_ckpt
            
    return None

def parse_metrics_from_log(log_output):
    """
    Use regex to extract metrics from the log output.
    expected format in log:
    AVG Test Loss: 0.080, Test SISDR: 10.682, Test SISDR Gain: 10.512, ...
    """
    metrics = {}
    
    # define patterns for each metric we want to extract
    # Format: "keyword: result"
    patterns = {
        "Loss": r"AVG Test Loss:\s*(-?\d+\.?\d*)",
        "SISDR": r"Test SISDR:\s*(-?\d+\.?\d*)",
        "SISDR_Gain": r"Test SISDR Gain:\s*(-?\d+\.?\d*)",
        "OUTPUT_SNR": r"Test SNR:\s*(-?\d+\.?\d*)",
        "SNR_Gain": r"Test SNR Gain:\s*(-?\d+\.?\d*)",
        "PESQ": r"Test PESQ:\s*(-?\d+\.?\d*)",
        "ESTOI": r"Test estoi:\s*(-?\d+\.?\d*)"
    }
    
    for key, pattern in patterns.items():
        # search for all occurrences of the pattern in the log output
        matches = re.findall(pattern, log_output)
        if matches:
            metrics[key] = float(matches[-1])
        else:
            metrics[key] = None
            
    return metrics

def run_batch_test():
    all_results = []    
    
    for encoder in ENCODERS:
        ckpt_path = find_best_checkpoint(encoder)
        
        if not ckpt_path:
            print(f"\n  [Skip] Not found {encoder}'s checkpoint.")
            continue
            
        print(f"\n{'='*60}")
        print(f"Test Visual Encoder: {encoder}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        # test all conditions and SNRs
        for condition in CONDITIONS:
            for snr in SNRS:
                print(f"▶️  Running: {condition} | SNR={snr} ... ", end="", flush=True)
                if condition == "noise_only" and snr != "mixed":
                    print("Skipping (noise_only only supports mixed SNR)")
                    continue
                if condition == "one_interfering_speaker" and snr == "mixed":
                    print("Skipping (one_interfering_speaker does not support mixed SNR)")
                    continue
                if condition == "three_interfering_speakers" and snr == "mixed":
                    print("Skipping (three_interfering_speakers does not support mixed SNR)")
                    continue

                cmd = [
                    "python", TEST_SCRIPT,
                    "--visual_encoder", encoder,
                    "--ckpt_path", ckpt_path,
                    "--test_condition", condition,
                    "--test_snr", snr,
                    "--fusion_method", FUSION_METHOD,
                ]
                
                try:
                    # process script and capture output
                    result = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True, 
                        text=True            
                    )
                    
                    log_output = result.stdout + result.stderr
                    metrics = parse_metrics_from_log(log_output)
                    
                    # check if SISDR is parsed successfully
                    if metrics.get("SISDR") is not None:
                        print(f"Complete (SISDR: {metrics['SISDR']:.2f})")
                        
                        row = {
                            "Encoder": encoder,
                            "Condition": condition,
                            "Target_SNR": snr,
                            "Checkpoint": os.path.basename(ckpt_path),
                            **metrics
                        }
                        all_results.append(row)
                    else:
                        print("Metrics Parse Failed (SISDR not found)")
                        row = {
                            "Encoder": encoder, "Condition": condition, "Target_SNR": snr,
                            "Error": "Metrics Parse Failed"
                        }
                        all_results.append(row)

                except subprocess.CalledProcessError as e:
                    print(f"Processed with error (Exit Code: {e.returncode})")

    # save CSV
    if all_results:
        df = pd.DataFrame(all_results)
        
        cols = ["Encoder", "Condition", "Target_SNR", "PESQ", "SISDR", "ESTOI", "SISDR_Gain", "Output_SNR", "SNR_Gain", "Loss", "Checkpoint"]
        cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
        df = df[cols]
        
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n All test done! Results will be saved to: {os.path.abspath(OUTPUT_CSV)}")
    else:
        print("\n  No results to save.")

if __name__ == "__main__":
    run_batch_test()
