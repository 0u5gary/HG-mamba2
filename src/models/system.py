import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import os  
import torchaudio 
import src.config as config
from src.utils.spec_audio_conversion import convert_to_audio_from_complex_spectrogram_after_compression_torch, convert_to_complex_spectrogram_with_compression_torch
from src.utils.denoiser import denoise_speech
from pytorch_lightning.utilities import grad_norm

class System(pl.LightningModule):
    def __init__(self, model, loss, metrics, fusion_method='gcn_mamba', condition_name='default', encoder_name='unknown'):
        super().__init__()
        self.model = model
        self.loss = loss
        self.metrics = nn.ModuleDict(metrics)
        self.fusion_method = fusion_method
        self.base_seed = 42
        self.error_files = []
        self.error_log_path = 'error_files.txt'
        self.test_dict = {
            'test_loss': [], 'test_pesq': [], 'test_sisdr': [], 
            'test_sisdr_gain': [], 'test_snr': [], 'test_snr_gain': [], 'test_estoi': []
        }
        self.test_step_outputs = []

        # Audio saving setup
        self.saved_counts = {}  # store how many samples saved per condition
        self.samples_per_condition = 20  # each condition saves up to 20 samples (adjustable)
        self.current_condition_name = condition_name
        self.encoder_name = encoder_name
        # base directory for saving audio samples
        self.save_audio_dir = config.PROJECT_ROOT + "/test_audio/" + fusion_method + "/" + encoder_name 

        os.makedirs(self.save_audio_dir, exist_ok=True)

    def forward(self, spec, face_embed, inference=False):
        if self.fusion_method == 'gcn':
            return self.model(spec, face_embed, inference=inference)
        else:
            return self.model(spec, face_embed)

    def common_step(self, batch, batch_idx, stage='train'):
        if batch is None: return None

        input_audio = batch['input_audio']
        mixed_audio = batch['mixed_audio']
        face_embed = batch['face_embed']
        
        with torch.no_grad():
            clean_audio_target = denoise_speech(input_audio)
    
        # Conversion
        mixed_mag_spec, mixed_phase = convert_to_complex_spectrogram_with_compression_torch(mixed_audio)
        clean_mag_spec, clean_phase = convert_to_complex_spectrogram_with_compression_torch(clean_audio_target)
        
        # Permute to (B, T, F) for model
        mixed_mag_spec = mixed_mag_spec.permute(0, 2, 1)
        mixed_phase = mixed_phase.permute(0, 2, 1)
        clean_mag_spec = clean_mag_spec.permute(0, 2, 1)
        clean_phase = clean_phase.permute(0, 2, 1)
        
        # Padding adjustments
        padded_length = mixed_mag_spec.size(1) - mixed_mag_spec.size(1) % 4
        mixed_mag_spec = mixed_mag_spec[:, :padded_length, :]
        mixed_phase = mixed_phase[:, :padded_length, :]
        clean_mag_spec = clean_mag_spec[:, :padded_length, :]
        clean_phase = clean_phase[:, :padded_length, :]
        
        # Prepare Target tuple
        clean_complex_spec_target = (clean_mag_spec, clean_phase)

        # --- Model Forward & Loss Logic ---
        loss = 0.0
        clean_audio_pred = None

        mag_spec_est = self.model(mixed_mag_spec, face_embed)
        clean_complex_spec_est = (mag_spec_est, mixed_phase)
        loss = self.loss(clean_complex_spec_est, clean_complex_spec_target)
        clean_audio_pred = convert_to_audio_from_complex_spectrogram_after_compression_torch(clean_complex_spec_est).to(input_audio.device)
            

        # --- Metrics ---
        clean_audio_target = clean_audio_target.detach().to(clean_audio_pred.device)
        with torch.no_grad():
            sisdr = self.metrics['sisdr'](clean_audio_pred, clean_audio_target)
            sisdr_gain = sisdr - self.metrics['sisdr'](mixed_audio, clean_audio_target)
            snr = self.metrics['snr'](clean_audio_pred, clean_audio_target)
            snr_gain = snr - self.metrics['snr'](mixed_audio, clean_audio_target)
            
        return loss, sisdr, sisdr_gain, snr, snr_gain, clean_audio_pred, clean_audio_target

    def training_step(self, batch, batch_idx):
        out = self.common_step(batch, batch_idx, stage='train')
        if out is None: return None
        loss, sisdr, sisdr_gain, snr, snr_gain, _, _ = out
        self.log_dict({"train/loss": loss, "train/sisdr": sisdr, "train/snr": snr})
        return loss
    
    def validation_step(self, batch, batch_idx):
        out = self.common_step(batch, batch_idx, stage='val')
        if out is None: return None
        loss, sisdr, sisdr_gain, snr, snr_gain, pred, target = out
        pesq = self.get_pesq(pred, target, batch)
        estoi = self.get_estoi(pred, target, batch)
        pesq = 0.0
        estoi = 0.0
        self.log_dict({"val/loss": loss, "val/sisdr": sisdr, "val/pesq": pesq})
        return loss
    
    def test_step(self, batch, batch_idx):
        out = self.common_step(batch, batch_idx, stage='test')
        if out is None: return None
        loss, sisdr, sisdr_gain, snr, snr_gain, pred, target = out
        pesq = self.get_pesq(pred, target, batch)
        estoi = self.get_estoi(pred, target, batch)

        self.test_dict["test_loss"].append(loss)
        self.test_dict["test_snr"].append(snr)
        self.test_dict["test_snr_gain"].append(snr_gain)
        self.test_dict["test_sisdr"].append(sisdr)
        self.test_dict["test_sisdr_gain"].append(sisdr_gain)
        self.test_dict["test_pesq"].append(pesq)
        self.test_dict["test_estoi"].append(estoi)


        # record detailed results for each sample in the batch (for CSV output later)
        if 'audio_fp' in batch:
            batch_size = pred.shape[0]
            for i in range(batch_size):
                filename = os.path.basename(batch['audio_fp'][i])
                speaker_id = batch['audio_fp'][i].split(os.sep)[-2]
                
                row = {
                    'filename': f"{speaker_id}_{filename}",
                    'pesq': pesq[i].item() if torch.is_tensor(pesq) and pesq.dim() > 0 else pesq,
                    'sisdr': sisdr[i].item() if torch.is_tensor(sisdr) and sisdr.dim() > 0 else sisdr,
                    'estoi': estoi[i].item() if torch.is_tensor(estoi) and estoi.dim() > 0 else estoi,
                }
                self.test_step_outputs.append(row)

        # Save Audio Samples
        if 'audio_fp' in batch:
            file_paths = batch['audio_fp']
            mixed_audios = batch['mixed_audio'] # save mixed audio for comparison
            condition_name = self.current_condition_name
            
            batch_size = pred.shape[0]
            
            for i in range(batch_size):
                full_path = file_paths[i]
                parts = full_path.split(os.sep)
                
                speaker_id = parts[-2]   
                original_name = parts[-1]
                filename = f"{speaker_id}_{original_name}"
                # print(f"Saving sample: {filename}")
  
                # Check if we have already saved enough samples for this condition
                if condition_name not in self.saved_counts:
                    self.saved_counts[condition_name] = 0
                
                if self.saved_counts[condition_name] < self.samples_per_condition:
                    save_dir = os.path.join(self.save_audio_dir, condition_name)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # 1. Enhanced audio (Predicted)
                    save_path_enh = os.path.join(save_dir, f"enhanced_{filename}")
                    torchaudio.save(save_path_enh, pred[i].cpu().unsqueeze(0), 16000)
                    
                    # 2. Mixed audio (Input) for comparison
                    save_path_mix = os.path.join(save_dir, f"mixed_{filename}")
                    torchaudio.save(save_path_mix, mixed_audios[i].cpu().unsqueeze(0), 16000)
                    
                    # 3. Clean audio (Target) for comparison
                    save_path_clean = os.path.join(save_dir, f"clean_{filename}")
                    torchaudio.save(save_path_clean, target[i].cpu().unsqueeze(0), 16000)
                    
                    self.saved_counts[condition_name] += 1
        
    def on_test_epoch_end(self):
        avg_loss = sum(self.test_dict["test_loss"]) / len(self.test_dict["test_loss"])
        avg_sisdr = sum(self.test_dict["test_sisdr"]) / len(self.test_dict["test_sisdr"])
        avg_sisdr_gain = sum(self.test_dict["test_sisdr_gain"]) / len(self.test_dict["test_sisdr_gain"])
        avg_snr = sum(self.test_dict["test_snr"]) / len(self.test_dict["test_snr"])
        avg_snr_gain = sum(self.test_dict["test_snr_gain"]) / len(self.test_dict["test_snr_gain"])
        avg_pesq = sum(self.test_dict["test_pesq"]) / len(self.test_dict["test_pesq"])
        avg_estoi = sum(self.test_dict["test_estoi"]) / len(self.test_dict["test_estoi"])
        
        # Save detailed test results to CSV
        if self.test_step_outputs:
            df = pd.DataFrame(self.test_step_outputs)
            
            csv_name = f"test_results_{self.current_condition_name}_{self.encoder_name}.csv"
            csv_path = os.path.join(self.save_audio_dir, csv_name)
            
            df.to_csv(csv_path, index=False)
            print(f"Detailed test results saved to: {csv_path}")
            
            self.test_step_outputs.clear()

        print(f"AVG Test Loss: {avg_loss}, Test SISDR: {avg_sisdr}, Test SISDR Gain: {avg_sisdr_gain}, Test SNR: {avg_snr}, Test SNR Gain: {avg_snr_gain}, Test PESQ: {avg_pesq}, Test estoi: {avg_estoi}")

        
        return avg_loss, avg_pesq, avg_sisdr, avg_sisdr_gain, avg_snr, avg_snr_gain, avg_estoi


    # Use Adam optimizer and ReduceLROnPlateau scheduler
    # if loss doesn't decrease for 1 epoch, reduce lr by half, with a minimum lr of 1e-7
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5) 
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=1,
            verbose=True,
            min_lr=1e-7
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
        
    def on_train_epoch_start(self):
        epoch_seed = self.base_seed + self.current_epoch 
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(epoch_seed)

    def get_pesq(self, audio_pred, audio_target, batch):
        try:
            return self.metrics['pesq'](audio_pred, audio_target)
        except Exception as e:
            return 0.0

    def get_estoi(self, audio_pred, audio_target, batch):
        try:
            return self.metrics['estoi'](audio_pred, audio_target)
        except Exception as e:
            return 0.0
            