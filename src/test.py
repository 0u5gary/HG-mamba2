import argparse
from src.models.system import System
from src.models.fusion_lstm import Fusion
from src.models.fusion_gcn_seq import FusionGCN_Sequential
from src.models.fusion_mamba import FusionMamba
from src.data.datamodule import LRS3DataModule
from src.losses.complex_mse import PSA_MSE
from pytorch_lightning import Trainer
from torchmetrics.audio import SignalNoiseRatio, PerceptualEvaluationSpeechQuality, ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
import torch.nn as nn
import os
import multiprocessing
import warnings
import src.config as config


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


def test(visual_encoder, ckpt_path, test_condition, test_snr, embedding_size, batch_size, num_workers, fusion_method='lstm'):
    datamodule = LRS3DataModule(
        data_path=config.LRS3_FOLDER_PATH,
        visual_encoder=visual_encoder,
        embedding_size=embedding_size,
        batch_size=batch_size,
        num_workers=num_workers
    )
    datamodule.setup(test_condition=test_condition, test_snr=test_snr)
    test_loader = datamodule.test_dataloader()

    # --- Model Selection Logic ---
    if fusion_method == 'lstm':
        print("Initializing Standard LSTM Fusion Model...")
        fusion_network = Fusion(embedding_size=embedding_size)
    elif fusion_method == 'gcn':
        print("Initializing Graph Fusion (GCN) Model...")
        fusion_network = FusionGCN_Sequential(embedding_size=embedding_size, hidden_dim=400, sequential_type=None)
    elif fusion_method == 'gcn_mamba':
        print("Initializing Graph Fusion (GCN + Mamba) Model...")
        fusion_network = FusionGCN_Sequential(embedding_size=embedding_size, hidden_dim=400, sequential_type='mamba')
    elif fusion_method == 'gcn_lstm':
        print("Initializing Graph Fusion (GCN + LSTM) Model...")
        fusion_network = FusionGCN_Sequential(embedding_size=embedding_size, hidden_dim=400, sequential_type='lstm')
    elif fusion_method == 'mamba':
        # [Ablation] Pure Mamba
        print("Initializing Baseline: FusionMamba...")
        fusion_network = FusionMamba(embedding_size=embedding_size, hidden_dim=400)
    elif fusion_method == 'mamba_film':
        # [Ablation] film Mamba
        print("Initializing Baseline: FusionMamba_film...")
        fusion_network = FusionMamba(embedding_size=embedding_size, hidden_dim=400, fusion_type='film')
    elif fusion_method == 'mamba_linear':
        # [Ablation] linear Mamba
        print("Initializing Baseline: FusionMamba_linear_attention...")
        fusion_network = FusionMamba(embedding_size=embedding_size, hidden_dim=400, fusion_type='linear_attn')
    elif fusion_method == 'mamba_cross':
        # [Ablation] cross Mamba
        print("Initializing Baseline: FusionMamba_cross_attention...")
        fusion_network = FusionMamba(embedding_size=embedding_size, hidden_dim=400, fusion_type='cross_attn')
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")

    loss = PSA_MSE()
    metrics = {
        'snr': SignalNoiseRatio(),
        'pesq': PerceptualEvaluationSpeechQuality(16000, 'wb'),
        'sisdr': ScaleInvariantSignalDistortionRatio(),
        'estoi': ShortTimeObjectiveIntelligibility(16000, True)
    }

    condition_folder_name = f"{test_condition}_{test_snr}"
    model = System(fusion_network, loss, metrics, fusion_method=fusion_method, condition_name=condition_folder_name, encoder_name=visual_encoder)
    
    trainer = Trainer(accelerator='gpu', devices=1, logger=False) 
    trainer.test(model, dataloaders=test_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AV fusion model")

    parser.add_argument("--visual_encoder", type=str, required=True,
                        choices=config.TEST_VISUAL_ENCODERS,
                        help="Visual encoder to use")
    parser.add_argument("--ckpt_path", type=str, required=True, 
                        help="Path to model checkpoint")
    parser.add_argument("--test_condition", type=str, default=config.TEST_CONDITION,
                        choices=config.TEST_ALL_CONDITIONS,
                        help="Test condition")
    parser.add_argument("--test_snr", type=str, default=config.TEST_SNR,
                        choices=config.TEST_ALL_SNRs,
                        help="Test SNR value")
    parser.add_argument("--embedding_size", type=int,
                        help="Embedding size; defaults to encoder-specific value")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for DataLoader")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for DataLoader")
    parser.add_argument("--fusion_method", type=str, default="gcn_mamba", 
                    choices=["lstm", "gcn", "gcn_mamba", "gcn_lstm", "mamba", "mamba_film", "mamba_linear", "mamba_cross"])

    args = parser.parse_args()

    embedding_size = args.embedding_size or config.embedding_size_dict[args.visual_encoder]
    print(f"Using embedding size: {embedding_size}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()

    os.environ["GLOG_minloglevel"] = "3"
    
    try:
        multiprocessing.set_start_method('spawn', True)
    except RuntimeError:
        pass

    test(
        visual_encoder=args.visual_encoder,
        ckpt_path=args.ckpt_path,
        test_condition=args.test_condition,
        test_snr=args.test_snr,
        embedding_size=embedding_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fusion_method=args.fusion_method  
    )
