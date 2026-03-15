import argparse
import os
import warnings
import multiprocessing
import sys
import copy
import torch
from src.models.system import System
from src.models.fusion_lstm import Fusion
from src.models.fusion_mamba import FusionMamba
from src.models.fusion_gcn_seq import FusionGCN_Sequential
from src.data.datamodule import LRS3DataModule
from src.losses.complex_mse import PSA_MSE
from src.utils.flop_counter import count_model_flops

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.audio import SignalNoiseRatio, PerceptualEvaluationSpeechQuality, ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility

import config

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

def train(args, train_from_checkpoint=True):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    datamodule = LRS3DataModule(
        data_path=config.LRS3_FOLDER_PATH,
        visual_encoder=args.visual_encoder,
        embedding_size=args.embedding_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    datamodule.setup()
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    logger = TensorBoardLogger(save_dir=args.logger_save_dir, name=args.logger_name, version=args.version_name)

    every_epoch_checkpoint = ModelCheckpoint(
        dirpath=args.checkpoint_dir, filename="epoch-{epoch:02d}-{step}",
        save_top_k=-1, every_n_epochs=1, save_last=True
    )

    best_checkpoint = ModelCheckpoint(
        dirpath=args.checkpoint_dir, filename="best-{epoch:02d}-{step}",
        monitor="val/loss", mode="min", save_top_k=1, save_last=True
    )

    # --- Model Selection Logic ---
    if args.fusion_method == 'lstm':
        print("Initializing Standard LSTM Fusion Model...")
        fusion_network = Fusion(embedding_size=args.embedding_size)
    elif args.fusion_method == 'gcn':
        print("Initializing Graph Fusion (GCN) Model...")
        fusion_network = FusionGCN_Sequential(embedding_size=args.embedding_size, hidden_dim=400, sequential_type=None)
    elif args.fusion_method == 'gcn_mamba':
        print("Initializing Graph Fusion (GCN mamba) Model...")
        fusion_network = FusionGCN_Sequential(embedding_size=args.embedding_size, hidden_dim=400, sequential_type='mamba')
    elif args.fusion_method == 'gcn_lstm':
        print("Initializing Graph Fusion (GCN lstm) Model...")
        fusion_network = FusionGCN_Sequential(embedding_size=args.embedding_size, hidden_dim=400, sequential_type='lstm')
    elif args.fusion_method == 'mamba':
        # [Ablation] Pure Mamba
        print("Initializing Baseline: FusionMamba...")
        fusion_network = FusionMamba(embedding_size=args.embedding_size, hidden_dim=400)
    elif args.fusion_method == 'mamba_film':
        # [Ablation] film Mamba
        print("Initializing Baseline: FusionMamba_film...")
        fusion_network = FusionMamba(embedding_size=args.embedding_size, hidden_dim=400, fusion_type='film')
    elif args.fusion_method == 'mamba_linear':
        # [Ablation] linear Mamba
        print("Initializing Baseline: FusionMamba_linear_attention...")
        fusion_network = FusionMamba(embedding_size=args.embedding_size, hidden_dim=400, fusion_type='linear_attn')
    elif args.fusion_method == 'mamba_cross':
        # [Ablation] cross Mamba
        print("Initializing Baseline: FusionMamba_cross_attention...")
        fusion_network = FusionMamba(embedding_size=args.embedding_size, hidden_dim=400, fusion_type='cross_attn')
    else:
        raise ValueError(f"Unknown fusion method: {args.fusion_method}")

    # ================= Compute FLOPs =================
    fusion_network_for_flops = copy.deepcopy(fusion_network)

    flops, params = count_model_flops(
        fusion_network_for_flops, 
        embedding_size=args.embedding_size, 
        seq_len=500,    
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    del fusion_network_for_flops  # release memory
    # =========================================================

    loss = PSA_MSE()
    metrics = {
        'snr': SignalNoiseRatio(),
        'pesq': PerceptualEvaluationSpeechQuality(16000, 'wb'),
        'sisdr': ScaleInvariantSignalDistortionRatio(),
        'estoi': ShortTimeObjectiveIntelligibility(16000, True)
    }

    # Pass fusion_method to System so it knows how to handle steps
    model = System(fusion_network, loss, metrics, fusion_method=args.fusion_method, encoder_name=args.visual_encoder)

    trainer = Trainer(
        max_epochs=args.max_epochs, accelerator='gpu', logger=logger,
        callbacks=[every_epoch_checkpoint, best_checkpoint],
        enable_progress_bar=True, val_check_interval=1.0,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=8,
        log_every_n_steps=1
    )

    if train_from_checkpoint:
        print(f"Resuming from checkpoint: {args.ckpt_path}")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.ckpt_path)
    else:
        print("Training from scratch...")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser(description="Train the AV speech model(LRS3 Version).")

    # Existing args...
    parser.add_argument("--visual_encoder", type=str, default=config.VISUAL_ENCODER, choices=config.embedding_size_dict.keys())
    parser.add_argument("--embedding_size", type=int)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--ckpt_path", type=str, default=config.CKPT_PATH)
    parser.add_argument("--version_name", type=str)
    parser.add_argument("--logger_save_dir", type=str, default="lightning_logs")
    parser.add_argument("--logger_name", type=str, default="pretrained_encoders")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--train_from_checkpoint", action="store_true")
    parser.add_argument("--fusion_method", type=str, default="gcn_mamba", 
                    choices=["lstm", "gcn", "gcn_mamba", "gcn_lstm", "mamba", "mamba_film", "mamba_linear", "mamba_cross"])

    args = parser.parse_args()
    
    # Auto-generate version name if not provided
    if args.version_name is None:
        args.version_name = f"{args.visual_encoder}_{args.fusion_method}"

    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(config.PROJECT_ROOT, f"checkpoints/{args.version_name}_LRS3")
    if args.embedding_size is None:
        args.embedding_size = config.embedding_size_dict[args.visual_encoder]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()

    os.environ["GLOG_minloglevel"] = "3"
    multiprocessing.set_start_method('spawn', True)

    train(args, train_from_checkpoint=args.train_from_checkpoint)