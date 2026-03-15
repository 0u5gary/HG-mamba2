# HG-mamba: An Efficient Heterogeneous Graph-Guided Mamba Framework for Structural Fusion in Audio-Visual Speech Enhancement

## Usage Instruction
Clone this GitHub repo

Create a virtual environment:
```bash
conda create -y -n hg-mamba python=3.10.18
conda activate hg-mamba
pip install -r requirements.txt
pip install causal_conv1d-1.5.3.post1+cu121torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-2.2.6.post3+cu121torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
**Note**
1. When running pip install -r requirements.txt, you may see dependency incompatibility warnings or errors (e.g., version conflicts).
These can be safely ignored — the installation will complete successfully.
2. We recommend downloading `causal-conv1d` and `mamba-ssm` from their original GitHub repositories and installing them via the `.whl` files.

:exclamation: FIRST, change `PROJECT_ROOT` in `config.py` before proceeding.

Then run 
`export PYTHONPATH='/your/path/to/this_project'` in your terminal and change directory to `src` folder. 

## Data Preprocessing

We use [LRS3](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/) for our training. Please download the dataset and change the `LRS3_FOLDER_PATH` in `config.py` to the folder path you saved the data to. 

We also use [DNS-Challenge](https://github.com/microsoft/DNS-Challenge) for creating our noisy input mixture (run their `download-dns-challenge-5-noise-ir.sh` script). Please download the dataset and change the `DNS_FOLDER_PATH` to the folder path you saved the DNS-Challenge data to. 

First, extract clean audio from the videos. Make sure the file paths in your data match our provided split files (`lrs3_split.csv` and `dns_noise_split.csv`).
```bash
python -W ignore data/extract_audio_from_video.py
```

### Extract Pretrained Visual Embeddings

Prior to extracting pretrained visual embeddings from the dataset, clone the corresponding GitHub repo into the `src/data/Visual_Encoder` folder and download the checkpoints of the selected pretrained model. Follow the environment setup instructions in each pretrained model's GitHub README and then run the feature extractor script in the setup environment from `src/data` folder. The features will be saved to `src/Visual_Feature/LRS3_{encoder}_Features/{split}`.

| Encoder Task | Encoder Name | GitHub Repo | Checkpoint Used | Feature Extractor Script |
|--------------|--------------|-------------|-----------------|---------------------------|
| AVSR         | [VSRiW](https://arxiv.org/pdf/2202.13084)        |[link](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages): save to `/benchmarks/GRID/models/` | `---` | `---` |
| AVSR         | AVHuBERT [Paper 1](https://arxiv.org/abs/2201.02184), [2](https://arxiv.org/pdf/2201.01763)    |[link](https://github.com/facebookresearch/av_hubert)<sup>1</sup>: save to `avhubert/conf/finetune/`              |base fine-tuned for VSR on LRS3-433h [\[src\]](https://dl.fbaipublicfiles.com/avhubert/model/lrs3/vsr/base_lrs3_433h.pt)<sup>2</sup> |`src/data/avhubert_extract_visual_features.py` |
| ASD          | [TalkNet](https://arxiv.org/pdf/2107.06592) |[link](https://github.com/TaoRuijie/TalkNet-ASD): save to repo root folder |  [\[src\]](https://drive.google.com/file/d/1NVIkksrD3zbxbDuDbPc_846bLfPSZcZm/view)   | `src/data/TalkNet_extract_visual_features.py`|

**Note**  
During extracing feature by AVHuBERT, **a small number of videos (about 10)** may fail due to face detection issues, very short clips, or low resolution. This is **normal** and expected on LRS3. The script will automatically skip them and continue. You can check the log file (`failed_avhubert_feat_{self.split}.txt`) to see which files were skipped.

<sup>1</sup> We have provided a patch with several fixes (see `src/data/Visual_Encoder/fix.patch`).

<sup>2</sup> Go to the official model checkpoint [page](https://facebookresearch.github.io/av_hubert/) and sign the license agreement first. 

### Simulate the Noisy Input Mixture

Run the below command to create the noisy input mixture, assuming you are at `src` level. The mixture will be saved to `src/LRS3_Mixed/{split}`.
```
python -W ignore utils/mix_speech_gpu.py
```

## Training
Run the below terminal command to start training the model. By default, logs and checkpoints will be saved to the CHECKPOINT_DIR defined in config.py. You can override parameters like visual encoder, batch size, and checkpoint directory using command-line arguments.

```
python -W ignore train.py
```
To resume training from a saved checkpoint, add the --train_from_checkpoint flag and specify the path using --ckpt_path:

```
python -W ignore train.py --train_from_checkpoint --ckpt_path=checkpoints/epoch-last.ckpt
```

## Evaluation

First generate test input mixtures of different conditions and SNR scenarios, run 
```
python -W ignore data/generate_test_data.py --condition=noise_only --snr=mixed
```
If you want to generate different conditions and SNRs at once, you could use comma to separate them. For example: `--condition="noise_only, one_interfering_speaker, three_interfering_speakers" --snr="mixed, 0, 5, 10"`

The mixture will be saved to `src/LRS3_Mixed/test/{condition}/{snr}/`. 


To evaluate a single visual encoder under a specific test condition and SNR, use test.py, which accepts command-line arguments for full flexibility:

```
python -W ignore test.py --visual_encoder=AVHuBERT_TalkNet_concatenate --ckpt_path path/to/checkpoint --test_condition=noise_only --test_snr=mixed --fusion_method gcn_mamba
```

To evaluate a single visual encoder under all test conditions and SNRs, use run_test_all.py.
```
python -W ignore run_test_all.py
``` 

## Acknowledgement

Our code is based in part on [RAVEN](https://github.com/Bose/RAVEN). Thanks for their great work.