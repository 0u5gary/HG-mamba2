import os

###############################################################
################# CHANGE YOUR PATH ############################
###############################################################

# TODO: replace with your own project root and data folder paths
# Remember export PYTHONPATH='/your/path/to/this_project' before use

PROJECT_ROOT = "/your/path/to/this_project"
LRS3_FOLDER_PATH = "/your/path/to/lrs3"
DNS_FOLDER_PATH = "/your/path/to/DNS-Challenge"

##############################################################
################## OVERALL CONFIGURATION #####################
##############################################################

embedding_size_dict = {
    "TalkNet": 512,
    "AVHuBERT": 768,
    "AVHuBERT_TalkNet_concatenate": 1280, 
}

VISUAL_ENCODER = "AVHuBERT_TalkNet_concatenate"
EMBEDDING_SIZE = embedding_size_dict[VISUAL_ENCODER]

AVHUBERT_PATH = os.path.join(PROJECT_ROOT, "data/Visual_Encoder/AV_HuBERT/avhubert")
VSRIW_PATH = os.path.join(PROJECT_ROOT, "data/Visual_Encoder/VSRiW")
TALKNET_PATH = os.path.join(PROJECT_ROOT, "data/Visual_Encoder/TalkNet")

# REPLACE WITH YOUR STORED MODEL CHECKPOINT
CKPT_PATH=os.path.join(PROJECT_ROOT, "checkpoints/path/to/checkpoint.ckpt")

##############################################################
################## TESTING CONFIGURATION #####################
##############################################################

# TEST_SNR available options: 
# "10", "5", "0", "mixed"
TEST_SNR = "mixed"

# TEST_CONDITION available options:
# "one_interfering_speaker", "three_interfering_speakers", "noise_only"
TEST_CONDITION = "noise_only"

TEST_VISUAL_ENCODERS = ["TalkNet", "AVHuBERT", "AVHuBERT_TalkNet_concatenate"]
TEST_ALL_CONDITIONS = ["one_interfering_speaker", "three_interfering_speakers", "noise_only"]
TEST_ALL_SNRs = ["mixed", "0", "5", "10"]
