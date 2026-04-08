#!/bin/bash

# This is to activate a python environment with conda
source ~/.bashrc

conda activate newenv

# Set variables


# ENHANCED_DIR=""


# DATA_DIR="./eval/wsj_test.json"
# DATA_DIR="./eval/vb_dmd.json"


# DATASET="VB"
# DATASET="WSJ0"

# Run command
python eval/statistics/compute_metrics.py \
    --enhanced_dir "$ENHANCED_DIR" \
    --data_dir "$DATA_DIR" \
    --save_dir "$ENHANCED_DIR" \
    --dataset  "$DATASET" \
    --dnn_mos \
    --noise
    
    # 
    # --input_metrics