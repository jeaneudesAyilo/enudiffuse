#!/bin/bash

# This is to activate a python environment with conda
source ~/.bashrc

conda activate newenv

# Set variables


# DATA_DIR="./eval/wsj_test.json"
# DATA_DIR="./eval/vb_dmd.json"


# DATASET="VB"
# DATASET="WSJ0"


# ENHANCED_DIR="",





# Run command
python eval/statistics/compute_metrics.py \
    --enhanced_dir "$ENHANCED_DIR" \
    --data_dir "$DATA_DIR" \
    --save_dir "$ENHANCED_DIR" \
    --dataset  "$DATASET" \
    --dnn_mos
    
    # --input_metrics        
    # --trimming
    

### VBDMD DNSMOS
# python eval/statistics/compute_metrics_add_vb_dnnmos.py \
#     --enhanced_dir "$ENHANCED_DIR" \
#     --data_dir "$DATA_DIR" \
#     --save_dir "$ENHANCED_DIR" \
#     --dataset  "$DATASET" \
#     --dnn_mos \
#     --input_metrics        
