#!/bin/bash

source ~/.bashrc

conda activate newenv


# ENHANCED_DIR=""
# ENHANCED_DIR=""


# DATA_LIST="/group_calculus/users/jayilo/proper_diffuse_noise/eval/wsj_test.json"
DATA_LIST="/group_calculus/users/jayilo/proper_diffuse_noise/eval/vb_dmd.json"


# DATASET="WSJ0"
DATASET="VB"



echo "$ENHANCED_DIR"
echo "$DATASET"
echo "$DATA_LIST"



python ./eval/statistics/DNS-Challenge/DNSMOS/dnsmos_local_general.py \
    --testset_dir "$ENHANCED_DIR" \
    --data_list "$DATA_LIST" \
    --dataset  "$DATASET"
# --unprocessed_metrics