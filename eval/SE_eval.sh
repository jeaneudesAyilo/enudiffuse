#!/bin/bash

# This is to activate a python environment with conda
source ~/.bashrc

conda activate newenv

cd /group_cacul/calcul/users/jayilo/enudiffuse


CUDA_LAUNCH_BLOCKING=1
# Set variables


# DATASET="VB"
# DATASET="WSJ0"


# DATA_DIR="./eval/wsj_test.json"
# DATA_DIR="./eval/vb_dmd.json"



SEGMENT="$1"
TOTAL_SEGMENTS=$2


# CKPT_PATH="./ckpts/separate_vbdmd_speech_modeling.ckpt"
# CKPT_PATH="./ckpts/separate_wsjqut_speech_modeling.ckpt"


# SAVE_ROOT="path to a saving folder"

SAVE_ROOT="./eval/result/"

ALGO_TYPE="diffuseen" 


if test "$ALGO_TYPE" = "separate_paradiffuseen"
then

    TAG="separate_paradiffuseen"
    NUM_E=30
    NUM_EM=1
    NBATCH=4 #8
    STARTSTEP=0
    LMBD=5.75     
    # CKPT_NOISE_PATH="./ckpts/separate_vbdmd_noise_modeling.ckpt"
    # CKPT_NOISE_PATH="./ckpts/separate_wsjqut_noise_modeling.ckpt"


elif test "$ALGO_TYPE" = "joint_paradiffuseen"  # "joint_paradiffuseen_pure_oracle"
then

    TAG="joint_paradiffuseen" 
    NUM_E=30
    NUM_EM=1
    NBATCH=4 #8
    STARTSTEP=0
    LMBD=5.75     
    CKPT_NOISE_PATH=""


elif test "$ALGO_TYPE" = "separate_paradiffusein"
then

    TAG="separate_paradiffusein" 
    NUM_E=30
    NUM_EM=1
    NBATCH=4 #8
    STARTSTEP=0
    LMBD=1.     
    # CKPT_NOISE_PATH="" ## not needed
    # CKPT_NOISE_PATH="" ## not needed


elif test "$ALGO_TYPE" = "joint_paradiffusein"
then

    TAG="joint_paradiffusein" 
    NUM_E=30
    NUM_EM=1
    NBATCH=4 #8
    STARTSTEP=0
    LMBD=1.     
    # CKPT_NOISE_PATH="" ## not needed
    # CKPT_NOISE_PATH="" ## not needed


elif test "$ALGO_TYPE" = "diffuseen"    #"diffuseen_pure_oracle"     # "diffuseen_oracle"
then

    TAG="diffuseen_bs4"     
    NUM_E=30
    NUM_EM=1
    NBATCH=4
    STARTSTEP=0
    LMBD=1.75
    CKPT_NOISE_PATH=""
   

elif test "$ALGO_TYPE" = "depse_il" 
then

    TAG="depse_il"     
    NUM_E=30
    NUM_EM=1
    NBATCH=4
    STARTSTEP=0
    LMBD=0
    CKPT_NOISE_PATH=""


elif test "$ALGO_TYPE" = "depse_tl"   
then

    TAG="depse_tl"     
    NUM_E=30
    NUM_EM=1
    NBATCH=4
    STARTSTEP=0
    LMBD=0
    CKPT_NOISE_PATH=""


elif test "$ALGO_TYPE" = "fudiffse"   #"fudiffse_oracle"  # "fudiffse_pure_oracle"
then

    TAG="fudiffuse_bs4"     
    NUM_E=30
    NUM_EM=1
    NBATCH=4
    STARTSTEP=0
    LMBD=1.5
    CKPT_NOISE_PATH=""


elif test "$ALGO_TYPE" = "udiffse"  #"udiffse_oracle"  # "udiffse_pure_oracle"
then

    TAG="udiffuse"    
    NUM_E=30
    NUM_EM=5
    NBATCH=4
    STARTSTEP=0
    LMBD=1.5
    CKPT_NOISE_PATH=""



elif test "$ALGO_TYPE" = "separate_paradepse"
then

    TAG="paradepse" 
    NUM_E=30
    NUM_EM=1
    NBATCH=4 #8
    STARTSTEP=0
    LMBD=0.     #lambda is useless here
    # CKPT_NOISE_PATH="./ckpts/separate_vbdmd_noise_modeling.ckpt"
    # CKPT_NOISE_PATH="./ckpts/separate_wsjqut_noise_modeling.ckpt"


elif test "$ALGO_TYPE" = "tl_diffuseen"     
then

    TAG="tl_diffuseen"     
    NUM_E=30
    NUM_EM=1
    NBATCH=4
    STARTSTEP=0
    LMBD=0 #lambda is useless here
    CKPT_NOISE_PATH=""


elif test "$ALGO_TYPE" = "il_diffuseen"    
then

    TAG="il_diffuseen"     
    NUM_E=30
    NUM_EM=1
    NBATCH=4
    STARTSTEP=0
    LMBD=0 #lambda is useless here
    CKPT_NOISE_PATH=""


else 
    echo "NOT AVAILABLE ALGO"
    break

fi

optimized_lambda="false" 

# Run command

python eval/evaluation.py \
    --dataset "$DATASET" \
    --segment "$SEGMENT" \
    --num_segments "$TOTAL_SEGMENTS" \
    --ckpt_path "$CKPT_PATH" \
    --algo_type "$ALGO_TYPE" \
    --tag "$TAG" \
    --data_dir "$DATA_DIR" \
    --save_root "$SAVE_ROOT" \
    --num_E "$NUM_E" \
    --num_EM "$NUM_EM" \
    --nbatch "$NBATCH" \
    --startstep "$STARTSTEP" \
    --lambda "$LMBD" \
    --ckpt_noise_path "$CKPT_NOISE_PATH"
    