#!/bin/bash

source ~/.bashrc
conda activate newenv
cd /group_calculus/users/jayilo/enudiffuse

python train.py \
  --transform_type exponent \
  --format wsj0 \
  --batch_size 8 \
  --vfeat_processing_order default \
  --backbone ncsnpp6M \
  --audio_only \
  --noise_modelling \
  --run_id noise_modelling_default_6M
#   --resume_from_checkpoint 