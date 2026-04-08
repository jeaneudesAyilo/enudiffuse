#!/bin/bash
# # This should not be changed
#OAR -q production
# # Remove `!` for CPUs only
#OAR -p cluster='gres'
# # Adapt as desired
#OAR -l host=1,walltime=96:00:00


source ~/.bashrc
conda activate newenv
cd /group_calculus/users/jayilo/enudiffuse

python train.py \
  --transform_type exponent \
  --format wsj0 \
  --batch_size 8 \
  --vfeat_processing_order default \
  --video_feature_type resnet \
  --backbone jointncsnpp6M \
  --audio_only \
  --joint_noise_clean_speech_training \
  --run_id wsj_speech_noise_modelling_default_6M
