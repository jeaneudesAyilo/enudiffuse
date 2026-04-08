#!/bin/bash
# # This should not be changed
#OAR -q production
# # Remove `!` for CPUs only
#OAR -p cluster='gres'
# # Adapt as desired
#OAR -l host=1,walltime=72:00:00


source ~/.bashrc
conda activate newenv
cd /group_calculus/users/jayilo/enudiffuse

python train.py --base_dir /group_storage/source_separation/VoiceBankDEMAND/train_valid_16k \
  --format vb \
  --batch_size 8 \
  --vfeat_processing_order default \
  --backbone ncsnpp6M \
  --audio_only \
  --noise_modelling \
  --run_id my_vb_noise_modelling_default_6M