
source ~/.bashrc
conda activate newenv
cd /group_calculus/users/jayilo/enudiffuse

python train.py \
	--transform_type exponent \
	--format tcd-timit \
	--batch_size 8 \
	--vfeat_processing_order default \
	--backbone ncsnpp6M \
	--audio_only \
	--run_id aonly_tcd_speech_modeling_default_6M