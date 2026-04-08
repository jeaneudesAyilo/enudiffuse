#!/bin/bash

set -e  # stop if error

conda create -n newenv python=3.8 -y
source activate newenv

conda install -y conda-forge::pytorch-lightning==1.9.0

python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

python -m pip install \
h5py==3.6.0 \
humanfriendly==10.0 \
hydra-core==1.3.2 \
omegaconf==2.3.0 \
librosa \
Ninja \
torch-ema \
pesq \
pystoi \
tensorboard \
tqdm \
opencv-python \
einops \
python_speech_features \
sentencepiece \
pandas \
wandb==0.12.11 \
protobuf==3.20.1 \
pyloudnorm \
onnxruntime \
matplotlib \
seaborn \
ipykernel

pip install https://github.com/vBaiCai/python-pesq/archive/master.zip