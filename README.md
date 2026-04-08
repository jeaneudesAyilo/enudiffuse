# Diffusion-based Frameworks for Unsupervised Speech Enhancement

This repository contains the official Pytorch implementation of the papers: 
- [Diffusion-based Frameworks for Unsupervised Speech Enhancement](https://arxiv.org/abs/2601.09931) and 
- [Posterior Transition Modeling for Unsupervised Diffusion-Based Speech Enhancement](https://ieeexplore.ieee.org/document/11053679/). 



## Environment setup
After cloning this repository, setup the environment by executing: 
```bash
cd enudiffuse
conda env create -f environment.yml
conda activate newenv
```
or 

```bash
cd enudiffuse
chmod +x alternative_install.sh
alternative_install.sh
```
Please, note that the installation could take some time. 

## Training
Available training recipes include: training a separate audio-only diffusion model for speech data, noise data; training jointly on speech data and noise data (audio-only); and training an audio-visual diffusion model for speech data.

- Training the speech diffusion model we used for WSJ0 dataset:

```bash
python train.py \
  --transform_type exponent \
  --format wsj0 \
  --batch_size 8 \
  --vfeat_processing_order default \
  --backbone ncsnpp6M \
  --audio_only \
  --run_id name_of_your_model
```

- Training the noise diffusion model we used for QUT dataset: 

```bash
python train.py \
  --transform_type exponent \
  --format wsj0 \
  --batch_size 8 \
  --vfeat_processing_order default \
  --backbone ncsnpp6M \
  --audio_only \
  --noise_modelling \
  --run_id name_of_your_model
```

- Training the joint diffusion model we used for speech and noise regarding the WSJ0-QUT dataset:

```bash
python train.py \
  --transform_type exponent \
  --format wsj0 \
  --batch_size 8 \
  --vfeat_processing_order default \
  --video_feature_type resnet \
  --backbone jointncsnpp6M \
  --audio_only \
  --joint_noise_clean_speech_training \
  --run_id name_of_your_model
```

Following the [fast_UdiffSE](https://github.com/jeaneudesAyilo/fast_UdiffSE/) repository, audio-visual model can be trained as follows: 

```bash
python train.py \
	--transform_type exponent \
	--format tcd-timit \
	--batch_size 8 \
	--vfeat_processing_order cut_extract \
	--video_feature_type avhubert \
	--backbone ncsnpp_continueconcat_attn_masking_noising_av_6m \
	--fusion concat_attn_masking_light \
	--no_project_video_feature \
	--p 0.0 \
	--fusion_level enc_dec \
	--run_id name_of_your_model
```


## Pretrained checkpoints
 
We provided the following checkpoints used in our work:

- [Audio-only diffusion model trained separately on WSJ0 speech dataset (~5.2 M parameters)](https://huggingface.co/jeaneudesAyilo/enudiffuse/blob/main/separate_wsjqut_speech_modeling.ckpt)

- [Audio-only diffusion model trained separately on QUT noise dataset (~5.2 M parameters)](https://huggingface.co/jeaneudesAyilo/enudiffuse/blob/main/separate_wsjqut_noise_modeling.ckpt)

- [Audio-only diffusion model trained jointly on WSJ0 speech and QUT noise datasets (~5.9 M parameters)](https://huggingface.co/jeaneudesAyilo/enudiffuse/blob/main/joint_wsjqut_speech_noise_modeling.ckpt)

- [Audio-only diffusion model trained separately on VoiceBank speech dataset (~5.2 M parameters).](https://huggingface.co/jeaneudesAyilo/enudiffuse/blob/main/separate_vbdmd_speech_modeling.ckpt)

- [separate_vbdmd_noise_modeling.ckpt: Audio-only diffusion model trained separately on DEMAND noise dataset (~5.2 M parameters)](https://huggingface.co/jeaneudesAyilo/enudiffuse/blob/main/separate_vbdmd_noise_modeling.ckpt)

- [Audio-only diffusion model trained jointly on VoiceBank speech and QUT noise datasets (~5.9 M parameters)](https://huggingface.co/jeaneudesAyilo/enudiffuse/blob/main/joint_vbdmd_speech_noise_modeling.ckpt)


## Evaluation
Inference algorithms include: UDiffSE, UDiffSE+ (ie fUDiffSE), DEPSE-IL, DEPSE-TL, DiffUSEEN, ParaDiffUSE-IN (separate and joint) and  ParaDiffUSE-EN (separate and joint).

- Run speech enhancement on WSJ0-QUT or VoiceBank-DEMAND test sets. 

After providing the correct information in [./eval/SE_eval.sh](./eval/SE_eval.sh), run :

```bash
cd ./eval
bash launch_SE_ALL.sh
```

- Compute metrics (optionally)
(being in the parent folder `enudiffuse`), run:
```bash
bash ./eval/statistics/run_metrics.sh
```

*Note that you would need to insert the parts to your data in some of the files involved in the evaluation and metrics computation.*


## TODO

- [ ] Reduce the redundancies in the algorithms, by constructing a base inference algorithm class that all the algorithms would inherite.
<!--- - [ ]  --->

<!---## Demo

A demo of the AV-UDiffSE framework is provided in [./demo_av.ipynb](./demo_av.ipynb) . This notebook provides a demonstration of sampling from clean speech prior learned via a diffusion-based generative model conditionned on lip video, followed by speech enhancement of a test noisy speech signal. --->

<!--- ## Supplementary material

Supplementary material, including additional details, discussions and parameter studies that serve to expand our work is provided in the `docs` directory ([direct link](./docs/UDiffSE_Supplementary.pdf)).--->

<!---
```--->

## Citations 

We would appreciate it if you could cite our papers in your publication when using any part of our research or code.

```bibtex
@article{ayilo2026diffusion,
  title={Diffusion-based Frameworks for Unsupervised Speech Enhancement},
  author={Ayilo, Jean-Eudes and Sadeghi, Mostafa and Serizel, Romain and Alameda-Pineda, Xavier},
  journal={arXiv preprint arXiv:2601.09931},
  year={2026}
}

@article{sadeghi2025posterior,
  title={Posterior transition modeling for unsupervised diffusion-based speech enhancement},
  author={Sadeghi, Mostafa and Ayilo, Jean-Eudes and Serizel, Romain and Alameda-Pineda, Xavier},
  journal={IEEE Signal Processing Letters},
  year={2025},
  publisher={IEEE}
}


```
