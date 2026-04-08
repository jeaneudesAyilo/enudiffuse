#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import os
import pandas as pd
import sys
import shutil
from glob import glob
from tqdm import tqdm
sys.path.append(".")
import json
import torch
from src import InferenceAlgoRegistry
import argparse
from datetime import datetime
import time
import soundfile as sf
from six.moves import cPickle as pickle 
from torchaudio import load
import subprocess
import numpy as np
from itertools import permutations

def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def save_rtf(save_dir, speaker_id, file_name, noise_type, snr, save_name, rtf):        
    with open(os.path.join(save_dir, "rtf.csv"), "a") as text_file:
        text_file.write(f"{speaker_id},{file_name},{noise_type},{snr},{save_name},{rtf}\n")



device = "cuda" if torch.cuda.is_available() else "cpu"


def speech_enhance(params):
    if params["dataset"] in ["WSJ0","VB"]:
        # Load file json
        with open(params["data_dir"], "r") as f:
            dataset = json.load(f)

    elif params["dataset"] in ["TCD-TIMIT", "TCD-DEMAND", "LRS3-DEMAND", "LRS3-NTCD","EARS-TAU","LIBRI-FSD50K"]:
        dataset = load_dict(params["data_dir"])

            

    seg_id = params["segment"]
    ind_start = 0
    ind_end = len(dataset)

    if seg_id >= 0:
        num_files = len(dataset)
        segment_size = num_files // params["num_segments"]
        ind_start, ind_end = seg_id * segment_size, (seg_id + 1) * segment_size
        if seg_id == params["num_segments"] - 1:
            ind_end = num_files
        print(f"Evaluating files at [{ind_start},{ind_end}]")

    # Skip files that have already been processed
    files_processed = [
        x[:-4] for x in os.listdir(params["save_dir"]) if x.endswith(".wav")
    ]

    # Init evaluation
    print(f"\nTotal number of files to evaluate: {ind_end - ind_start}\n")

    compute_rtf = True
    if not os.path.isfile(os.path.join(params["save_dir"], "rtf.csv")) and compute_rtf: ## then, write the head of the file
        save_rtf(params["save_dir"], 'speaker_id', 'file_name', 'noise_type', 'snr', 'enhanced', 'rtf')     



    verbose = False
    listen_noise = False

    algo_cls = InferenceAlgoRegistry.get_by_name(params["algo_type"])


    if "para" in params["algo_type"]:
        
        verbose = False
        listen_noise = False        

    enhance = algo_cls(ckpt_path=params["ckpt_path"], 
                       num_E=params["num_E"], 
                       verbose=verbose, 
                    #    optimized_lambda=params["optimized_lambda"],                      
                    #    print_metrics=False,
                    #    listen_noise=listen_noise, #with verbose=True, ## if listen_noise is enabled and in the run the clean file path is not none then, we record noise 
                       ckpt_noise=params["ckpt_noise_path"],
                       )


    if "para" in params["algo_type"]:
        if enhance.verbose:
            ##folder to store estimated clean speech, estimated noise, re-estimated clean speech after postprocess
            params["save_dir_speech"] = os.path.join(params["save_dir"],"speech")
            params["save_dir_noise"] = os.path.join(params["save_dir"],"noise")
            params["save_dir_respeech"] = os.path.join(params["save_dir"],"respeech")

            for folder in [params["save_dir_speech"],params["save_dir_noise"],params["save_dir_respeech"] ]:
                os.makedirs(folder, exist_ok=True)
            
            files_processed = [
                x[:-4] for x in os.listdir(params["save_dir_speech"]) if x.endswith(".wav")
            ] 


    if params["dataset"] in ["WSJ0","VB"]:

        for ind_i, (ind_mix, mix_info) in (enumerate(dataset.items())):
            if (
                ind_start <= ind_i < ind_end
                and mix_info["utt_name"] not in files_processed
            ):
                utt_name = mix_info["utt_name"]
                mix_file = mix_info["noisy_wav"].format(noisy_root=params["noisy_root"])
                clean_file = mix_info["clean_wav"].format(
                    clean_root=params["clean_root"]
                )

                                                
                if enhance.verbose : 
                    start_time = time.time() #datetime.now()
                    s_hat, _, n_hat, _,s_rehat, _ = enhance.run(
                        mix_file=mix_file,
                        clean_file = None if "oracle" not in params["algo_type"] else clean_file,
                        video_file =  '',
                        nmf_rank=params["nmf_rank"],
                        num_EM=params["num_EM"],
                        lmbd=params["lambda"],
                        nbatch=params["nbatch"],
                        startstep=params["startstep"],
                        # divide_s0hat = params["divide_s0hat"]                        
                    )
                    end_time = time.time() #datetime.now()
                    duration = end_time-start_time                       

                    # duration = duration.total_seconds()    
                    if compute_rtf:
                        mix,sr = load(mix_file)  
                        assert sr==sr
                        rtf = (duration)/(mix.shape[1]/sr)  
                        ## write the rtf
                        save_rtf(params["save_dir"], mix_info['p_id'], mix_info['utt_name'], mix_info['noise_type'], mix_info['snr'], utt_name, rtf)

                    sf.write(os.path.join(params["save_dir_speech"], utt_name + ".wav"), s_hat, 16000)
                    sf.write(os.path.join(params["save_dir_noise"], utt_name + ".wav"), n_hat, 16000)
                    sf.write(os.path.join(params["save_dir_respeech"], utt_name + ".wav"), s_rehat, 16000)


                else :
                    recon_file = os.path.join(params["save_dir"], utt_name + ".wav") 
                    start_time = time.time() # datetime.now()                   
                    s_hat, _ = enhance.run(
                        mix_file=mix_file,
                        clean_file = None if "oracle" not in params["algo_type"] else clean_file,
                        nmf_rank=params["nmf_rank"],
                        num_EM=params["num_EM"],
                        lmbd=params["lambda"],
                        nbatch=params["nbatch"],
                        startstep=params["startstep"],
                        # divide_s0hat = params["divide_s0hat"],
                    )

                    end_time = time.time() #datetime.now()
                    duration = end_time-start_time                    

                    # duration = duration.total_seconds()    
                    if compute_rtf:
                        mix,sr = load(mix_file)  
                        assert sr==sr
                        rtf = (duration)/(mix.shape[1]/sr)                            
                        save_rtf(params["save_dir"], mix_info['p_id'], mix_info['utt_name'], mix_info['noise_type'], mix_info['snr'], utt_name, rtf)

                    sf.write(recon_file, s_hat, 16000)

        
        saving = params["save_dir_speech"] if enhance.verbose else params["save_dir"]

        if len([x[:-4] for x in os.listdir(saving) if x.endswith(".wav")]) == len(dataset):

            if params["dataset"] in ["WSJ0"]:               
                output = subprocess.Popen(f'python eval/statistics/compute_metrics.py  --enhanced_dir {saving}  --data_dir {params["data_dir"]} --save_dir {saving}  --dataset  {params["dataset"]} --dnn_mos', shell=True)
                stdout, stderr = output.communicate() 

            else:
                output = subprocess.Popen(f'python eval/statistics/compute_metrics.py  --enhanced_dir {saving}  --data_dir {params["data_dir"]} --save_dir {saving}  --dataset  {params["dataset"]}', shell=True)
                stdout, stderr = output.communicate() 

                output2 = subprocess.Popen(f'python eval/statistics/compute_metrics_add_vb_dnnmos.py  --enhanced_dir {saving}  --data_dir {params["data_dir"]} --save_dir {saving}  --dataset  {params["dataset"]} --dnn_mos', shell=True)
                stdout2, stderr2 = output2.communicate() 


    elif params["dataset"] in ["TCD-TIMIT","TCD-DEMAND", "LRS3-DEMAND", "LRS3-NTCD", "EARS-TAU","LIBRI-FSD50K"]:


        for ind_i, mix_info in enumerate(dataset):
            if (
                ind_start <= ind_i < ind_end
                and f"{mix_info['speaker_id']}_{mix_info['noise_type']}_{mix_info['snr']}_{mix_info['file_name']}"
                not in files_processed
            ):
                mix_file = mix_info["mix_file"]
                clean_file = mix_info["speech_file"]
                save_name = f"{mix_info['speaker_id']}_{mix_info['noise_type']}_{mix_info['snr']}_{mix_info['file_name']}"
                

                ##collect the suitable video file
                if not enhance.audio_only :  
                    #video_file = mix_info["video_file"] 

                    if enhance.vfeat_processing_order in ["cut_extract"] :
                    
                        if params["dataset"] in ["TCD-TIMIT","TCD-DEMAND"]:

                            if enhance.video_feature_type in  ["resnet", "avhubert", "raw_image"]: 
                                video_path  = "/group_storage/audio_visual/CROPPED_MOUTH_ldmark_48_68_size_88_88/TCD-TIMIT/test/{speaker_id}/straightcam/{filename}_mouthcrop.mp4"                
                            
                            elif enhance.video_feature_type in  ["flow_avse"]: 
                                video_path  = "/group_storage/audio_visual/CROPPED_MOUTH_ldmark_28_68_size_112_112/TCD-TIMIT/test/{speaker_id}/straightcam/{filename}_mouthcrop.mp4"                                             


                        elif params["dataset"] in ["LRS3-DEMAND", "LRS3-NTCD"]:                            

                            if enhance.video_feature_type in  ["resnet", "avhubert", "raw_image"]: 
                                video_path  =  "/group_storage/audio_visual/CROPPED_MOUTH_ldmark_48_68_size_88_88/LRS3/test/{speaker_id}/{filename}_mouthcrop.mp4"             
                            
                            elif enhance.video_feature_type in  ["flow_avse"]: 
                                video_path  = "/group_storage/audio_visual/CROPPED_MOUTH_ldmark_28_68_size_112_112/LRS3/test/{speaker_id}/{filename}_mouthcrop.mp4"
                    
                            
                        video_file = video_path.format(speaker_id=mix_info['speaker_id'], filename=mix_info['file_name'])

                    else:
                        
                        raise NotImplementedError(f"vfeat_processing_order {enhance.vfeat_processing_order} in audio_visual case is not implemented in eval !")
                    

                else:
                    video_file = None


                if enhance.verbose : 
                    start_time = time.time() #datetime.now()
                    s_hat, _, n_hat, _,s_rehat, _ = enhance.run(
                        mix_file=mix_file,
                        clean_file = None if "oracle" not in params["algo_type"] else clean_file,
                        video_file =  video_file,
                        nmf_rank=params["nmf_rank"],
                        num_EM=params["num_EM"],
                        lmbd=params["lambda"],
                        nbatch=params["nbatch"],
                        startstep=params["startstep"],
                        # divide_s0hat = params["divide_s0hat"]                        
                    )
                    end_time = time.time() #datetime.now()

                    duration = end_time-start_time
                    # duration = duration.total_seconds()                                                                            
                    
 
                    #save_speech
                    try:
                        sf.write(os.path.join(params["save_dir_speech"], save_name + ".wav"), s_hat, 16000)
                    except:
                        reduced_name_noise_type = mix_info['noise_type'][:len(mix_info['noise_type'])//2]
                        save_name = f"{mix_info['speaker_id']}_{reduced_name_noise_type}_{mix_info['snr']}_{mix_info['file_name']}"
                        sf.write(os.path.join(params["save_dir_speech"], save_name + ".wav"), s_hat, 16000)


                    #save_noise
                    try:
                        sf.write(os.path.join(params["save_dir_noise"], save_name + ".wav"), n_hat, 16000)
                    except:
                        reduced_name_noise_type = mix_info['noise_type'][:len(mix_info['noise_type'])//2]
                        save_name = f"{mix_info['speaker_id']}_{reduced_name_noise_type}_{mix_info['snr']}_{mix_info['file_name']}"
                        sf.write(os.path.join(params["save_dir_noise"], save_name + ".wav"), n_hat, 16000)


                    #save_respeech
                    try:
                        sf.write(os.path.join(params["save_dir_respeech"], save_name + ".wav"), s_rehat, 16000)
                    except:
                        reduced_name_noise_type = mix_info['noise_type'][:len(mix_info['noise_type'])//2]
                        save_name = f"{mix_info['speaker_id']}_{reduced_name_noise_type}_{mix_info['snr']}_{mix_info['file_name']}"
                        sf.write(os.path.join(params["save_dir_respeech"], save_name + ".wav"), s_rehat, 16000)

                    saving = params["save_dir_speech"]


  
                else:
                    recon_file = os.path.join(params["save_dir"], save_name + ".wav")
                    start_time = time.time() #datetime.now()
                    s_hat, _ = enhance.run(
                        mix_file=mix_file,
                        clean_file = None if "oracle" not in params["algo_type"] else clean_file,
                        video_file = video_file,
                        nmf_rank=params["nmf_rank"],
                        num_EM=params["num_EM"],
                        lmbd=params["lambda"],
                        nbatch=params["nbatch"],
                        startstep=params["startstep"],
                                                
                    )
                    end_time = time.time() #datetime.now()

                    duration = end_time-start_time
                    # duration = duration.total_seconds()

                    try:
                        sf.write(recon_file, s_hat, 16000)
                    except:
                        reduced_name_noise_type = mix_info['noise_type'][:len(mix_info['noise_type'])//2]
                        save_name = f"{mix_info['speaker_id']}_{reduced_name_noise_type}_{mix_info['snr']}_{mix_info['file_name']}"
                        recon_file = os.path.join(params["save_dir"], save_name + ".wav")
                        sf.write(recon_file, s_hat, 16000)

                    saving = params["save_dir"]


                ##compute rtf and save it as a file
                if compute_rtf:
                    mix,sr = load(mix_file)  
                    assert sr==sr
                    rtf = (duration)/(mix.shape[1]/sr)      
                    save_rtf(params["save_dir"], mix_info['speaker_id'], mix_info['file_name'], mix_info['noise_type'], mix_info['snr'], save_name, rtf)


        saving = params["save_dir_speech"] if enhance.verbose else params["save_dir"]
        if len([x[:-4] for x in os.listdir(saving) if x.endswith(".wav")]) == len(dataset):

            if "TCD" in params["dataset"] and "LRS3" not in params["dataset"]:
                output = subprocess.Popen(f'python eval/statistics/compute_metrics.py  --enhanced_dir {saving}  --data_dir {params["data_dir"]} --save_dir {saving}  --dataset  {params["dataset"]} --dnn_mos', shell=True)

            elif "LRS3" in params["dataset"] :
                output = subprocess.Popen(f'python eval/statistics/compute_metrics.py  --enhanced_dir {saving}  --data_dir {params["data_dir"]} --save_dir {saving}  --dataset  {params["dataset"]}', shell=True)
            
            if "EARS" in params["dataset"] :
                output = subprocess.Popen(f'python eval/statistics/compute_metrics.py  --enhanced_dir {saving}  --data_dir {params["data_dir"]} --save_dir {saving}  --dataset  {params["dataset"]} --dnn_mos', shell=True)
                    
        stdout, stderr = output.communicate() 


                

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        self.parser.add_argument(
            "--segment",
            type=int,
            choices=[-1] + list(range(0, 1000)),
            default=-1,
            help="Segment ID of the test files to evaluate on",
        )
        self.parser.add_argument(
            "--num_segments",
            type=int,
            default=4,
            help="total number of segments to evaluate on",
        )
        self.parser.add_argument(
            "--nbatch",
            type=int,
            default=4,
            help="number of batches for parallel estimation",
        )
        self.parser.add_argument(
            "--lambda",
            type=float,
            default=1.5,
            help="weight parameter for posterior sampler",
        )
        self.parser.add_argument(
            "--exp_name", type=str, default="p232", help="experiment name"
        )
        self.parser.add_argument(
            "--dataset",
            type=str,
            default="WSJ0",
            choices=["TCD-TIMIT", "WSJ0","WSJ0-2mix_8k", "VB", "avsec2", "TCD-DEMAND", "LRS3-DEMAND", "LRS3-NTCD", "EARS-TAU","LIBRI-FSD50K"],
            help="dataset",
        )
        self.parser.add_argument(
            "--ckpt_path",
            type=str,
            default="./data/checkpoints/diffusion_gen_nonlinear_transform.ckpt",
            help="path to the ckpt",
        )

        self.parser.add_argument(
            "--ckpt_noise_path",
            type=str,
            default="",
            help="path to the noise ckpt, useful for ParaDiffuSE",
        )
        
        self.parser.add_argument(
            "--algo_type",
            type=str,
            choices= InferenceAlgoRegistry.get_all_names(),
            required=True,
            help="SE algorithm",
        )


        self.parser.add_argument(
            "--tag",
            type=str,
            default="orig",
            help="Tag given to the specific version of SE algorithm",
        )
        self.parser.add_argument(
            "--data_dir",
            type=str,
            default="wsj_test.json",
            help="json file for audios to be enhanced",
        )
        self.parser.add_argument(
            "--save_root", type=str, default="/tmp", help="path to denoised data"
        )
        self.parser.add_argument(
            "--log_type", type=int, default=1, choices=[1, 2], help="1 file, 2 stream"
        )
        self.parser.add_argument("--nmf_rank", type=int, default=4, help="NMF rank")
        self.parser.add_argument(
            "--num_EM", type=int, default=100, help="number of EM iterations"
        )
        self.parser.add_argument(
            "--num_E", type=int, default=1, help="number of iterations in the E-step"
        )
        self.parser.add_argument(
            "--startstep", type=int, default=0, help="start step (0 or greater)"
        )

        self.parser.add_argument(
            "--divide_s0hat", type=str, choices=("yes", "no"), default="no", help="whether to divide s0hat by gamma_t or not"
        )    

        self.parser.add_argument(
            "--set_v_to_zero", type=str, choices=("yes", "no"), default="no", help="whether to set v to 0 or not. If yes, then v=0, if no use the right value of v"
        )         

        # self.parser.add_argument(
        #     "--divide_s0hat", action="store_true", help="whether to divide s0hat by gamma_t or not"
        # )        

        self.parser.add_argument(
            "--optimized_lambda",
            action="store_true",
            help="to compute automatically the lambda or not",
        )

    def get_params(self):
        self._initial()
        self.opt = self.parser.parse_args()
        params = vars(self.opt)
        return params


if __name__ == "__main__":
    params = Options().get_params()

    if params["dataset"] == "VB":
        params["noisy_root"] = "/group_storage/source_separation/VoiceBankDEMAND/noisy_testset_wav_16k"
        params["clean_root"] = "/group_storage/source_separation/VoiceBankDEMAND/clean_testset_wav_16k"
    
    elif params["dataset"] == "WSJ0":
        params["noisy_root"] = (
            "/group_storage/source_separation/QUT_WSJ0/test"
        )
        params["clean_root"] = (
            "/group_storage/source_separation/WSJ0_SE/wsj0_si_et_05"
        )

    params["save_dir"] = str(
        os.path.join(
            params["save_root"],
            params["ckpt_path"].split("/")[-2], #os.path.basename(params["ckpt_path"])[:-5],
            params["dataset"],
            params["algo_type"],
            str(params["lambda"]),
            params["tag"],
        )
    )

    if not os.path.isdir(params["save_dir"]):
        os.makedirs(params["save_dir"], exist_ok=True)

    # save the input args
    args_file = f"{params['save_dir']}/commandline_args.txt"
    with open(args_file, "w") as f:
        json.dump(params, f, indent=2)

    source_file = f"src/{params['algo_type']}.py"
    destination_directory = params["save_dir"]
    if not os.path.exists(os.path.join(destination_directory, f"{params['algo_type']}.py")):
        shutil.copy(source_file, destination_directory)

    speech_enhance(params)
