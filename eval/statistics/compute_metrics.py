"""
Adapted from original code by Clarity Challenge
https://github.com/claritychallenge/clarity
"""

import os
from tqdm import tqdm
import pandas as pd
from soundfile import SoundFile
from librosa.effects import trim
from concurrent.futures import ProcessPoolExecutor
from argparse import ArgumentParser
import numpy as np
from six.moves import cPickle as pickle  # for performance
import json
import sys

sys.path.append(".")
from src.eval_metrics import compute_dnnmos, compute_pesq, compute_sisdr, compute_stoi, energy_ratios
import warnings

warnings.filterwarnings("ignore")


def save_dict(di_, filename_):
    with open(filename_, "wb") as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di


def read_audio(filename):
    """Read a wavefile and return as numpy array of floats.
    Args:
        filename (string): Name of file to read
    Returns:
        tuple: Tuple containing audio signal and a boolean indicating success
    """
    success = True  # Assume success by default

    try:
        wave_file = SoundFile(filename)
    except:
        success = False  # An error occurred

    if success:
        audio_signal = wave_file.read()
        return audio_signal
    else:
        return []


def run_metrics(input_file, save_dir, noise= False, trimming=False, dnn_mos=False, input_metrics=False, v2s=False):
    fs = 16000
    
    if input_metrics and not v2s:
        enh_file = input_file["noisy"]

    if not input_metrics and v2s:
        enh_file = input_file["v2s"]

    if not input_metrics and not v2s:
        enh_file = input_file["enhanced"]

    if input_metrics and v2s:
        raise Exception("Sorry, input_metrics and v2s can not be True simultaneously")         
        

    tgt_file = input_file["clean"]    
        

    metrics_file = os.path.join(
        save_dir,
        f"{input_file['speaker_id']}_{input_file['noise_type']}_{input_file['snr']}_{input_file['file_name']}.pkl",
    )

    # Skip processing with files dont exist or metrics have already been computed
    if (
        (not os.path.isfile(enh_file))
        or (not os.path.isfile(tgt_file))
        or (os.path.isfile(metrics_file))
    ):
        return

    # Read enhanced signal
    enh = read_audio(enh_file)

    if not noise:
        # Read clean/target signal
        clean = read_audio(tgt_file)
        n = read_audio(input_file["noisy"]) - clean

        if trimming :
            
            clean_trimmed, index = trim(clean,  top_db=30)
            if len(clean_trimmed)/16000 >= 3.2:
                clean = clean_trimmed
                enh = enh[index[0]:index[1]]
                n =  n[index[0]:index[1]]
        

    else:
        clean = read_audio(input_file["noisy"]) - read_audio(tgt_file) #groundtruth noise, it is considered as the "clean speech " when computing metrics for noise
        n = read_audio(tgt_file) # clean speech, it is consisered as the noise when computing metrics for noise
        
        if trimming :
            n_trimmed, index = trim(n,  top_db=30)
            if len(n_trimmed)/16000 >= 3.2:
                n = n_trimmed
                clean = clean[index[0]:index[1]]
                enh = enh[index[0]:index[1]]


    ##note : for v2s, there may not be a correct matching in the length
    if len(enh) != 0: 
        len_x = np.min([len(enh), len(clean)])
        clean = clean[:len_x]
        enh = enh[:len_x]

        # almost useless as we make clean and enh to have same length
        # Check that both files are the same length, otherwise computing the metrics results in an error
        if len(clean) != len(enh):
            raise Exception(
                f"Wav files {enh_file} and {tgt_file} should have the same length"
            )

        # Compute metrics
        m_stoi = compute_stoi(clean, enh, fs)
        m_pesq = compute_pesq(clean, enh, fs)
        m_sisdr = compute_sisdr(clean, enh)

        assert len(clean) == len(enh) == len(n)        
        m_sisir, m_sisar = energy_ratios(enh, clean, n)[1], energy_ratios(enh, clean, n)[2]


        if dnn_mos:
            m_dnnmos = compute_dnnmos(enh, fs)
            di_ = {
                "id_file": f'{input_file["speaker_id"]}_{input_file["file_name"]}',
                "speaker_id":input_file["speaker_id"],
                "File name": input_file["file_name"],
                "Noise Type": input_file["noise_type"],
                "Noise SNR": input_file["snr"],
                "SI-SDR": m_sisdr,
                "STOI": m_stoi,
                "PESQ": m_pesq,
                "SI-SIR": m_sisir,
                "SI-SAR": m_sisar,
                "MOS_SIG": m_dnnmos["sig_mos"],
                "MOS_BAK": m_dnnmos["bak_mos"],
                "MOS_OVR": m_dnnmos["ovr_mos"],
            }
        else:
            di_ = {
                "id_file": f'{input_file["speaker_id"]}_{input_file["file_name"]}',
                "speaker_id":input_file["speaker_id"],                
                "File name": input_file["file_name"],
                "Noise Type": input_file["noise_type"],
                "Noise SNR": input_file["snr"],
                "SI-SDR": m_sisdr,
                "STOI": m_stoi,
                "PESQ": m_pesq,
                "SI-SIR": m_sisir,
                "SI-SAR": m_sisar,                
            }
        save_dict(di_, metrics_file)


def compute_metrics(input_params, save_dir, noise, trimming, dnn_mos, input_metrics, v2s):

    futures = []
    ncores = 20
    with ProcessPoolExecutor(max_workers=ncores) as executor:
        for param_ in input_params:
            futures.append(
                executor.submit(run_metrics, param_, save_dir, noise, trimming, dnn_mos, input_metrics, v2s)
            )
        proc_list = [future.result() for future in tqdm(futures)]

    if dnn_mos:
        df_metrics = pd.DataFrame(       
            columns=["id_file", "speaker_id", "File name", "Noise Type", "Noise SNR", "PESQ", "STOI", "SI-SDR","SI-SIR","SI-SAR","MOS_SIG","MOS_BAK","MOS_OVRL"]
        )
    else:
        df_metrics = pd.DataFrame(       
            columns=["id_file", "speaker_id", "File name", "Noise Type", "Noise SNR", "PESQ", "STOI", "SI-SDR","SI-SIR","SI-SAR"]
        ) 

    pkl_files = [f for f in os.listdir(save_dir) if f.endswith(".pkl")]

    # Store results in one file
    for this_file in tqdm(pkl_files):
        this_file_path = os.path.join(save_dir, this_file)
        this_res = load_dict(this_file_path)

        if dnn_mos:
            df_metrics = pd.concat(
                [
                    df_metrics,
                    pd.DataFrame.from_dict(
                        {
                            "id_file": [this_res["id_file"]],
                            "speaker_id":[this_res["speaker_id"]],                            
                            "File name": [this_res["File name"]],
                            "Noise Type": [this_res["Noise Type"]],
                            "Noise SNR": [this_res["Noise SNR"]],
                            "PESQ": [this_res["PESQ"]],
                            "STOI": [this_res["STOI"]],
                            "SI-SDR": [this_res["SI-SDR"]],
                            "SI-SIR": [this_res["SI-SIR"]],
                            "SI-SAR": [this_res["SI-SAR"]],                            
                            "MOS_SIG": [this_res["MOS_SIG"]],
                            "MOS_BAK": [this_res["MOS_BAK"]],
                            "MOS_OVRL": [this_res["MOS_OVR"]],
                        }
                    ),
                ],
                ignore_index=True,
            )
        else:
            df_metrics = pd.concat(
                [
                    df_metrics,
                    pd.DataFrame.from_dict(
                        {
                            "id_file": [this_res["id_file"]],
                            "speaker_id":[this_res["speaker_id"]],                             
                            "File name": [this_res["File name"]],
                            "Noise Type": [this_res["Noise Type"]],
                            "Noise SNR": [this_res["Noise SNR"]],
                            "PESQ": [this_res["PESQ"]],
                            "STOI": [this_res["STOI"]],
                            "SI-SDR": [this_res["SI-SDR"]],
                            "SI-SIR": [this_res["SI-SIR"]],
                            "SI-SAR": [this_res["SI-SAR"]],                             
                        }
                    ),
                ],
                ignore_index=True,
            )

        # remove tmp file
        os.system(f"rm {this_file_path}")

    # Save the DataFrame to a CSV file
    if input_metrics:
        if not trimming:
            df_metrics.to_csv(os.path.join(save_dir, "input_metrics.csv"), index=False)
        else:
            df_metrics.to_csv(os.path.join(save_dir, "input_metrics_trimmed.csv"), index=False)


    else:
        if not trimming:
            df_metrics.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)
        else:
            df_metrics.to_csv(os.path.join(save_dir, "metrics_trimmed.csv"), index=False)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run performance evaluation metrics on the enhanced signals."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory to the test data."
    )
    parser.add_argument(
        "--dnn_mos", action="store_true", help="Whether to compute DNNMOS or not."
    )
    parser.add_argument(
        "--noise", action="store_true", help="Whether the computation of the metrics is for noise or not."
    )    

    parser.add_argument(
        "--trimming",
        action="store_true",
        help="Whether to cut off silent part in the clean speech before computing the metric, or not.",
    )

    parser.add_argument(
        "--input_metrics",
        action="store_true",
        help="Whether to compute input (mixture) metrics or not.",
    )

    parser.add_argument(
        "--v2s",
        action="store_true",
        help="Whether to compute lip to speech metrics or not.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="TCD-TIMIT",
        help="Directory to the test data.",
    )
    parser.add_argument(
        "--enhanced_dir",
        type=str,
        required=True,
        help="Directory to the enhanced test data.",
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save the results."
    )
    args = parser.parse_args()

    assert not (args.input_metrics==True and args.v2s==True)

    if args.dataset in ["TCD-TIMIT","TCD-DEMAND","TCD-QUT", "LRS3-DEMAND", "LRS3-NTCD", "EARS-TAU", "LIBRI-FSD50K"]:
        # Load file list and select the target segment to process
        files_list = load_dict(args.data_dir)
        input_params = [
            {
                "noisy": filename["mix_file"],
                "clean": filename["speech_file"],
                "file_name": filename["file_name"],
                "noise_type": filename["noise_type"],
                "snr": filename["snr"],
                "speaker_id": filename["speaker_id"],
                "enhanced": f"{args.enhanced_dir}/{filename['speaker_id']}_{filename['noise_type']}_{filename['snr']}_{filename['file_name']}.wav",  # VAE-SE
                # "enhanced": f"{args.enhanced_dir}/{filename['file_name']}_{filename['speaker_id']}_{filename['noise_type']}_{filename['snr']}.wav",  # SGMSE

            }
            for filename in files_list
        ]
    elif args.dataset in [ "WSJ0","WSJ0-DEMAND_-5_0_5", "VB"]:        

        if args.dataset =="WSJ0": ##WSJ0QUT
            clean_root = "/group_storage/source_separation/WSJ0_SE/wsj0_si_et_05"
            noisy_root = "/group_storage/source_separation/QUT_WSJ0/test"

        elif args.dataset == "WSJ0-DEMAND_-5_0_5":
            clean_root = "/group_storage/source_separation/WSJ0_SE/wsj0_si_et_05"
            noisy_root = "/group_storage/source_separation/NEW_MIX/WSJ0_DEMAND/test_-5_0_5"

        elif args.dataset == "VB":
            clean_root = "/group_storage/source_separation/VoiceBankDEMAND/clean_testset_wav_16k"
            noisy_root = "/group_storage/source_separation/VoiceBankDEMAND/noisy_testset_wav_16k"

        # Load file json
        with open(args.data_dir, "r") as f:
            dataset = json.load(f)
        input_params = [
            {
                "noisy": filename["noisy_wav"].format(noisy_root=noisy_root),
                "clean": filename["clean_wav"].format(clean_root=clean_root),
                "file_name": filename["utt_name"],
                "noise_type": filename["noise_type"],
                "snr": filename["snr"],
                "speaker_id": filename["p_id"],
                "enhanced": f"{args.enhanced_dir}/{filename['utt_name']}.wav",  # VAE-SE
                # "enhanced": f"{args.enhanced_dir}/{filename['utt_name']}_{filename['noise_type']}_{filename['snr']}.wav",  # SGMSE
                
            }
            for (_, filename) in dataset.items()
        ]
            

    elif args.dataset in ["AVSEC2"]:
        # Load file list and select the target segment to process
        files_list = load_dict(args.data_dir)
        input_params = [
            {
                "noisy": filename["mix_file"],
                "clean": filename["speech_file"],
                "file_name": filename["id_scene"],
                "noise_type": "unknown",
                "snr": "unknown",
                "speaker_id": filename["id_scene"],
                "enhanced": f"{args.enhanced_dir}/{filename['id_scene']}.wav",  # VAE-SE

            }
            for filename in files_list
        ]

    elif args.dataset in ["LRS3-TAU"]:
        # Load file list and select the target segment to process
        files_list = load_dict(args.data_dir)
        input_params = [
            {
                "noisy": filename["mix_file"],
                "clean": filename["speech_file"],
                "file_name": filename["file_name"],
                "noise_type": filename["noise_type"],
                "snr": filename["snr"],
                "speaker_id": filename["speaker_id"],
                "enhanced": f"{args.enhanced_dir}/{filename['speaker_id']}_{filename['file_name']}.wav",
                "v2s": filename["v2s_file"],
                
            }
            for filename in files_list
        ]        

    
    else:
        raise ValueError("Invalid dataset.")

    compute_metrics(input_params, args.save_dir, args.noise, args.trimming, args.dnn_mos, args.input_metrics, args.v2s)
