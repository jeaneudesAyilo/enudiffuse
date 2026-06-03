# Usage:

"""

python ./dnsmos_local_general.py \
    --testset_dir "$ENHANCED_DIR" \
    --data_list "$DATA_LIST" \
    --dataset  "$DATASET"

"""
#

import argparse
import concurrent.futures
import glob
import os
import json
import librosa
import numpy as np
import numpy.polynomial.polynomial as poly
import onnxruntime as ort
import pandas as pd
import soundfile as sf
from requests import session
from tqdm import tqdm
from six.moves import cPickle as pickle 

# from metrics_utils import compute_pesq, compute_sisdr, compute_stoi, energy_ratios

import sys
sys.path.append(".")
from src.eval_metrics import compute_pesq, compute_sisdr, compute_stoi, energy_ratios

def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di




SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

class ComputeScore:
    def __init__(self, primary_model_path, p808_model_path) -> None:
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)
        
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, param, sampling_rate, is_personalized_MOS, unprocessed_metrics):
        
        fs = sampling_rate

        if unprocessed_metrics:
            enh_file = param["noisy"]

        else:
            enh_file = param["enhanced"]

        tgt_file = param["clean"]


        audio, enhanced_fs = sf.read(enh_file)
        clean, clean_fs = sf.read(tgt_file)

        # Read noisy signal and extract noise
        try:
            noisy, noisy_fs = sf.read(param["noisy"])
            n = noisy - clean
        except:
            print(f'problem with file {param["noisy"]}')  

        assert noisy_fs == enhanced_fs == clean_fs == fs  


        #### Do this only for DVAE
        if len(audio) != 0:
            len_x = np.min([len(audio), len(clean)])
            clean = clean[:len_x]
            audio = audio[:len_x]
            n = n[:len_x] 


        if len(clean) != len(audio):
            raise Exception(
                f"Wav files {enh_file} and {tgt_file} should have the same length"
            )        

        # Compute metrics
        m_stoi = compute_stoi(clean, audio, fs)
        m_pesq = compute_pesq(clean, audio, fs)
        m_sisdr = compute_sisdr(clean, audio)

        comp = energy_ratios(audio, clean, n)
        m_sisir, m_sisar = comp[1], comp[2]


        ###DNSMOS Computation
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH*fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        
        num_hops = int(np.floor(len(audio)/fs) - INPUT_LENGTH)+1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw,mos_bak_raw,mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw,is_personalized_MOS)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)


        clip_dict = {'filename': param["enhanced"], 'len_in_sec': actual_audio_len/fs, 'sr':fs}
        clip_dict['num_hops'] = num_hops
        clip_dict['OVRL_raw'] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict['SIG_raw'] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict['BAK_raw'] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict['OVRL'] = np.mean(predicted_mos_ovr_seg)
        clip_dict['SIG'] = np.mean(predicted_mos_sig_seg)
        clip_dict['BAK'] = np.mean(predicted_mos_bak_seg)
        clip_dict['P808_MOS'] = np.mean(predicted_p808_mos)
        clip_dict["SI-SDR"] = m_sisdr
        clip_dict["STOI"] = m_stoi
        clip_dict["PESQ"] = m_pesq
        clip_dict["SI-SIR"] = m_sisir
        clip_dict['SI-SAR'] = m_sisar
        clip_dict["id_file"]= f'{param["speaker_id"]}_{param["file_name"]}'
        clip_dict["speaker_id"]=param["speaker_id"]
        clip_dict["File name"]= param["file_name"]
        clip_dict["Noise Type"]= param["noise_type"]
        clip_dict["Noise SNR"] = param["snr"]

        return clip_dict

def main(args):
    models = glob.glob(os.path.join(args.testset_dir, "*"))
    audio_clips_list = []
    # p808_model_path = os.path.join('DNSMOS', 'model_v8.onnx')
    p808_model_path = os.path.join('./eval/statistics/DNS-Challenge/DNSMOS/DNSMOS', 'model_v8.onnx')

    if args.personalized_MOS:
        # primary_model_path = os.path.join('pDNSMOS', 'sig_bak_ovr.onnx')
        primary_model_path = os.path.join('./eval/statistics/DNS-Challenge/DNSMOS/pDNSMOS', 'sig_bak_ovr.onnx')

    else:
        # primary_model_path = os.path.join('DNSMOS', 'sig_bak_ovr.onnx')
        primary_model_path = os.path.join('./eval/statistics/DNS-Challenge/DNSMOS/DNSMOS', 'sig_bak_ovr.onnx')

    compute_score = ComputeScore(primary_model_path, p808_model_path)

    is_personalized_eval = args.personalized_MOS
    unprocessed = args.unprocessed_metrics
    desired_fs = SAMPLING_RATE

    rows = []


    if args.dataset == "WSJ0": #"WSJ0-QUT": 
        clean_root = "/group_storage/source_separation/WSJ0_SE/wsj0_si_et_05"
        noisy_root = "/group_storage/source_separation/QUT_WSJ0/test"


    elif args.dataset == "VB": #"VB-DMD":
        clean_root = "/group_storage/source_separation/VoiceBankDEMAND/clean_testset_wav_16k"
        noisy_root = "/group_storage/source_separation/VoiceBankDEMAND/noisy_testset_wav_16k"

    # Load file json
    with open(args.data_list, "r") as f:
        dataset = json.load(f)
    input_params = [
        {
            "noisy": filename["noisy_wav"].format(noisy_root=noisy_root),
            "clean": filename["clean_wav"].format(clean_root=clean_root),
            "file_name": filename["utt_name"],
            "noise_type": filename["noise_type"],
            "snr": filename["snr"],
            "speaker_id": filename["p_id"],
            "enhanced": f"{args.testset_dir}/{filename['utt_name']}.wav",
            
        }
        for (_, filename) in dataset.items()
    ]


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
                "enhanced": f"{args.testset_dir}/{filename['speaker_id']}_{filename['noise_type']}_{filename['snr']}_{filename['file_name']}.wav",
            }
            for filename in files_list
        ]


    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_url = {executor.submit(compute_score, input_param, desired_fs, is_personalized_eval, unprocessed): input_param for input_param in input_params}
        for future in tqdm(concurrent.futures.as_completed(future_to_url)):
            input_param = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (input_param, exc))
            else:
                rows.append(data)            

    df = pd.DataFrame(rows)

    if args.csv_path==None: ##automatic naming
        if args.unprocessed_metrics:
            csv_path = os.path.join(args.testset_dir,"_UNPROCESSED_GENERAL_METRICS.csv")

        else: 
            csv_path = os.path.join(args.testset_dir,"_GENERAL_METRICS.csv")

    else: ##use the explicitly specified path
        csv_path = args.csv_path
        
    df.to_csv(csv_path)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--testset_dir", default='.', 
                        help='Path to the dir containing audio clips in .wav to be evaluated')
    parser.add_argument('-o', "--csv_path", default=None, help='Dir to the csv that saves the results')
    parser.add_argument('-p', "--personalized_MOS", action='store_true', 
                        help='Flag to indicate if personalized MOS score is needed or regular')
    
    parser.add_argument(
        "--unprocessed_metrics",
        action="store_true",
        help="Whether to compute input (mixture) metrics or not.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the test data set",
    )

    parser.add_argument(
        "--data_list", type=str, required=True, help="List of clean speech and noisy speech files with their characteristics (noise type, snr,...)"
    )

    args = parser.parse_args()

    main(args)
