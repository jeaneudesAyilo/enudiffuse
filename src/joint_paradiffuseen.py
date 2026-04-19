#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from tqdm import tqdm
from src.utils import LinearScheduler, calc_metrics
from sgmse.sdes import OUVESDE
from sgmse.model import ScoreModel
from torchaudio import load
from sgmse.util.other import pad_spec
# from sgmse.util.utils_video import load_array,resample_video, prep_video, videocap

from . import InferenceAlgoRegistry

@InferenceAlgoRegistry.register("joint_paradiffuseen")
class JointParaDiffUSEEN:
    def __init__(
        self,
        ckpt_path= "", #ckpt_speech="",
        ckpt_noise="", ## ckpt_noise will always be empty here
        num_E=30,
        transform_type="exponent",
        delta=1e-10,
        eps=0.03,
        snr=0.5,
        sr=16000,
        verbose=False,
        listen_noise = False,
        device= "cuda",
        print_metrics = False,
        set_v_to_zero = "no",
        optimized_lambda= False,
        
    ):
        """
        Parallel Diffusion-based models for "Unsupervised" Speech Enhancement (ParaDiffUSE). Algorithm using two parallel diffusion
        models, one for modeling cleen speech, the other for modelling noise. NMF is no longer used at inference.

        Args:
            ckpt_path: Path to the pre-trained diffusion model.
            num_E: Number of iterations for the E step (reverse diffusion process).
            verbose: Whether to print progress information.
            ll_approximation: Approximation used to compute the log-likelyhood: uninformative_prior or posterior_mean (aka tweedie or dps)
        """

        self.snr = snr
        self.sr = sr
        self.delta = delta
        self.num_E = num_E
        

        self.verbose = verbose
        self.listen_noise = listen_noise
        self.device = device

        self.scheduler = LinearScheduler(N=num_E, eps=eps)
        
        # ==== For joint prior speech noise model ====        
        self.sde = OUVESDE(theta=1.5, sigma_min=0.05, sigma_max=0.5, N=num_E)

        ### if using a checkpoint trained with code like the one present in this repo and which implements the backbones jointncsnpp; then use the following lines 
       
        self.model = ScoreModel.load_from_checkpoint(
            ckpt_path, base_dir="", batch_size=1, num_workers=0, kwargs=dict(gpu=False)
        )
        self.model.data_module.transform_type = transform_type
        self.model.eval(no_ema=False)
        self.model.to(self.device)

        #for gres cluster
        self.model_cpu = ScoreModel.load_from_checkpoint(
            ckpt_path, base_dir="", batch_size=1, num_workers=0, kwargs=dict(gpu=False)
        )
        self.model_cpu.data_module.transform_type = transform_type


        ### if using a checkpoint trained with a code different than the one present in this repo and which implements 
        ### the backbones jointncsnpp... but don't name it as that (neural network archicture is however the same); then use the following lines. 
		### if using the demo notebook with the provided checkpoints or a checkpoint trained with backbone jointncsnpp-like present in this repo,
		### please do not uncomment these following lines, it is already taken into account in the notebook.
		### uncomment for example when using the provided checkpoint to run the inference on the full test set. 


		###start

        # self.model = ScoreModel.load_from_checkpoint(ckpt_path,joint_noise_clean_speech_training=True,                                                            
        # base_dir="", batch_size=1, num_workers=0, kwargs=dict(gpu=False), backbone="jointncsnpp6M")
        # self.model.data_module.transform_type = "exponent"
        # self.model.eval(no_ema=False)
        # self.model.to("cuda")
        # self.model_cpu = ScoreModel.load_from_checkpoint(ckpt_path,joint_noise_clean_speech_training=True,
        # base_dir="", batch_size=1, num_workers=0, kwargs=dict(gpu=False), backbone="jointncsnpp6M") ## not necessarly required


        # self.model = ScoreModel.load_from_checkpoint(ckpt_path, joint_noise_clean_speech_training=True,
        # base_dir="", batch_size=1, num_workers=0, kwargs=dict(gpu=False), backbone="jointncsnpp6M")
        # self.model.data_module.transform_type = "exponent"
        # self.model.eval(no_ema=False)
        # self.model.to("cuda")
        # self.model_cpu = ScoreModel.load_from_checkpoint(ckpt_path, joint_noise_clean_speech_training=True,
        # base_dir="", batch_size=1, num_workers=0, kwargs=dict(gpu=False), backbone="jointncsnpp6M") ## not necessarly required
		
        ###end


        self.audio_only = self.model.audio_only
        if not self.audio_only : 
            from sgmse.util.utils_video import load_array,resample_video, prep_video, videocap

            self.fps =30
            self.video_feature_type = self.model.dnn.video_feature_type
            self.vfeat_processing_order = self.model.dnn.vfeat_processing_order
            self.set_v_to_zero = set_v_to_zero
        else: 
            self.vfeat_processing_order = "default"  
        

        self.print_metrics = print_metrics


    def pick_zeta_schedule(self, schedule, t, zeta, linear_t=None, clip=50_000, max_step=0.9, decay_rate=1.0, increase_rate=1.0):
        if schedule == "none":
            return None
        if schedule == "constant":
            zeta_t = zeta
        if schedule == "lin-decrease":
            zeta_t = zeta * t
        if schedule == "lin-increase":
            zeta_t = zeta * (1 - t)
        if schedule == "half-cycle":
            zeta_t = zeta * np.sin(np.pi * t)
        if schedule == "sqrt-increase":
            zeta_t = zeta * np.sqrt(1e-10 + t)
        if schedule == "exp-increase":
            zeta_t = zeta * np.exp(t)
        if schedule == "exp-decrease":
            zeta_t = zeta * (np.exp(increase_rate * linear_t / max_step) - 1) / (np.exp(increase_rate) - 1)
        if schedule == "log-increase":
            zeta_t = zeta * np.log(1 + 1e-10 + t)
        if schedule == "div-sig":
            zeta_t = zeta / t
        if schedule == "div-sig-square":
            zeta_t = zeta / t**2
        if schedule == "sigma":
            zeta_t = zeta * self.sde._std(t)
        if schedule == "sigma_like":
            zeta_min = 1e-5
            zeta_max = 0.5
            logsig = np.log(zeta_max / zeta_min)
            theta = 1.5
            zeta_t = zeta * torch.sqrt(
                        (
                            zeta_min**2
                            * torch.exp(-2 * theta * t)
                            * (torch.exp(2 * (theta + logsig) * t) - 1)
                            * logsig
                        )
                        /
                        (theta + logsig)
                    )
        if schedule == "saw-tooth-increase":
            if linear_t < max_step:  # ramp from 0 to zeta0 in rho_max
                zeta_t = zeta / max_step * linear_t
            else:
                zeta_t = zeta + zeta * (max_step - linear_t) / (1 - max_step)

        if schedule == "saw-tooth-exp":
            if linear_t < max_step:  # exponential increase to zeta at max_step
                # Normalize the exponential function to ensure peak at zeta
                zeta_t = zeta * (np.exp(increase_rate * linear_t / max_step) - 1) / (np.exp(increase_rate) - 1)
            else:
                # Exponential decrease with decay_rate
                zeta_t = zeta * np.exp(-decay_rate * (linear_t - max_step) / (1 - max_step))
        return min(zeta_t, clip)




    def load_visual_data(self, vfile_path):   

        if self.vfeat_processing_order in ["cut_extract"]:            
            video_size_dict = {"avhubert":88,"resnet":88, "raw_image":88, "flow_avse":112}

            v = prep_video(video_path=vfile_path, 
                    start_frame=0, video_size=video_size_dict[self.video_feature_type],
                    video_feature_type=self.video_feature_type)     

            v = v.to(self.device) 

            if self.video_feature_type in ["resnet", "avhubert"]: 
                nb_v_frame = v.shape[1]

            elif self.video_feature_type in ["flow_avse"]: 
                nb_v_frame= v.shape[0]

            elif self.video_feature_type in ["flow_avse"]: 
                nb_v_frame= v.shape[1]

        return v, nb_v_frame
    


    def load_data(self, file_path, add_noise=False, vfile_path = None):
        """
        Load speech data and compute spectrogram.
        """
        x, sr = load(file_path)
        if add_noise:
            x += 1e-4*torch.randn_like(x)
        assert sr == self.sr
        self.T_orig = x.size(1)

        # X = pad_spec(
        #     torch.unsqueeze(self.model._forward_transform(self.model._stft(x)), 0)
        # ).to(self.device)  

        #for gres cluster
        X = pad_spec(
            torch.unsqueeze(self.model_cpu._forward_transform(self.model_cpu._stft(x)), 0)
        ).to(self.device)  

              

        ##processing video
        if not self.audio_only:
            assert vfile_path is not None
            # if vfile_path is None : vfile_path = file_path.replace(".wav","Raw.npy")            v,_ = self.load_visual_data(vfile_path)  


            if self.vfeat_processing_order in ["cut_extract"]:
                v,_=self.load_visual_data(vfile_path)                   
        else:
            v = None          

        # print(f"######### X.shape {X.shape} #######")
        # print(f"######### v.shape {v.shape} #######")
        return x, X, v 


    def to_audio(self, specto):
        return self.model.to_audio(specto.squeeze(), self.T_orig).cpu().reshape(1, -1)

    # def to_audio_tr(self, specto):
    #     return self.model._istft(specto, self.T_orig).cpu().reshape(1, -1)

    def to_audio_tr(self, specto): #for gres cluster
        specto = specto.cpu()
        return self.model_cpu._istft(specto, self.T_orig).cpu().reshape(1, -1)    
    

    # def enforce_orthogonality(self, s, n):
    # def refinement_step(self, S0, endtstep, noise=False, method="tweedie"):


    def predictor_corrector(self, St, label, t, laststep, dt,): #self, St, label, v, t, laststep, dt, noise=False

        # Corrector  
        with torch.no_grad():            
            score = self.model.forward(St, t, label) #score_model.forward(St, t, v)
            std = self.sde.marginal_prob(St, t)[1]
            step_size = (self.snr * std) ** 2
            z = torch.randn_like(St)
            St = (
                St
                + step_size[:, None, None, None] * score
                + torch.sqrt(step_size * 2)[:, None, None, None] * z
            )

        # Predictor
        with torch.no_grad():
            f, g = self.sde.sde(St, t)
            score = self.model.forward(St, t, label)  #score_model.forward(St, t, v)
            z = (
                torch.zeros_like(St) if laststep else torch.randn_like(St)
            )  # if not laststep else torch.zeros_like(St)
            St = (
                St
                - f * dt
                + (g**2)[:, None, None, None] * score * dt
                + g[:, None, None, None] * torch.sqrt(dt) * z
            )
            torch.cuda.empty_cache()

        return St, std, score, g

    def likelihood_update_individual(self, St, N0, t, std, std_noise, dt, lmbd, w_up=True):
        """
        Pseudo-likelihood update. Similar to udiffse
        """

        with torch.no_grad():
            # clean speech / noise
            theta = self.sde.theta
            mu_t = torch.exp(-theta * t)[:, None, None, None]
            _, g = self.sde.sde(St, t)          

            difference = self.X - (St / mu_t + N0 ) 
            w = 1e-3
            nppls = (
                (1 / mu_t)
                * difference
                / ((std[:, None, None, None] / mu_t) ** 2 + w) 
                # / ((std[:, None, None, None] / mu_t) ** 2 + (std[:, None, None, None]) ** 2 + 5e-5) #1e-3)
            ).type(torch.complex64)

            weight = lmbd * (g**2)[:, None, None, None]
            St = St + weight * nppls * dt

            return St 

    def prior_sampler(self, clean_file = None, vfile_path = None, noise=False):
        """
        Prior sampling algorithm to (un)conditionally or conditionally generate a clean speech or noise signal.
        """
       
        if not noise : 
            label = torch.tensor(np.array([1], dtype=np.int64),device=self.device)        
        else : 
            label = torch.tensor(np.array([0], dtype=np.int64),device=self.device)
      
        timesteps = self.scheduler.timesteps()
        self.NF = 1

        window_length = self.model.data_module.n_fft
        freq_bins_stft = 1 + window_length//2 ##256


        if (self.audio_only==True and noise==False) or (noise==True): #unconditional generation of an audio of 5s
            ##default settings
            self.T_orig = 80000
            nb_stft_frame = 640
            v = None
        else :
            ##to generate a speech consistent with the duration of the video;but for the denoising we'll use the nb_stft_frame of noisy spec
            assert vfile_path is not None , print("Provide vfile_path")
            assert clean_file is not None , print("Provide clean_file for reference purpose")            

            audio, spec, v = self.load_data(file_path=clean_file,vfile_path = vfile_path)
            
            v = v.unsqueeze(dim=0) #(1,1,T,H,W) or #(1,nbframe,embsize,) 
            self.T_orig = audio.size(1)  #but in fact this is already done in the line above with :self.T_orig = x.size(1)
            nb_stft_frame = spec.shape[-1]


        # Set the very first sample at t=1
        St = torch.randn(
            1, 1, freq_bins_stft, nb_stft_frame, dtype=torch.cfloat, device=self.device
        ) * self.sde._std(torch.ones(1, device=self.device))
        

        # Discretised time-step
        dt = torch.tensor(1 / self.num_E, device=self.device)

        # Sampling iterations
        for i in tqdm(range(0, self.num_E)):
            t = torch.tensor([timesteps[i]], device=self.device)
            St, _, _, _ = self.predictor_corrector(
                St=St,                
                t=t,
                label = label,
                #v=v,
                laststep=i == (self.num_E - 1),
                dt=dt,                
            )

        st = self.to_audio(St)
        St = self.model._backward_transform(St)

                
        return st, St
        

    def posterior_sampler(self, startstep=0, S0=None, N0=None):  
        """
        Posterior sampler algorithm that functions as the E-step for the EM process of UDiffSE.
        """
        timesteps = self.scheduler.timesteps()

        t = torch.tensor([timesteps[startstep]], device=self.device).repeat(self.nbatch)
        S_mean, _ = self.sde.marginal_prob(self.X, t)
        # S_mean = self.X

        if S0 is None:
            # Set the very first sample at t=1
            St = (
                torch.randn_like(self.X) * self.sde._std(timesteps[startstep]*torch.ones(1, device=self.device))
                + S_mean
            )
        else:
            # Set the very first sample at t=1
            St = (
                torch.randn_like(self.X) * self.sde._std(timesteps[startstep]*torch.ones(1, device=self.device))
                + S0
            )          

        N_mean, _ = self.sde.marginal_prob(St - self.X, t)
        # N_mean = St - self.X

        if N0 is None:
            Nt = (
                torch.randn_like(self.X) * self.sde._std(timesteps[startstep]*torch.ones(1, device=self.device))
                + N_mean
            )
        else:
            Nt = (
                torch.randn_like(self.X) * self.sde._std(timesteps[startstep]*torch.ones(1, device=self.device))
                + N0
            )

        # Discretised time-step
        dt = torch.tensor(1 / self.num_E, device=self.device)

        if self.verbose:
            range_i = tqdm(range(startstep, self.num_E))
        else:
            range_i = range(startstep, self.num_E)

        if S0 is not None:
            S0hat = S0
            N0hat = N0
        else:
            S0hat = St
            N0hat = Nt        


        for i in range_i:
            # Predictor-Corrector iteration
            t = torch.tensor([timesteps[i]], device=self.device).repeat(self.nbatch)
            
            St_old = St.clone()
            Nt_old = Nt.clone()

            # update St: p(St)
            St, std, S_score, _ = self.predictor_corrector(
                St=St,                                
                t=t,
                label = self.label_speech,
                # v=self.visual_feature,
                laststep=i == (self.num_E - 1),
                dt=dt,                
            )

            # update Nt: p(Nt)
            Nt, std_noise, N_score, _ = self.predictor_corrector(
                St=Nt,                 
                t=t,
                label = self.label_noise,
                # v=None,
                laststep=i == (self.num_E - 1),
                dt=dt,                
            )


            lmbd = self.pick_zeta_schedule(
                schedule="sigma",
                t=torch.tensor([timesteps[i]], device=self.device),
                zeta=self.lmbd,
                linear_t=(self.num_E - i) / self.num_E,
                max_step=0.99,
                decay_rate=1.0, 
                increase_rate=1.0,
            )

            # Likelihood term & parameter update
            if i % self.project_every_k_steps == 0 and i < self.num_E - 1:

                # N0hat
                theta = self.sde.theta
                gamma_t = torch.exp(-theta * t)[:, None, None, None]
                N0hat = (
                    Nt + torch.tensor(std**2)[:, None, None, None] * N_score
                ) / gamma_t

                # S0hat
                theta = self.sde.theta
                gamma_t = torch.exp(-theta * t)[:, None, None, None]
                S0hat = (
                    St + torch.tensor(std**2)[:, None, None, None] * S_score
                ) / gamma_t

                # P(X| St, N0hat)
                St = self.likelihood_update_individual(
                    St=St,
                    N0=N0hat,
                    t=t,
                    std=std,
                    std_noise=std,
                    dt=dt,
                    lmbd=lmbd,
                )

                # P(X| Nt, S0hat)
                Nt = self.likelihood_update_individual(
                    St=Nt,
                    N0=S0hat,
                    t=t,
                    std=std,
                    std_noise=std,
                    dt=dt,
                    lmbd=lmbd,
                )


        return St, Nt


    def run(
        self,
        mix_file,
        clean_file=None,  
        video_file = None,   
        lmbd=5.75,
        nbatch=8,
        num_EM=1,
        nmf_rank=4,
        project_every_k_steps=1,
        std_measurement = 0.15,       
        startstep=0,
        wiener_filter=True,
        mixture_consistency=False,
        refine=False,
        S0=None,
        N0=None,        
    ):
        self.lmbd = lmbd
        self.project_every_k_steps = project_every_k_steps
        self.nbatch = nbatch
        self.std_measurement = std_measurement
        self.wiener_filter = wiener_filter

 
        x, X, v = self.load_data(file_path = mix_file,  add_noise=True, vfile_path = video_file)
        self.x = x
        self.NF = X.abs().max()
        X = X / self.NF


        if self.verbose and clean_file != None:
            s_ref, S_ref, _  = self.load_data(file_path=clean_file, add_noise=False, vfile_path = video_file)
            self.s_ref = s_ref
            self.S_ref = S_ref
            s_ref = s_ref.numpy().reshape(-1)
                     

            x_withoutgaussian_noise, _, _  = self.load_data(file_path = mix_file, add_noise=False, vfile_path = video_file)
            x_withoutgaussian_noise = x_withoutgaussian_noise.numpy().reshape(-1)
            
            if self.print_metrics:
                #calc_metrics(ref,hat,n)
                
                metrix = calc_metrics(s_ref, x_withoutgaussian_noise, x_withoutgaussian_noise - s_ref)
                
                print(
                    f"Input PESQ: {metrix['pesq']:.4f} --- SI-SDR: {metrix['si_sdr']:.4f} --- ESTOI: {metrix['estoi']:.4f} --- SI-SIR: {metrix['si_sir']:.4f} --- SI-SAR: {metrix['si_sar']:.4f}",                    
                )


                metrix_noise = calc_metrics(x_withoutgaussian_noise - s_ref, x_withoutgaussian_noise, s_ref)
                print(
                    f"Input Noise PESQ: {metrix_noise['pesq']:.4f} --- SI-SDR: {metrix_noise['si_sdr']:.4f} --- ESTOI: {metrix_noise['estoi']:.4f} --- SI-SIR: {metrix_noise['si_sir']:.4f} --- SI-SAR: {metrix_noise['si_sar']:.4f}",                  
                )      
                      
                print()


        if S0 is None:
            self.X = X.repeat(self.nbatch, 1, 1, 1)
        else:
            self.X = S0
 
        self.label_speech = torch.tensor(np.array([1]*self.nbatch, dtype=np.int64),device=self.device)
        self.label_noise = torch.tensor(np.array([0]*self.nbatch, dtype=np.int64),device=self.device)

        
        if not self.audio_only:        
            if self.vfeat_processing_order in ["cut_extract"]:
                if self.video_feature_type in  ["resnet", "avhubert"]: 
                    self.visual_feature = v.repeat(self.nbatch, 1, 1, 1, 1) #(b,1,nb_frame,h,w)
                
                elif self.video_feature_type in  ["flow_avse"]: 
                    self.visual_feature = v.repeat(self.nbatch, 1, 1, 1) #(b,nb_frame,h,w)

                elif self.video_feature_type in  ["raw_image"]:              
                    self.visual_feature = v.repeat(self.nbatch, 1, 1) #(b,h*w,nb_frame)
            
        else : self.visual_feature = None

        # metrix = {"pesq": 0.0, "si_sdr": 0.0, "estoi": 0.0, "si_sir":0.0, "si_sar":0.0}
        # metrix_noise = {"pesq": 0.0, "si_sdr": 0.0, "estoi": 0.0, "si_sir":0.0, "si_sar":0.0}

        St, Nt = self.posterior_sampler(startstep=startstep, S0=S0, N0=N0 )


        S0, N0 = St.clone(), Nt.clone()
        self.S0, self.N0 = S0, N0



        # First, inverse transform, then average
        # self.NF = 1
        St = self.model._backward_transform(St)
        St = St * self.NF


        if mixture_consistency:
            St_hat = St.clone()
            St_hat = St_hat.mean(0) 
            Nt_hat = self.model._backward_transform(Nt)
            Nt_hat = Nt_hat.mean(0) 
            Nt_hat = Nt_hat * self.NF
            X_hat = St_hat + Nt_hat
            X = self.model._backward_transform(X).squeeze()
            X_true = X * self.NF
            St = St_hat + 0.5*(X_true - X_hat)
            Nt = Nt_hat + 0.5*(X_true - X_hat)

        if self.wiener_filter:
            X = X * self.NF
            St_abs_2 = (S0.abs().pow(2)/(S0.abs().pow(2) + N0.abs().pow(2))).mean(0) * X.abs().pow(2)
            St = St_abs_2.sqrt() * torch.exp(1j * torch.angle(S0.mean(0)))
            Nt_abs_2 = (N0.abs().pow(2)/(S0.abs().pow(2) + N0.abs().pow(2))).mean(0) * X.abs().pow(2)
            Nt = Nt_abs_2.sqrt() * torch.exp(1j * torch.angle(N0.mean(0)))
            St = self.model._backward_transform(St).squeeze()
            Nt,Nt_postprocess = self.model._backward_transform(Nt).squeeze(), self.model._backward_transform(Nt).squeeze()
            X = self.model._backward_transform(X).squeeze()

            self.St, self.Nt, self.Xt = St, Nt, X.squeeze()


        elif not mixture_consistency and not refine:
            St = St.mean(0)


        if refine:
            St_hat = St.clone()
            Nt_hat = Nt.clone()
            St_abs = torch.maximum(self.X.abs() - Nt_hat.abs(), torch.tensor(0.0))
            St = St_abs * torch.exp(1j * torch.angle(St_hat))
            St = self.model._backward_transform(St).squeeze()
            St = St.mean(0)

        st = self.to_audio_tr(St).numpy().reshape(-1)

        
        if self.verbose and clean_file != None:

            if self.print_metrics:
                metrix = calc_metrics(s_ref, st, x_withoutgaussian_noise - s_ref)
                print(
                    f"Output PESQ: {metrix['pesq']:.4f} --- SI-SDR: {metrix['si_sdr']:.4f} --- ESTOI: {metrix['estoi']:.4f} --- SI-SIR: {metrix['si_sir']:.4f} ---SI-SAR: {metrix['si_sar']:.4f}",
                    end="\r",
                )
                print("") 

        
        if not self.listen_noise:
            return st, St
        
        else:

            # First, inverse transform, then average
            if not self.wiener_filter:
                Nt, Nt_postprocess = self.model._backward_transform(Nt), self.model._backward_transform(Nt)
                Nt = Nt.mean(0) 
                Nt = Nt * self.NF
            nt = self.to_audio_tr(Nt).numpy().reshape(-1)

            if self.verbose and clean_file != None:
                
                if self.print_metrics:
                    metrix_noise = calc_metrics(x_withoutgaussian_noise-s_ref, nt, s_ref) # previously calc_metrics(_ref, nt) to compute the similarity between estimated noise and clean speech. in the prediction of the noise, the speech would be considered as a noise
                    print(
                        f"Output Noise PESQ: {metrix_noise['pesq']:.4f} --- SI-SDR: {metrix_noise['si_sdr']:.4f} --- ESTOI: {metrix_noise['estoi']:.4f} --- SI-SIR: {metrix_noise['si_sir']:.4f} ---SI-SAR: {metrix_noise['si_sar']:.4f}",
                        end="\r",
                    )
                    print("")   

        
                St_postprocess = self.X - Nt_postprocess
                St_postprocess = St_postprocess.mean(0) 
                St_postprocess = St_postprocess * self.NF
                st_postprocess = self.to_audio_tr(St_postprocess).numpy().reshape(-1)                

                # st_postprocess = x_withoutgaussian_noise-nt

                if self.print_metrics:
                    metrix_postprocess = calc_metrics(s_ref, st_postprocess, x_withoutgaussian_noise-s_ref) # previously calc_metrics(_ref, nt) to compute the similarity between estimated noise and clean speech. in the prediction of the noise, the speech would be considered as a noise
                    print(
                        f"Output Postprocess_Speech PESQ: {metrix_postprocess['pesq']:.4f} --- SI-SDR: {metrix_postprocess['si_sdr']:.4f} --- ESTOI: {metrix_postprocess['estoi']:.4f} --- SI-SIR: {metrix_postprocess['si_sir']:.4f} ---SI-SAR: {metrix_postprocess['si_sar']:.4f}",
                        end="\r",
                    )
                    print("")                                   

                
            return st, St, nt, Nt, st_postprocess , St_postprocess
