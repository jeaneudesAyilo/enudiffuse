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
from sgmse.util.utils_video import load_array,resample_video, prep_video, videocap
from sgmse.data_module import SpecsDataModule
from . import InferenceAlgoRegistry


@InferenceAlgoRegistry.register("diffuseen")
class DiffUSEEN:
    def __init__(
        self,
        ckpt_path="data/checkpoints/diffusion_gen_nonlinear_transform.ckpt",
        num_E=30,
        transform_type="exponent",
        delta=1e-10,
        eps=0.03,
        snr=0.5,
        sr=16000,
        verbose=False,
        device="cuda",
        set_v_to_zero = "no",
        optimized_lambda=False,
        #the followings are not really used for DiffUSEEN        
        print_metrics=False,
        listen_noise=False,
        ckpt_noise="",    
    ):
        """
        Fast Unsupervised Diffusion-Based Speech Enhancement algorithm with explicit introduction of noise (n).

        Args:
            ckpt_path: Path to the pre-trained diffusion model.
            num_E: Number of iterations for the E step (reverse diffusion process).
            verbose: Whether to print progress information.
        """

        self.snr = snr
        self.sr = sr
        self.delta = delta
        self.num_E = num_E
        self.optimized_lambda = optimized_lambda

        self.verbose = verbose
        self.device = device
        self.scheduler = LinearScheduler(N=num_E, eps=eps)
        self.sde = OUVESDE(theta=1.5, sigma_min=0.05, sigma_max=0.5, N=num_E)

        # ==== Prior model ====
        self.model = ScoreModel.load_from_checkpoint(
            ckpt_path, base_dir="", batch_size=1, num_workers=0, kwargs=dict(gpu=False),data_module_cls=SpecsDataModule,
        )

        #for gres cluster
        self.model_cpu = ScoreModel.load_from_checkpoint(
            ckpt_path, base_dir="", batch_size=1, num_workers=0, kwargs=dict(gpu=False),data_module_cls=SpecsDataModule,
        )
        
        self.model.data_module.transform_type = transform_type
        self.model_cpu.data_module.transform_type = transform_type
        self.model.eval(no_ema=False)
        self.model.to(self.device)


        print("##### finshing launching model ######")
        self.audio_only = self.model.audio_only
        if not self.audio_only : 
            from sgmse.util.utils_video import load_array,resample_video, prep_video, videocap

            self.fps =25
            self.video_feature_type = self.model.dnn.video_feature_type
            self.vfeat_processing_order = self.model.dnn.vfeat_processing_order            
            self.set_v_to_zero = set_v_to_zero
        else: 
            self.vfeat_processing_order = "default" 


    def pick_zeta_schedule(self, schedule, t, zeta, linear_t=None, clip=50_000):
        if schedule == "none":
            return None
        if schedule == "constant":
            zeta_t = zeta
        if schedule == "lin-increase":
            zeta_t = zeta * t
        if schedule == "lin-decrease":
            zeta_t = zeta * (1 - t)
        if schedule == "half-cycle":
            zeta_t = zeta * np.sin(np.pi * t)
        if schedule == "sqrt-increase":
            zeta_t = zeta * np.sqrt(1e-10 + t)
        if schedule == "exp-increase":
            zeta_t = zeta * np.exp(t)
        if schedule == "log-increase":
            zeta_t = zeta * np.log(1 + 1e-10 + t)
        if schedule == "div-sig":
            zeta_t = zeta / t
        if schedule == "div-sig-square":
            zeta_t = zeta / t**2
        if schedule == "saw-tooth-increase":
            max_step = 0.9
            if linear_t < max_step:  # ramp from 0 to zeta0 in rho_max
                zeta_t = zeta / max_step * linear_t
            else:
                zeta_t = zeta + zeta * (max_step - linear_t) / (1 - max_step)
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


    def load_data(self, file_path, vfile_path = None):
        """
        Load speech data and compute spectrogram.
        """
        x, sr = load(file_path)
        assert sr == self.sr
        self.T_orig = x.size(1)

        # X = pad_spec(
        #     torch.unsqueeze(self.model._forward_transform(self.model._stft(x)), 0)
        # ).to(self.device)


        #for gres
        X = pad_spec(
            torch.unsqueeze(self.model_cpu._forward_transform(self.model_cpu._stft(x)), 0)
        ).to(self.device) 

        ##processing video
        if not self.audio_only:
            assert vfile_path is not None
            # if vfile_path is None : vfile_path = file_path.replace(".wav","Raw.npy")
            v,_ = self.load_visual_data(vfile_path)  
       
            if self.vfeat_processing_order in ["cut_extract"]:
                v,_=self.load_visual_data(vfile_path)           
        
        else:
            v = None          

        # print(f"######### X.shape {X.shape} #######")
        # print(f"######### v.shape {v.shape} #######")
        return x, X, v


    # def to_audio(self, specto):
    #     specto = specto * self.NF
    #     return self.model.to_audio(specto.squeeze(), self.T_orig).cpu().reshape(1, -1)

    #for gres
    def to_audio(self, specto):
        specto = specto * self.NF
        specto = specto.cpu()
        return self.model_cpu.to_audio(specto.squeeze(), self.T_orig).cpu().reshape(1, -1)    

    # def to_audio_tr(self, specto):
    #     return self.model._istft(specto, self.T_orig).cpu().reshape(1, -1)

    def to_audio_tr(self, specto):
        specto = specto.cpu()
        return self.model_cpu._istft(specto, self.T_orig).cpu().reshape(1, -1)    
    
    def predictor_corrector(self, St, t, v, laststep, dt):
        with torch.no_grad():
            # Corrector
            score = self.model.forward(St, t, v)
            std = self.sde.marginal_prob(St, t)[1]
            step_size = (self.snr * std) ** 2
            z = torch.randn_like(St)
            St = (
                St
                + step_size[:, None, None, None] * score
                + torch.sqrt(step_size * 2)[:, None, None, None] * z
            )

            # Predictor
            f, g = self.sde.sde(St, t)
            score = self.model.forward(St, t, v)
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

    def likelihood_update(self, St, N0, t, std, dt, lmbd, S_tgt):
        """
        Pseudo-likelihood update. Similar to udiffse. 
        If we are using the optimized lambda, lambda is computed automatically thanks to S_tgt 
        and the provided lambda is useless. If optimized lambda is not used, S_tgt is None and the provided lambda is used.
        """

        with torch.no_grad():
            # clean speech / noise
            theta = self.sde.theta
            mu_t = torch.exp(-theta * t)[:, None, None, None]
            _, g = self.sde.sde(St, t)          

            difference = self.X - (St / mu_t + N0 ) 
            
            nppls = (
                (1 / mu_t)
                * difference
                / ((std[:, None, None, None] / mu_t) ** 2 + self.w) 
            ).type(torch.complex64)

            if not self.optimized_lambda:
                weight = lmbd * (g**2)[:, None, None, None]
                St = St + weight * nppls * dt
            
            else:
                X_eff = S_tgt - St
                Y_eff = (g**2)[:, None, None, None] * nppls * dt
                sum_dims = [1, 2, 3]
                lmbd = torch.sum((X_eff.conj() * Y_eff), dim=sum_dims).real / torch.sum(Y_eff.abs() ** 2, dim=sum_dims)
                lmbd = torch.clamp(lmbd, min=0) * 0.5

                weight = lmbd[:, None, None, None] * (g**2)[:, None, None, None]
                St = St + weight * nppls * dt


            return St 

    def prior_sampler(self, clean_file = None, vfile_path = None):
        """
        Prior sampling algorithm to unconditionally generate a clean speech signal.
        """
        timesteps = self.scheduler.timesteps()
        self.NF = 1
        window_length = self.model.data_module.n_fft
        freq_bins_stft = 1 + window_length//2 ##256

        if self.audio_only : #unconditional generation of an audio of 5s
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

            #to make these commented lines work, one needs to unsure a padding  for the nb_stft_frame, such that it is a muliple of 64        
            #v, nb_v_frame = self.load_visual_data(vfile_path)
            # self.T_orig = int(nb_v_frame/self.fps)*self.sr
            # hop_length = self.model.data_module.hop_length
            # nb_stft_frame = int(np.floor((self.T_orig - window_length)/hop_length) ) + 1


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
                v=v,
                laststep=i == (self.num_E - 1),
                dt=dt,
            )

        st = self.to_audio(St)
        St = self.model._backward_transform(St)

        return st, St

    def posterior_sampler(self, Wt, Ht, startstep=0):
        """
        Posterior sampler algorithm that functions as the E-step for the EM process of UDiffSE.
        """
        timesteps = self.scheduler.timesteps()

        t_T = torch.tensor([timesteps[startstep]], device=self.device).repeat(self.nbatch)
        _, std_T = self.sde.marginal_prob(self.X, t_T)

        # Set the very first sample at t=1
        St = (
            torch.randn_like(self.X) * std_T[:,None,None,None]
            + self.X
        )

        # Discretised time-step
        dt = torch.tensor(1 / self.num_E, device=self.device)

        if self.verbose:
            range_i = tqdm(range(startstep, self.num_E))
        else:
            range_i = range(startstep, self.num_E)

        S0hat = St
        for i in range_i:

            t = torch.tensor([timesteps[i]], device=self.device).repeat(self.nbatch)

            if not self.optimized_lambda:
                lmbd = self.pick_zeta_schedule(
                    schedule= "constant", #"saw-tooth-increase",
                    t=torch.tensor([timesteps[i]], device=self.device),
                    zeta=self.lmbd,
                    linear_t=(self.num_E - i) / self.num_E,
                )
            else:
                lmbd=None

            # Predictor-Corrector iteration
            St, std, score, _ = self.predictor_corrector(
                St=St,
                t=t,
                v = self.visual_feature,
                laststep=i == (self.num_E - 1),
                dt=dt,
            )

            # Update parameters
            theta = self.sde.theta
            gamma_t = torch.exp(-theta * t)[:, None, None, None]
            S0hat = (
                St + torch.tensor(std**2)[:, None, None, None] * score
            )  / gamma_t

            # Noise sampling: n ~ p(n|x, st) \propto p(x|n,st) * p(n), where p(x|n,st) can be derived in the same way as in paradiffuse
            var_x = (std[:, None, None, None] / gamma_t) ** 2 + self.w 
            Mean_n = (self.Vt/(self.Vt + var_x)) * (self.X - S0hat)
            Sigma_n = ((self.Vt * var_x)/(self.Vt + var_x))
            N0 = Mean_n + torch.sqrt(Sigma_n) * torch.randn_like(S0hat)
            
            # Likelihood term & parameter update
            if not self.optimized_lambda:
                S_tgt=None
            else:
                S_tgt= S0hat * gamma_t

            St = self.likelihood_update(
                St=St,
                N0=Mean_n,
                t=t,
                std=std,
                dt=dt,
                lmbd=lmbd,
                S_tgt = S_tgt
            )

            Wt, Ht = self.parameter_update(Mean_n, Sigma_n, Wt, Ht)
            self.Vt = Wt @ Ht

        # print(f"####### Mean_n shape {Mean_n.shape} ######")

        return St, Mean_n

    def parameter_update(self, Mean_n, Sigma_n, W, H):
        Vm = ((Mean_n).abs().pow(2) + Sigma_n).mean(0).unsqueeze(0)
        # temporary
        V = W @ H

        # Update W
        num = (Vm * V.pow(-2)) @ H.permute(0, 1, 3, 2)
        den = V.pow(-1) @ H.permute(0, 1, 3, 2)
        W = W * (num / den)
        W = torch.maximum(W, torch.tensor([self.delta], device=self.device))

        # Update V
        V = W @ H

        # Update H
        num = W.permute(0, 1, 3, 2) @ (Vm * V.pow(-2))  # transpose
        den = W.permute(0, 1, 3, 2) @ V.pow(-1)
        H = H * (num / den)
        H = torch.maximum(H, torch.tensor([self.delta], device=self.device))

        # Normalise
        norm_factor = torch.sum(W.abs(), axis=2)
        W = W / torch.unsqueeze(norm_factor, 2)
        H = H * torch.unsqueeze(norm_factor, 3)

        return W, H

    def run(
        self,
        mix_file,
        clean_file=None,
        video_file = None, 
        num_EM=1,
        lmbd=1.75, #1.90,
        nbatch=2,
        nmf_rank=4,
        project_every_k_steps=1,
        startstep=0,
        wiener_filter=False,
        w = 5e-4,
        S0=None,
        N0=None,
    ):
        self.lmbd = lmbd
        self.project_every_k_steps = project_every_k_steps
        self.nbatch = nbatch
        self.wiener_filter = wiener_filter
        self.w = w

        x, X, v = self.load_data(file_path = mix_file, vfile_path = video_file)
        self.x = x
        self.NF = X.abs().max()
        X = X / self.NF

        if self.verbose and clean_file != None:
            s_ref, S_ref,_  = self.load_data(clean_file,vfile_path = video_file)
            self.s_ref = s_ref
            self.S_ref = S_ref
            s_ref = s_ref.numpy().reshape(-1)
            x = x.numpy().reshape(-1)
            metrix = calc_metrics(s_ref, x, x-s_ref)
            print(
                f"Input PESQ: {metrix['pesq']:.4f} --- SI-SDR: {metrix['si_sdr']:.4f} --- ESTOI: {metrix['estoi']:.4f}  --- SI-SIR: {metrix['si_sir']:.4f} --- SI-SAR: {metrix['si_sar']:.4f}",
                end="\r",
            )
            print("")

        if S0 is None:
            self.X = X.repeat(self.nbatch, 1, 1, 1)
        else:
            self.X = S0


        if not self.audio_only:        
            if self.vfeat_processing_order in ["cut_extract"]:
                if self.video_feature_type in  ["resnet", "avhubert"]: 
                    self.visual_feature = v.repeat(self.nbatch, 1, 1, 1, 1) #(b,1,nb_frame,h,w)
                
                elif self.video_feature_type in  ["flow_avse"]: 
                    self.visual_feature = v.repeat(self.nbatch, 1, 1, 1) #(b,nb_frame,h,w)

                elif self.video_feature_type in  ["raw_image"]:              
                    self.visual_feature = v.repeat(self.nbatch, 1, 1) #(b,h*w,nb_frame)
            
            
        else : self.visual_feature = None


        metrix = {"pesq": 0.0, "si_sdr": 0.0, "estoi": 0.0}

        # Initialise W and H (NMF matrices)
        _, _, T, F = X.shape
        Wt = torch.rand(T, nmf_rank, device=self.device).clamp_(min=self.delta)[
            None, None, :, :
        ]
        Ht = torch.rand(nmf_rank, F, device=self.device).clamp_(min=self.delta)[
            None, None, :, :
        ]
        self.Vt = Wt @ Ht

        # fUDiffSE algorithm
        St, N0 = self.posterior_sampler(Wt, Ht, startstep=startstep)

        if self.wiener_filter:
            X = X * self.NF
            St_abs_2 = (St.abs().pow(2)/(St.abs().pow(2) + N0.abs().pow(2))).mean(0) * X.abs().pow(2)
            St = St_abs_2.sqrt() * torch.exp(1j * torch.angle(St.mean(0)))
            St = self.model._backward_transform(St).squeeze()

        else:
            # First, inverse transform, then average
            St = self.model._backward_transform(St)
            St = St.mean(0) 
        
        st = self.to_audio_tr(St).numpy().reshape(-1)

        if self.verbose and clean_file != None:
            metrix = calc_metrics(s_ref, st,  x-s_ref)
            print(
                f"Output PESQ: {metrix['pesq']:.4f} --- SI-SDR: {metrix['si_sdr']:.4f} --- ESTOI: {metrix['estoi']:.4f} --- SI-SIR: {metrix['si_sir']:.4f} --- SI-SAR: {metrix['si_sar']:.4f}",
                end="\r",
            )
            print("")

        return st, St