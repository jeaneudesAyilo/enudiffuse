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
from torchaudio import load, save
from . import InferenceAlgoRegistry

@InferenceAlgoRegistry.register("joint_paradiffusein")
class JointParaDiffUSEIN:
	def __init__(
		self,
		ckpt_path="", #ckpt_speech
		ckpt_noise="", ## ckpt_noise will always be empty here
		num_E=30,
		transform_type="exponent",
		delta=1e-10,
		eps=0.03,
		snr=0.5,
		sr=16000,
		verbose=False,
		listen_noise = False,
		device="cuda",
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
		
		# ==== For prior speech model ====        
		self.sde = OUVESDE(theta=1.5, sigma_min=0.05, sigma_max=0.5, N=num_E)

		
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

		
		# # ==== For prior noise model ====        
		# self.sde_noise = OUVESDE(theta=1.5, sigma_min=0.05, sigma_max=0.5, N=num_E)
		
		# self.model_noise = ScoreModel.load_from_checkpoint(
		# 	ckpt_noise, base_dir="", batch_size=1, num_workers=0, kwargs=dict(gpu=False)
		# )

		# self.model_noise.data_module.transform_type = transform_type
		# self.model_noise.eval(no_ema=False)
		# self.model_noise.to(self.device)


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

	def load_data(self, file_path, add_noise=False):
		"""
		Load speech data and compute spectrogram.
		"""
		x, sr = load(file_path)
		if add_noise:
			x += 1e-6*torch.randn_like(x)
		assert sr == self.sr
		self.T_orig = x.size(1)

		# X = pad_spec(
		# 	torch.unsqueeze(self.model._forward_transform(self.model._stft(x)), 0)
		# ).to(self.device)

		#for gres cluster
		X = pad_spec(
			torch.unsqueeze(self.model_cpu._forward_transform(self.model_cpu._stft(x)), 0)
		).to(self.device) 

		return x, X
	
	# def to_audio(self, specto):
	# 	return self.model.to_audio(specto.squeeze(), self.T_orig).cpu().reshape(1, -1)

	#for gres
	def to_audio(self, specto):
		specto = specto.cpu()
		return self.model_cpu.to_audio(specto.squeeze(), self.T_orig).cpu().reshape(1, -1) 

	# def to_audio_tr(self, specto):
	# 	return self.model._istft(specto, self.T_orig).cpu().reshape(1, -1)

	def to_audio_tr(self, specto): #for gres cluster
		specto = specto.cpu()
		return self.model_cpu._istft(specto, self.T_orig).cpu().reshape(1, -1)    


	def predictor_corrector(self, St, label, t, laststep, dt, snr=0.5, v=None):

		std = self.sde.marginal_prob(St, t)[1]

		# Corrector  
		with torch.no_grad():            
			score = self.model.forward(St, t, label)
			step_size = (snr * std) ** 2
			z = torch.randn_like(St)
			St = (
				St
				+ step_size[:, None, None, None] * score
				+ torch.sqrt(step_size * 2)[:, None, None, None] * z
			)

		# Predictor
		with torch.no_grad():
			f, g = self.sde.sde(St, t)
			score = self.model.forward(St, t, label)
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

	def likelihood_update_individual(self, St, S0hat, t, std, std_noise, dt, lmbd, w_up=True):
		"""
		Pseudo-likelihood update. Similar to udiffse
		"""

		with torch.no_grad():
			# clean speech / noise
			theta = self.sde.theta
			mu_t = torch.exp(-theta * t)[:, None, None, None]
			_, g = self.sde.sde(St, t)          

			z = torch.zeros_like(St) 
			score_noise = self.model.forward(self.X - (St / mu_t) - (std[:, None, None, None] / mu_t) * z, t, self.label_noise)
			# score_noise = score_model.forward(self.X - S0hat- (std[:, None, None, None] / mu_t) * z, t)

			nppls = -(
				(1 / mu_t)
				* score_noise
			).type(torch.complex64)

			weight = lmbd * (g**2)[:, None, None, None]
			St = St + weight * nppls * dt

			return St 

	def prior_sampler(self, clean_file = None, vfile_path = None, noise=False):
		"""
		Prior sampling algorithm to (un)conditionally generate a clean speech or noise signal.
		"""

		self.prior_sampling = True 

		if not noise : 
			label = torch.tensor(np.array([1], dtype=np.int64),device=self.device)        
		else : 
			label = torch.tensor(np.array([0], dtype=np.int64),device=self.device)
			

		timesteps = self.scheduler.timesteps()
		self.NF = 1
		self.T_orig = 80000

		# Set the very first sample at t=1
		St = torch.randn(
			1, 1, 256, 640, dtype=torch.cfloat, device=self.device
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
		self.prior_sampling = False
		t = torch.tensor([timesteps[startstep]], device=self.device).repeat(self.nbatch)
		S_mean, _ = self.sde.marginal_prob(self.X, t)

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

		# Discretised time-step
		dt = torch.tensor(1 / self.num_E, device=self.device)

		if self.verbose:
			range_i = tqdm(range(startstep, self.num_E))
		else:
			range_i = range(startstep, self.num_E)

		if S0 is not None:
			S0hat = S0
		else:
			S0hat = St
		std = self.sde.marginal_prob(St, t)[1]
		for i in range_i:
			
			t = torch.tensor([timesteps[i]], device=self.device).repeat(self.nbatch)

			# update St: p(St)
			St, std, S_score, _ = self.predictor_corrector(
				St=St,
				t=t,
				label = self.label_speech,
				laststep=i == (self.num_E - 1),
				dt=dt,				
			)

			# N0hat
			theta = self.sde.theta
			gamma_t = torch.exp(-theta * t)[:, None, None, None]

			# S0hat
			S0hat = (
				St + torch.tensor(std**2)[:, None, None, None] * S_score
			) / gamma_t


			lmbd = self.pick_zeta_schedule(
				schedule="constant",
				t=torch.tensor([timesteps[i]], device=self.device),
				zeta=self.lmbd,
				linear_t=(self.num_E - i) / self.num_E,
				max_step=0.99,
				decay_rate=1.0, 
				increase_rate=1.0,
			)

				
			# P(X| St, N0hat)
			St = self.likelihood_update_individual(
				St=St,
				S0hat=S0hat,
				t=t,
				std=std,
				std_noise=std,
				dt=dt,
				lmbd=lmbd,
			)

			if self.print_progress:
				st = self.to_audio(St.mean(0)).numpy().reshape(-1)
				metrix = calc_metrics(self.s_ref.numpy().reshape(-1), st, self.x.numpy().reshape(-1) - self.s_ref.numpy().reshape(-1))
				# print(
				# 	f"Output PESQ: {metrix['pesq']:.4f} --- SI-SDR: {metrix['si_sdr']:.4f} --- ESTOI: {metrix['estoi']:.4f} --- Data fidelity loss {torch.norm(self.X - S0hat - N0hat, p='fro')/torch.norm(self.X, p='fro'):.4f}",
				# 	end="\r",
				# )

				print(
					f"Output PESQ: {metrix['pesq']:.4f} --- SI-SDR: {metrix['si_sdr']:.4f} --- ESTOI: {metrix['estoi']:.4f}",
					end="\r",
				)

				print("")
				
		Nt = self.X - St
		return St, Nt

	def run(
		self,
		mix_file,
		video_file=None,
		clean_file=None,     
		lmbd=1.00,
		nbatch=2,
		num_EM=1,
		nmf_rank=4,
		project_every_k_steps=1,
		std_measurement = 0.15,       
		startstep=0,
		wiener_filter=True,
		mixture_consistency=False,
		refine=False,
		print_progress=False,
		S0=None,
		N0=None,
	):
		self.lmbd = lmbd
		self.project_every_k_steps = project_every_k_steps
		self.nbatch = nbatch
		self.std_measurement = std_measurement
		self.wiener_filter = wiener_filter
		self.print_progress = print_progress

		x, X = self.load_data(file_path = mix_file, add_noise=True)
		self.x = x
		self.NF = X.abs().max()
		X = X / self.NF

		if self.verbose and clean_file != None:
			s_ref, S_ref = self.load_data(file_path=clean_file)
			self.s_ref = s_ref
			self.S_ref = S_ref
			s_ref = s_ref.numpy().reshape(-1)
			x = x.numpy().reshape(-1)
			n_ref = x - s_ref
			metrix = calc_metrics(s_ref, x, n=x - s_ref)
			print(
				f"Input PESQ: {metrix['pesq']:.4f} --- SI-SDR: {metrix['si_sdr']:.4f} --- SI-SIR: {metrix['si_sir']:.4f} --- SI-SAR: {metrix['si_sar']:.4f} --- ESTOI: {metrix['estoi']:.4f}",
				end="\r",
			)
			print("")

		if S0 is None:
			self.X = X.repeat(self.nbatch, 1, 1, 1)
		else:
			self.X = S0

		metrix = {"pesq": 0.0, "si_sdr": 0.0, "estoi": 0.0}

		self.label_speech = torch.tensor(np.array([1]*self.nbatch, dtype=np.int64),device=self.device)
		self.label_noise = torch.tensor(np.array([0]*self.nbatch, dtype=np.int64),device=self.device)

		St, Nt = self.posterior_sampler(startstep=startstep, S0=S0, N0=N0)

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
			Nt = self.model._backward_transform(Nt).squeeze()
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
			# save(
			# 	"./s_hat.wav",
			# 	self.to_audio_tr(St).type(torch.float32).cpu().squeeze().unsqueeze(0),
			# 	self.sr,
			# )
			metrix = calc_metrics(s_ref, st, n=x - s_ref)
			print(
				f"Output PESQ: {metrix['pesq']:.4f} --- SI-SDR: {metrix['si_sdr']:.4f} --- SI-SIR: {metrix['si_sir']:.4f} --- SI-SAR: {metrix['si_sar']:.4f} --- ESTOI: {metrix['estoi']:.4f}",
				end="\r",
			)
			print("")       
		
		if not self.listen_noise:
			return st, St
		
		else:

			# First, inverse transform, then average
			if not self.wiener_filter:
				Nt = self.model._backward_transform(Nt)
				Nt = Nt.mean(0) 
				Nt = Nt * self.NF
			nt = self.to_audio_tr(Nt).numpy().reshape(-1) 
 
			return st, St, nt, Nt
