#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DEPSE-IL algorithm: Diffusion-based Explicit Posterior Sampling for Speech Enhancement with Intractable Likelihood

@author: Mostafa Sadeghi,  copyright (c) 2025, Inria 

Reference paper: https://ieeexplore.ieee.org/document/11053679/
"""

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

@InferenceAlgoRegistry.register("depse_il")
class DEPSE_IL:
	def __init__(
		self,
		ckpt_path="data/checkpoints/diffusion_gen_nonlinear_transform.ckpt",
		ckpt_noise="", 
		num_E=30,
		transform_type= "exponent", #"normalise",
		delta=1e-10,
		eps=0.03,
		snr=0.5,
		sr=16000,
		verbose=False,
		device="cuda",
	):
		"""
		Fast Unsupervised Diffusion-Based Speech Enhancement (fUDiffSE) algorithm.

		Args:
			ckpt_path: Path to the pre-trained diffusion model.
			num_E: Number of iterations for the E step (reverse diffusion process).
			verbose: Whether to print progress information.
		"""

		self.snr = snr
		self.sr = sr
		self.delta = delta
		self.num_E = num_E

		self.verbose = verbose
		self.device = device
		self.scheduler = LinearScheduler(N=num_E, eps=eps)
		self.sde = OUVESDE(theta=1.5, sigma_min=0.05, sigma_max=0.5, N=num_E)

		# ==== Prior model ====
		self.model = ScoreModel.load_from_checkpoint(
			ckpt_path, base_dir="", batch_size=1, num_workers=0, kwargs=dict(gpu=False)
		)

		#for gres cluster
		self.model_cpu = ScoreModel.load_from_checkpoint(
			ckpt_path, base_dir="", batch_size=1, num_workers=0, kwargs=dict(gpu=False)
		)

		self.model.data_module.transform_type = transform_type
		self.model.eval(no_ema=False)
		self.model.to(self.device)

	def load_data(self, file_path):
		"""
		Load speech data and compute spectrogram.
		"""
		x, sr = load(file_path)
		assert sr == self.sr
		self.T_orig = x.size(1)

		# X = pad_spec(
		# 	torch.unsqueeze(self.model._forward_transform(self.model._stft(x)), 0)
		# ).to(self.device)

		#for gres
		X = pad_spec(
			torch.unsqueeze(self.model_cpu._forward_transform(self.model_cpu._stft(x)), 0)
		).to(self.device) 

		return x, X

	# def to_audio(self, specto):
	# 	specto = specto * self.NF
	# 	return self.model.to_audio(specto.squeeze(), self.T_orig).cpu().reshape(1, -1)

	#for gres
	def to_audio(self, specto):
		specto = specto * self.NF
		specto = specto.cpu()
		return self.model_cpu.to_audio(specto.squeeze(), self.T_orig).cpu().reshape(1, -1) 

	# def to_audio_tr(self, specto):
	# 	return self.model._istft(specto, self.T_orig).cpu().reshape(1, -1)

	def to_audio_tr(self, specto):
		specto = specto.cpu()
		return self.model_cpu._istft(specto, self.T_orig).cpu().reshape(1, -1)


	@torch.no_grad()
	def cal_mean_variance_PC(self, St, t, dt=1., laststep=False, v=None):
		"""
		Calculate the mean and variance for $q(S_{t-1} | S_t, S_0)$
		"""

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

		mean = St - f * dt + (g**2)[:, None, None, None] * score * dt
		var = g[:, None, None, None]**2 * dt

		return mean, var, score

	@torch.no_grad()
	def cal_posterior_mean_variance_v2(self, St, t, dt):
		"""
		Calculate the mean and variance for $q(S_{t-1} | S_t, S_0)$
		"""

		mean_prior, var_prior, score = self.cal_mean_variance_PC(St, t, dt)

		sigma2_t_1 = self.sde.marginal_prob(St, t)[1] ** 2 
		gamma = self.sde.theta

		Sigma_c_t = (sigma2_t_1) / (torch.exp(torch.tensor(-2*gamma*t)))
		
		Sigma_x_t_o = Sigma_c_t + self.w
		Sigma_x_t = Sigma_x_t_o * torch.exp(torch.tensor(-2*gamma*t))

		# var is a constant
		mean_posterior = Sigma_x_t / (Sigma_x_t + var_prior) * mean_prior + (var_prior) / (Sigma_x_t + var_prior) * (torch.exp(torch.tensor(-gamma*t)) * (self.X - self.N0hat)) 
		var_posterior = (var_prior * Sigma_x_t) / (Sigma_x_t + var_prior) 

		return mean_posterior, var_posterior, score

	@torch.no_grad()
	def cal_posterior_mean_variance(self, St, t, dt):
		"""
		Calculate the mean and variance for $q(S_{t-1} | S_t, S_0)$
		"""

		mean_prior, var_prior, score = self.cal_mean_variance_PC(St, t, dt)

		gamma = self.sde.theta
		sigma2_t_1 = self.sde.marginal_prob(St, t)[1] ** 2 

		Sigma_c_t = (sigma2_t_1) / (torch.exp(torch.tensor(-2*gamma*t)))
		
		Sigma_x_t = (Sigma_c_t + self.Vt) * torch.exp(torch.tensor(-2*gamma*t))

		# var is a constant
		mean_posterior = Sigma_x_t / (Sigma_x_t + var_prior) * mean_prior + (var_prior) / (Sigma_x_t + var_prior) * (torch.exp(torch.tensor(-gamma*t)) * self.X) 
		var_posterior = (var_prior * Sigma_x_t) / (Sigma_x_t + var_prior) 

		return mean_posterior, var_posterior, score

	@torch.no_grad()
	def cal_posterior_mean_variance_nmf(self, St, t, dt):
		"""
		Calculate the mean and variance for $q(S_{t-1} | S_t, S_0)$
		"""

		mean_prior, var_prior, score = self.cal_mean_variance_PC(St, t, dt)

		gamma = self.sde.theta
		sigma2_t_1 = self.sde.marginal_prob(St, t)[1] ** 2 

		Sigma_s_t = (sigma2_t_1) / (torch.exp(torch.tensor(-2*gamma*t)))

		Sigma_c_t = (Sigma_s_t * self.V_s) / (Sigma_s_t + self.V_s)

		Mean_c_t_factor = (torch.exp(torch.tensor(gamma*t)) * self.V_s) / (Sigma_s_t + self.V_s)

		Sigma_x_t = (Sigma_c_t + self.Vt) / (Mean_c_t_factor ** 2)

		# var is a constant
		mean_posterior = Sigma_x_t / (Sigma_x_t + var_prior) * mean_prior + (var_prior) / (Sigma_x_t + var_prior) * (self.X / Mean_c_t_factor) 
		var_posterior = (var_prior * Sigma_x_t) / (Sigma_x_t + var_prior) 

		return mean_posterior, var_posterior, score


	@torch.no_grad()
	def sample_one_step_prior(self, St, t, dt, laststep=False):
		"""
		Calculate $S{t-1}$ according to $St$
		"""
		mean, var, score = self.cal_mean_variance_PC(St, t, dt)

		z = torch.randn_like(St) if not laststep else 0
		S_t_1 = mean + torch.sqrt(var) * z

		if torch.isnan(S_t_1).int().sum() != 0:
			raise ValueError("nan in tensor!")

		return S_t_1, score

	@torch.no_grad()
	def sample_one_step_posterior(self, St, t, dt, laststep=False, likelihood_inclusion=True):
		"""
		Calculate $S{t-1}$ according to $St$
		"""
		if likelihood_inclusion:

			if self.s0_nmf:
				mean, var, score = self.cal_posterior_mean_variance_nmf(St, t, dt)
			else:
				mean, var, score = self.cal_posterior_mean_variance(St, t, dt)
			
		else:
			mean, var, score = self.cal_mean_variance_PC(St, t, dt)

		z = torch.randn_like(St) if not laststep else 0
		S_t_1 = mean + torch.sqrt(var) * z

		# var is a constant
		if torch.isnan(S_t_1).int().sum() != 0:
			raise ValueError("nan in tensor!")

		self.mean = mean
		self.var = var

		return S_t_1, score


	@torch.no_grad()
	def sample_one_step_posterior_v2(self, St, t, dt, laststep=False, likelihood_inclusion=True):
		"""
		Calculate $S{t-1}$ according to $St$
		"""
		if likelihood_inclusion:
			mean, var, score = self.cal_posterior_mean_variance_v2(St, t, dt)
		else:
			mean, var, score = self.cal_mean_variance_PC(St, t, dt)

		z = torch.randn_like(St) if not laststep else 0
		S_t_1 = mean + torch.sqrt(var) * z

		# var is a constant
		if torch.isnan(S_t_1).int().sum() != 0:
			raise ValueError("nan in tensor!")

		return S_t_1, score


	@torch.no_grad()
	def prior_sampler(self):

		self.T_orig = 80000
		self.NF = 1
		timesteps = self.scheduler.timesteps()

		# # Set the very first sample at t=1
		# St = torch.randn(
		# 	1, 1, 256, 640, dtype=torch.cfloat, device=self.device
		# ) 

		St = torch.randn(
			1, 1, 256, 640, dtype=torch.cfloat, device=self.device
		) * self.sde._std(torch.ones(1, device=self.device))
		
		# Discretised time-step
		dt = torch.tensor(1 / self.num_E, device=self.device)

		# Sampling iterations t_up = t, t = t-1
		for i in tqdm(range(0, self.num_E)):
			t = torch.tensor([timesteps[i]], device=self.device)

			St = self.sample_one_step_prior(St, t, dt, laststep=(i == (self.num_E - 1)))

		st = self.to_audio(St)
		St = self.model._backward_transform(St)

		return st, St


	@torch.no_grad()
	def posterior_sampler(self, Wt, Ht, Wt_s=None, Ht_s=None, St_0=None, likelihood_inclusion=True, startstep=0):

		self.NF = 1
		timesteps = self.scheduler.timesteps()

		t_T = torch.tensor([timesteps[startstep]], device=self.device).repeat(self.nbatch)
		_, std_T = self.sde.marginal_prob(self.X, t_T)

		# Set the very first sample at t=1
		if St_0 is None:
			St = (
				torch.randn_like(self.X) * std_T[:,None,None,None]
				+ self.X
			)
		else:
			St = (
				torch.randn_like(self.X) * std_T[:,None,None,None]
				+ St_0
			)			

		# Discretised time-step
		dt = torch.tensor(1 / self.num_E, device=self.device)

		if self.verbose:
			range_i = tqdm(range(0, self.num_E))
		else:
			range_i = range(0, self.num_E)

		S0hat = St
		# Sampling iterations
		for i in range_i:
			t = torch.tensor([timesteps[i]], device=self.device)

			# St_old = St.clone()
			St, score = self.sample_one_step_posterior(St, t, dt, laststep=(i == (self.num_E - 1)), likelihood_inclusion=likelihood_inclusion)

			if not self.full_EM and not self.oracle_noise:
				theta = self.sde.theta
				std = self.sde.marginal_prob(St, t)[1]
				gamma_t = torch.exp(-theta * t)[:, None, None, None]

				S0hat = (
				  St + torch.tensor(std**2)[:, None, None, None] * score
				) #/ gamma_t

				if self.s0_nmf:
					Wt_s, Ht_s = self.parameter_update_NMF(self.model._backward_transform(S0hat), Wt_s, Ht_s)
					self.V_s = Wt_s @ Ht_s

			if not self.full_EM and not self.oracle_noise:
				if self.normal_nmf:
					Wt, Ht = self.parameter_update_NMF(self.X - S0hat, Wt, Ht)
				else:
					Wt, Ht = self.parameter_update_NMF_NEW(self.X - self.mean/gamma_t, self.var / (gamma_t**2), Wt, Ht, a = (std**2) / (gamma_t**2))

				self.Vt =  Wt @ Ht
				
		return St


	@torch.no_grad()
	def posterior_sampler_v2(self, Wt, Ht, St_0=None, likelihood_inclusion=True, startstep=0):

		self.NF = 1
		timesteps = self.scheduler.timesteps()

		t_T = torch.tensor([timesteps[startstep]], device=self.device).repeat(self.nbatch)
		_, std_T = self.sde.marginal_prob(self.X, t_T)

		# Set the very first sample at t=1
		if St_0 is None:
			St = (
				torch.randn_like(self.X) * std_T[:,None,None,None]
				+ self.X
			)
		else:
			St = (
				torch.randn_like(self.X) * std_T[:,None,None,None]
				+ St_0
			)			

		# Discretised time-step
		dt = torch.tensor(1 / self.num_E, device=self.device)

		S0hat = St
		self.N0hat = self.X - St

		# Sampling iterations
		for i in tqdm(range(0, self.num_E)):
			t = torch.tensor([timesteps[i]], device=self.device)

			# St_old = St.clone()
			St, score = self.sample_one_step_posterior_v2(St, t, dt, laststep=(i == (self.num_E - 1)), likelihood_inclusion=True)

			theta = self.sde.theta
			std = self.sde.marginal_prob(St, t)[1]
			gamma_t = torch.exp(-theta * t)[:, None, None, None]
			S0hat = (
				St + torch.tensor(std**2)[:, None, None, None] * score
			) / gamma_t

			# Noise sampling: n ~ p(n|x, st) \propto p(x|n,st) * p(n), where p(x|n,st) can be derived in the same way as in paradiffuse
			var_x = (std[:, None, None, None] / gamma_t) ** 2 + self.w 
			Mean_n = (self.Vt/(self.Vt + var_x)) * (self.X - S0hat)
			Sigma_n = ((self.Vt * var_x)/(self.Vt + var_x))
			self.N0hat = Mean_n
			
			Wt, Ht = self.parameter_update(Mean_n, Sigma_n, Wt, Ht)
			self.Vt =  Wt @ Ht

		return St, Mean_n

	def parameter_update_NMF(self, X_init_st, W, H):
		Vm = (X_init_st).abs().pow(2).mean(0).unsqueeze(0)
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

	def parameter_update_NMF_NEW(self, Mean_n, Sigma_n, W, H, a=0.0):
		Vm = ((Mean_n).abs().pow(2) + Sigma_n).mean(0).unsqueeze(0)
		
		# temporary: current reconstruction estimate V = W @ H
		V = W @ H

		# Update W using multiplicative rules adapted to the new loss
		# Note: we replace V.pow(-1) with (a+V).pow(-1), and similarly for V.pow(-2)
		num = (Vm * (a + V).pow(-2)) @ H.permute(0, 1, 3, 2)
		den = (a + V).pow(-1) @ H.permute(0, 1, 3, 2)
		W = W * (num / den)
		# Ensure non-negativity with a small positive constant delta
		W = torch.maximum(W, torch.tensor([self.delta], device=self.device))

		# Recompute V after updating W
		V = W @ H

		# Update H using the similar adapted multiplicative rule
		num = W.permute(0, 1, 3, 2) @ (Vm * (a + V).pow(-2))
		den = W.permute(0, 1, 3, 2) @ (a + V).pow(-1)
		H = H * (num / den)
		H = torch.maximum(H, torch.tensor([self.delta], device=self.device))

		# Normalise the factors: adjust W and H to maintain scale (if required)
		norm_factor = torch.sum(W.abs(), axis=2)
		W = W / torch.unsqueeze(norm_factor, 2)
		H = H * torch.unsqueeze(norm_factor, 3)
		
		return W, H


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

	def NMF_compupation(self, X_diff, W_init, H_init, nmf_rank=4):
		N_abs_2 = (X_diff).abs().pow(2).mean(0).squeeze().cpu().numpy()
		K = nmf_rank
		W_init = W_init.cpu().numpy().squeeze()
		H_init = H_init.cpu().numpy().squeeze()
		nmfmodel = NMF(
			n_components=K,
			init="custom",
			solver="mu",
			beta_loss="itakura-saito",
			random_state=0,
			verbose=False,
			max_iter=1000,
		)
		Wt = nmfmodel.fit_transform(
			X=N_abs_2 + self.delta, W=np.float32(W_init), H=np.float32(H_init)
		)
		Ht = nmfmodel.components_
		Wt = torch.from_numpy(Wt).to(self.device).unsqueeze(0).unsqueeze(0)
		Ht = torch.from_numpy(Ht).to(self.device).unsqueeze(0).unsqueeze(0)

		return Wt, Ht

	def run(
		self,
		mix_file,
		video_file=None,
		clean_file=None,
		num_EM=1,
		lmbd=1.75,
		nbatch=2,
		nmf_rank=4,
		nmf_rank_s=8,
		project_every_k_steps=1,
		oracle_noise=False,
		startstep=0,
		wiener_filter=False,
		w = 5e-4,
		snr=0.5,
		se_method="v1",
		s0_nmf=False,
		normal_nmf=True,
		do_corrector=False,
		full_EM=False,
		NMF_sklearn=False,
		S0=None,
		N0=None,
	):

		self.normal_nmf = normal_nmf
		self.oracle_noise = oracle_noise
		self.s0_nmf = s0_nmf
		self.do_corrector = do_corrector
		self.lmbd = lmbd
		self.project_every_k_steps = project_every_k_steps
		self.nbatch = nbatch
		self.wiener_filter = wiener_filter
		self.w = w
		self.full_EM = full_EM
		self.snr = snr
		x, X = self.load_data(mix_file)
		self.x = x
		self.NF = X.abs().max()
		X = X / self.NF

		# print("### using depse_il ###")

		if self.verbose and clean_file != None:
			s_ref, S_ref = self.load_data(clean_file)
			self.s_ref = s_ref
			self.S_ref = S_ref
			s_ref = s_ref.numpy().reshape(-1)
			x = x.numpy().reshape(-1)
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

		# Initialise W and H (NMF matrices)
		_, _, T, F = X.shape
		Wt = torch.rand(T, nmf_rank, device=self.device).clamp_(min=self.delta)[
			None, None, :, :
		]
		Ht = torch.rand(nmf_rank, F, device=self.device).clamp_(min=self.delta)[
			None, None, :, :
		]
		self.Vt = Wt @ Ht

		if self.s0_nmf:

			St = (
				torch.randn_like(self.X) * self.sde._std(torch.ones(1, device=self.device))
				+ self.X
			)

			# prior of s_0
			Wt_s = torch.rand(T, nmf_rank_s, device=self.device).clamp_(min=self.delta)[
				None, None, :, :
			]
			Ht_s = torch.rand(nmf_rank_s, F, device=self.device).clamp_(min=self.delta)[
				None, None, :, :
			]
			self.V_s = Wt_s @ Ht_s

			# S_est = self.model._backward_transform(self.S_ref).repeat(self.nbatch, 1, 1, 1)
			S_est = self.model._backward_transform(St)

			for i in range(100):
				Wt_s, Ht_s = self.parameter_update_NMF(S_est, Wt_s, Ht_s)
				self.V_s = Wt_s @ Ht_s

		else:
			Wt_s = Wt
			Ht_s = Ht


		if oracle_noise:
			# ======== oracle calculation ========
			N_abs_2 = torch.squeeze(X - S_ref).abs().pow(2).cpu().numpy()
			T, F = N_abs_2.shape
			K = nmf_rank
			W_init = np.maximum(np.random.rand(T, K), self.delta)
			H_init = np.maximum(np.random.rand(K, F), self.delta)
			nmfmodel = NMF(
				n_components=K,
				init="custom",
				solver="mu",
				beta_loss="itakura-saito",
				random_state=0,
				verbose=False,
				max_iter=1000,
			)
			Wt = nmfmodel.fit_transform(
				X=N_abs_2 + self.delta, W=np.float32(W_init), H=np.float32(H_init)
			)
			Ht = nmfmodel.components_
			Wt = torch.from_numpy(Wt).to(self.device).unsqueeze(0).unsqueeze(0)
			Ht = torch.from_numpy(Ht).to(self.device).unsqueeze(0).unsqueeze(0)
			self.Vt = Wt @ Ht

		if se_method=="v1":
			if not self.full_EM:
				St = self.posterior_sampler(Wt, Ht, Wt_s, Ht_s)
			else:
				# EM algorithm
				St_0=None
				for j in range(num_EM):
					St = self.posterior_sampler(Wt, Ht, likelihood_inclusion=(j>=0))
					# St_0 = St.clone()

					if NMF_sklearn:
						Wt, Ht = self.NMF_compupation(self.X - St, Wt, Ht, nmf_rank=nmf_rank)
					else:
						Wt, Ht = self.parameter_update_NMF(self.X - St, Wt, Ht)
					self.Vt =  Wt @ Ht

					if self.verbose and clean_file != None:
						# First, inverse transform, then average
						St_tmp = self.model._backward_transform(St)
						St_tmp = St_tmp.mean(0) 
						st_tmp = self.to_audio_tr(St_tmp).numpy().reshape(-1)
						metrix = calc_metrics(s_ref, st_tmp, n=x - s_ref)
						print(
							f"Input PESQ: {metrix['pesq']:.4f} --- SI-SDR: {metrix['si_sdr']:.4f} --- SI-SIR: {metrix['si_sir']:.4f} --- SI-SAR: {metrix['si_sar']:.4f} --- ESTOI: {metrix['estoi']:.4f}",
							end="\r",
						)
						print("")

		elif se_method=="v2":
			if not self.full_EM:
				St, N0 = self.posterior_sampler_v2(Wt, Ht)

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
			metrix = calc_metrics(s_ref, st, n=x - s_ref)
			print(
				f"Output PESQ: {metrix['pesq']:.4f} --- SI-SDR: {metrix['si_sdr']:.4f} --- SI-SIR: {metrix['si_sir']:.4f} --- SI-SAR: {metrix['si_sar']:.4f} --- ESTOI: {metrix['estoi']:.4f}",
				end="\r",
			)
			print("")

		return st, St

