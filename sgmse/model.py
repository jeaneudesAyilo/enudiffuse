import time
from math import ceil
import warnings

import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage

from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.other import pad_spec


class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", choices=("mse", "mae"), help="The type of loss function to use.")
        parser.add_argument(
            "--joint_noise_clean_speech_training",
            action="store_true",
            help="Specify if the score model is trained jointly on noise and clean speech", #$#
        )      ### this is not required in fact. We can get it by kwargs["initial_joint_noise_clean_speech_training"]
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=3e-2, joint_noise_clean_speech_training=False,
        num_eval_files=20, loss_type='mse', data_module_cls=None, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(joint_noise_clean_speech_training= joint_noise_clean_speech_training, **kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files

        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, joint_noise_clean_speech_training= joint_noise_clean_speech_training,
                                            gpu=kwargs.get('gpus', 0) > 0)

        self.joint_noise_clean_speech_training = joint_noise_clean_speech_training #$#

        self.vfeat_processing_order = kwargs["vfeat_processing_order"]
        
        self.audio_only = kwargs["audio_only"]

        # self.joint_noise_clean_speech_training = kwargs["joint_noise_clean_speech_training"]
        

        if self.audio_only:
            assert self.vfeat_processing_order == "default"


        if self.joint_noise_clean_speech_training:
            assert self.audio_only  
            assert self.vfeat_processing_order == "default"


        if not self.audio_only:
            assert not self.joint_noise_clean_speech_training
            assert "attn" in backbone or "flow_avse" in backbone

        self.backbone = backbone ## pay attention to the backbone to be used when using , 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err):
        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def _step(self, batch, batch_idx):
        
        if not self.joint_noise_clean_speech_training and self.audio_only :
            
            x= batch
            y = None


        else: #  self.audio_only with joint_noise_clean_speech_training, or not self.audio_only  (audiovisual)
            x,y = batch

        
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, t)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z
        # theta = self.sde.theta
        # gamma_t = torch.exp(-theta * t)[:, None, None, None]
        # score = self(perturbed_data, t)
        # err = (sigmas ** 2 * score + perturbed_data) - gamma_t * x # original noise prediction loss
        # loss = self._loss(err)
    
        score = self(perturbed_data, t, y)
        err = score * sigmas + z # original noise prediction loss
        # err = score + z / sigmas # denoising score-matching objective
        loss = self._loss(err)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

        return loss

    def forward(self, x, t, y):

        """
        x: clean speech stft or noise stft 
        t: timesteps
        y: None if traning audio only model without jointly training on speech and noise, or
           Label if traning audio only model with jointly training on speech and noise, or 
           video if traning audio-visual model without jointly training on speech and noise. In this case, x should be clean speech
        """      

        _, std = self.sde.marginal_prob(x, t)
        sigmas = std[:, None, None, None]
        
        if self.audio_only and not self.joint_noise_clean_speech_training:
            dnn_input = x 
            score = -self.dnn(dnn_input, t)/sigmas
        
        elif self.audio_only and self.joint_noise_clean_speech_training:
            dnn_input = [x, y]  ## y is the label
            score = -self.dnn(dnn_input, t)/sigmas
        
        elif not self.audio_only and self.vfeat_processing_order == "cut_extract":
            dnn_input = x 
            score = -self.dnn(dnn_input, t, y)/sigmas ## y is video actually

        else:
            raise NotImplementedError(f"forward not implemented when audio_only audio_only is {self.audio_only} and vfeat_processing_order is {self.vfeat_processing_order} and joint_noise_clean_speech_training {self.joint_noise_clean_speech_training}")
      
        return score



    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)


    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)


