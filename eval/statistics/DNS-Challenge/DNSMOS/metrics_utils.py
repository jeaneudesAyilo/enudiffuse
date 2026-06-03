"""
Adapted from original code by Clarity Challenge
https://github.com/claritychallenge/clarity
"""

import os
import numpy as np
from pypesq import pesq
from pystoi import stoi


def compute_sisdr(reference, estimate):
    """Compute the scale invariant SDR.

    Parameters
    ----------
    estimate : array of float, shape (n_samples,)
        Estimated signal.
    reference : array of float, shape (n_samples,)
        Ground-truth reference signal.

    Returns
    -------
    sisdr : float
        SI-SDR.

    Example
    --------
    >>> import numpy as np
    >>> from sisdr_metric import compute_sisdr
    >>> np.random.seed(0)
    >>> reference = np.random.randn(16000)
    >>> estimate = np.random.randn(16000)
    >>> compute_sisdr(estimate, reference)
    -48.1027283264049
    """
    eps = np.finfo(estimate.dtype).eps
    alpha = (np.sum(estimate * reference) + eps) / (
        np.sum(np.abs(reference) ** 2) + eps
    )
    sisdr = 10 * np.log10(
        (np.sum(np.abs(alpha * reference) ** 2) + eps)
        / (np.sum(np.abs(alpha * reference - estimate) ** 2) + eps)
    )
    return sisdr


def si_sdr_components(s_hat, s, n):
    """
    """
    # s_target
    alpha_s = np.dot(s_hat, s) / np.linalg.norm(s)**2
    s_target = alpha_s * s

    # e_noise
    alpha_n = np.dot(s_hat, n) / np.linalg.norm(n)**2
    e_noise = alpha_n * n

    # e_art
    e_art = s_hat - s_target - e_noise
    
    return s_target, e_noise, e_art


def energy_ratios(s_hat, s, n):
    """
    """
    s_target, e_noise, e_art = si_sdr_components(s_hat, s, n)

    si_sdr = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_noise + e_art)**2)
    si_sir = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_noise)**2)
    si_sar = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_art)**2)

    return si_sdr, si_sir, si_sar


def compute_pesq(target, enhanced, sr):
    """Compute PESQ using PyPESQ
    Args:
        target (string): Name of file to read
        enhanced (string): Name of file to read
        sr (int): sample rate of files
    Returns:
        PESQ metric (float)
    """
    len_x = np.min([len(target), len(enhanced)])
    target = target[:len_x]
    enhanced = enhanced[:len_x]

    return pesq(target, enhanced, sr)


def compute_stoi(target, enhanced, sr):
    """Compute STOI from: https://github.com/mpariente/pystoi
    Args:
        target (string): Name of file to read
        enhanced (string): Name of file to read
        sr (int): sample rate of files
    Returns:
        STOI metric (float)
    """
    len_x = np.min([len(target), len(enhanced)])
    target = target[:len_x]
    enhanced = enhanced[:len_x]

    return stoi(target, enhanced, sr, extended=True)
