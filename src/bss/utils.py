"""Shared utilities for BSS algorithms."""

import torch


def nmf_update(Tn, Vn, Y_n, eps: float = 1e-20):
    """Shared NMF update used by ILRMA variants.

    Args:
        Tn: (I, K) nonnegative NMF basis matrix.
        Vn: (K, J) nonnegative NMF activation matrix.
        Y_n: (I, J) complex or real spectrogram for source n.
        eps: small constant for numerical stability.

    Returns:
        Tn_new, Vn_new, Rn_new where Rn_new = Tn_new @ Vn_new (I, J).
    """
    Rn = Tn @ Vn  # (I, J)
    mag_R = Rn.abs()
    Y_d_R = Y_n.abs() ** 2 / (mag_R ** 2 + eps)
    Rn_d = 1.0 / (mag_R + eps)

    Tn_num = Y_d_R @ Vn.T
    Tn_den = Rn_d @ Vn.T + eps
    Tn_new = Tn * torch.sqrt(Tn_num / Tn_den)

    Rn = Tn_new @ Vn
    mag_R_new = Rn.abs()
    Rn_d_new = 1.0 / (mag_R_new + eps)

    Vn_num = Tn_new.T @ Y_d_R
    Vn_den = Tn_new.T @ Rn_d_new + eps
    Vn_new = Vn * torch.sqrt(Vn_num / Vn_den)

    Rn_new = Tn_new @ Vn_new
    return Tn_new, Vn_new, Rn_new
