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


def select_target_index(Y, sr=16000, n_fft=512):
    """Select the target speech index based on robust kurtosis in speech frequency band.

    The method calculates the kurtosis of the separated signals' amplitude envelope
    specifically within the speech frequency range (e.g., 300Hz - 3500Hz).
    This avoids low-frequency DC components and high-frequency artifacts/music noise.

    Args:
        Y: Separated spectrograms with shape (N, J, I, 2) (Sources, Time, Freq, Real/Imag).
        sr (int): Sampling rate in Hz. Default: 16000.
        n_fft (int): FFT size. Default: 512.

    Returns:
        tuple: (target_index, Y_target)
            target_index (int): The index of the target source.
            Y_target (torch.Tensor): The spectrogram of the target source with shape (J, I, 2).
    """
    N, J, I, _ = Y.shape
    scores = []

    # 1. Define speech core frequency band indices (e.g., 300Hz - 3500Hz)
    # This is a crucial engineering trick to avoid low-freq DC and high-freq aliasing/artifacts.
    freq_res = sr / n_fft  # Frequency resolution
    low_idx = int(300 / freq_res)
    high_idx = int(3500 / freq_res)
    
    # Ensure indices are within valid range
    low_idx = max(0, low_idx)
    high_idx = min(I, high_idx)

    for n in range(N):
        y_n = Y[n, ...]  # (J, I, 2)

        # Calculate magnitude for each T-F bin: sqrt(real^2 + imag^2)
        mag = torch.sqrt(y_n[..., 0]**2 + y_n[..., 1]**2)  # (J, I)

        # 2. Use energy only from the speech frequency band
        # Avoids interference from high-frequency "musical noise"
        target_band_mag = mag[:, low_idx:high_idx]

        # Sum across frequency bins to get the amplitude envelope (similar to time domain)
        envelope = torch.sum(target_band_mag, dim=1)  # (J,)

        # Normalize to prevent volume level from affecting the judgment
        envelope = envelope - torch.mean(envelope)
        envelope = envelope / (torch.std(envelope) + 1e-10)

        # 3. Calculate metrics: Combine Kurtosis and Variance, or use Dynamic Range
        # Speech characteristics: mostly silence, occasional high peaks (sparse).

        # Calculate Kurtosis: E[x^4] / (E[x^2])^2 - 3
        m4 = torch.mean(envelope**4)
        m2 = torch.mean(envelope**2)
        kurt = m4 / (m2**2 + 1e-10) - 3

        # Calculate Variance (or Coefficient of Variation)
        # Speech envelopes typically fluctuate more than stationary noise
        variance = torch.var(envelope)

        # Comprehensive Score:
        # In practice, sometimes Variance is more stable than Kurtosis,
        # because Kurtosis can be fooled by extremely sharp artifacts.
        # We assume: higher fluctuation (Variance) and higher sparsity (Kurtosis) -> likely speech.
        
        # You can adjust weights based on experiments, or use just one.
        score = variance + 0.3 * kurt
        scores.append(score)

    # Select the source with the highest score
    target_index = torch.argmax(torch.stack(scores)).item()
    Y_target = Y[target_index, ...]

    return target_index, Y_target
