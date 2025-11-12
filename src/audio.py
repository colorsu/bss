"""
STFT (Short-Time Fourier Transform) utility module.

This module provides a PyTorch-based STFT implementation for audio processing.
"""

import torch
import numpy as np


class STFT(torch.nn.Module):
    """
    Short-Time Fourier Transform (STFT) module.

    Args:
        win_len (int): Window length for STFT. Default: 1024
        shift_len (int): Hop length (shift) for STFT. Default: 512
        window (torch.Tensor, optional): Window function. If None, uses Hanning window.
    """

    def __init__(self, win_len=1024, shift_len=512, window=None):
        super(STFT, self).__init__()
        if window is None:
            window = torch.from_numpy(np.sqrt(np.hanning(win_len).astype(np.float32)))
        self.win_len = win_len
        self.shift_len = shift_len
        self.window = window

    def transform(self, input_data):
        """
        Compute STFT of input signal.

        Args:
            input_data (torch.Tensor): Input audio signal

        Returns:
            torch.Tensor: STFT spectrogram with shape [batch, time, freq, 2]
        """
        self.window = self.window.to(input_data.device)
        spec = torch.stft(input_data, n_fft=self.win_len, hop_length=self.shift_len, win_length=self.win_len,
                          window=self.window, center=True, pad_mode='constant', return_complex=True)
        spec = torch.view_as_real(spec)
        return spec.permute(0, 2, 1, 3)

    def inverse(self, spec):
        """
        Compute inverse STFT to reconstruct time-domain signal.

        Args:
            spec (torch.Tensor): STFT spectrogram with shape [batch, time, freq, 2]

        Returns:
            torch.Tensor: Reconstructed audio signal
        """
        self.window = self.window.to(spec.device)
        torch_wav = torch.istft(torch.view_as_complex(torch.permute(spec, [0, 2, 1, 3]).contiguous()), n_fft=self.win_len,
                                hop_length=self.shift_len, win_length=self.win_len, window=self.window, center=True)
        return torch_wav

    def forward(self, input_data):
        """
        Forward pass: Transform and inverse transform (for testing reconstruction).

        Args:
            input_data (torch.Tensor): Input audio signal

        Returns:
            torch.Tensor: Reconstructed audio signal
        """
        stft_res = self.transform(input_data)
        reconstruction = self.inverse(stft_res)
        return reconstruction
