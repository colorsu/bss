import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from src.audio import STFT, load_audio_sf, save_audio_sf
from src.bss import RCSCME

class RCSCME_Runner(torch.nn.Module):
    """
    RCSCME runner for source separation.

    Args:
        n_iter (int): Number of iterations for RCSCME. Default: 30
    """

    def __init__(self, n_iter=30, frame_shift=256, n_components=2, k_NMF_bases=8, sr=16000):
        super().__init__()
        self.n_iter = n_iter
        self.frame_shift = frame_shift
        self.stft = STFT(win_len=frame_shift*2, shift_len=frame_shift)
        self.rcscme = RCSCME(n_components=n_components, k_NMF_bases=k_NMF_bases, n_iter=n_iter, sr=sr, n_fft=frame_shift*2)
        print(f"Initialized RCSCME_Runner with n_iter={n_iter}, frame_shift={frame_shift}, n_components={n_components}, k_NMF_bases={k_NMF_bases}, sr={sr}")

    def forward(self, mix):
        # mix: (channels, time)
        mix_spec = self.stft.transform(mix)
        print(f"Mix spectrogram shape: {mix_spec.shape}")

        # Y: (I, N, J), Y_target: (I, J)
        Y, Y_target = self.rcscme(mix_spec)
        
        Y_target_spec = Y_target.unsqueeze(0)
        
        target_audio = self.stft.inverse(Y_target_spec)

        return target_audio.squeeze(0)

if __name__ == "__main__":
    mix_fname = "/Users/kolor/myWork/data/地铁-0626.wav"
    # Check if file exists, if not use a dummy path or handle error gracefully
    if not os.path.exists(mix_fname):
        print(f"Warning: {mix_fname} not found. Please set the correct path.")
        # For testing purposes, we might want to exit or use a different file
        # sys.exit(1)


    mix, sr = load_audio_sf(mix_fname, n_channels=2)
    print(f"Loaded {mix.shape[1] / sr:.2f} seconds of audio at {sr} Hz")

    model = RCSCME_Runner(frame_shift=512, n_iter=50, n_components=2, k_NMF_bases=8, sr=sr)
    
    target_audio = model(mix)

    fname = os.path.basename(mix_fname).split('.')[0]
    out_path = f"mix_out_rcscme_{fname}.wav"
    print(f"Saving output to {out_path}")
    save_audio_sf(out_path, target_audio, sr)
    
