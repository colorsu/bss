import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.audio import STFT, load_audio_sf, save_audio_sf
from src.bss import IVA_NG

class IVA_NG_GRAPH(torch.nn.Module):
    """
    IVA_NG computational graph for source separation.

    Args:
        n_iter (int): Number of iterations for IVA_NG. Default: 30
    """

    def __init__(self, n_iter=30, learning_rate=0.1, frame_shift=256, n_components=2, k_NMF_bases=8):
        super(IVA_NG_GRAPH, self).__init__()
        self.n_iter = n_iter
        self.frame_shift = frame_shift
        self.stft = STFT(win_len=self.frame_shift*2, shift_len=self.frame_shift)
        self.iva = IVA_NG(n_components=n_components, learning_rate=learning_rate, n_iter=n_iter, ref_mic=1)
        print(f"Initialized IVA_NG_GRAPH with n_iter={n_iter}, learning_rate={learning_rate}, frame_shift={self.frame_shift}, n_components={n_components}, k_NMF_bases={k_NMF_bases}")

    def forward(self, mix, n_src=1):
        mix_spec = self.stft.transform(mix)
        print(f"Mix spectrogram shape: {mix_spec.shape}")

        mix_out_spec = self.iva(mix_spec)
        mix_out = self.stft.inverse(mix_out_spec)

        return mix_out

if __name__ == "__main__":
    mix_fname = "/Users/kolor/myWork/data/地铁-0626.wav"
    # mix_fname = "/Users/kolor/myWork/bss/generated_mix.wav"
    # mix_fname = "../noise_test_2ch.wav"
    # mix_fname = "../train_low_snr.wav"
    # mix_fname = "../train_high_snr.wav"

    mix, sr = load_audio_sf(mix_fname, n_channels=2)
    print(f"Loaded {mix.shape[1] / sr:.2f} seconds of audio at {sr} Hz")

    model = IVA_NG_GRAPH(frame_shift=256, n_iter=200, learning_rate=0.1, n_components=2, k_NMF_bases=8)

    fname = mix_fname.split('/')[-1].split('.')[0]
    out_path = f"   {fname}_K8_C2_256.wav"
    print(f"Saving output to {out_path}")

    # Need to run model again to get outp ut since profile returns dict
    with torch.no_grad():
        mix_out = model(mix)
    save_audio_sf(out_path, mix_out, sr)