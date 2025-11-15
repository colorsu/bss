import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.audio import STFT
from src.bss import ILRMA, ILRMA_V2
import soundfile as sf

class ILRMA_GRAPH(torch.nn.Module):
    """
    ILRMA computational graph for source separation.

    Args:
        n_iter (int): Number of iterations for ILRMA. Default: 30
    """

    def __init__(self, n_iter=30, frame_shift=256, n_components=2, k_NMF_bases=8):
        super(ILRMA_GRAPH, self).__init__()
        self.n_iter = n_iter
        self.frame_shift = frame_shift
        self.stft = STFT(win_len=frame_shift*2, shift_len=frame_shift)
        # self.ilrma = ILRMA(n_components=n_components, k_NMF_bases=k_NMF_bases, n_iter=n_iter)
        self.ilrma = ILRMA_V2(n_components=n_components, k_NMF_bases=k_NMF_bases, n_iter=n_iter)
        print(f"Initialized ILRMA_GRAPH with n_iter={n_iter}, frame_shift={frame_shift}, n_components={n_components}, k_NMF_bases={k_NMF_bases}")

    def forward(self, mix, n_src=1):
        mix_spec = self.stft.transform(mix)
        print(f"Mix spectrogram shape: {mix_spec.shape}")

        mix_out_spec = self.ilrma(mix_spec)
        mix_out = self.stft.inverse(mix_out_spec)

        return mix_out

def load_audio_sf(path, n_channels=None):
    # data is a numpy array, float64 or float32
    # samplerate is an int
    data, samplerate = sf.read(path, dtype='float32')

    # Soundfile usually returns shape [Time, Channels]
    # PyTorch typically uses [Batch, Channels, Time] or [Channels, Time] for signals
    # We need to transpose here
    tensor = torch.from_numpy(data)

    if tensor.ndim == 2:
        tensor = tensor.t() # transpose to [Channels, Time]

    if n_channels is not None:
        if tensor.shape[0] < n_channels:
            raise ValueError(f"Input audio has {tensor.shape[0]} channels, but {n_channels} were requested.")
        elif tensor.shape[0] > n_channels:
            tensor = tensor[:n_channels, :]

    return tensor, samplerate

def save_audio_sf(path, tensor, samplerate):
    # tensor shape [Channels, Time] or [Time]
    data = tensor.cpu().numpy()

    if data.ndim == 2:
        data = data.T  # transpose to [Time, Channels]

    sf.write(path, data, samplerate)

if __name__ == "__main__":
    # mix_fname = "/Users/kolor/myWork/data/地铁-0626.wav"
    # mix_fname = "../noise_test_2ch.wav"
    # mix_fname = "../train_low_snr.wav"
    mix_fname = "../train_high_snr.wav"
    mix, sr = load_audio_sf(mix_fname, n_channels=2)
    model = ILRMA_GRAPH(frame_shift=256, n_iter=50, n_components=2, k_NMF_bases=32)

    mix_out = model(mix)

    fname = mix_fname.split('/')[-1].split('.')[0]
    out_path = f"mix_out_{fname}_K32_C2_V2_256.wav"
    print(f"Saving output to {out_path}")
    save_audio_sf(out_path, mix_out, sr)