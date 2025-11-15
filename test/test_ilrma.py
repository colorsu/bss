import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.audio import STFT
from src.bss import ILRMA
import soundfile as sf

class ILRMA_GRAPH(torch.nn.Module):
    """
    ILRMA computational graph for source separation.

    Args:
        n_iter (int): Number of iterations for ILRMA. Default: 30
    """

    def __init__(self, n_iter=30, frame_shift=256):
        super(ILRMA_GRAPH, self).__init__()
        self.n_iter = n_iter
        self.frame_shift = frame_shift
        self.stft = STFT(win_len=512, shift_len=frame_shift)
        self.ilrma = ILRMA(n_components=2, k_NMF_bases=8, n_iter=n_iter)

    def forward(self, mix, n_src=1):
        mix_spec = self.stft.transform(mix)
        print(f"Mix spectrogram shape: {mix_spec.shape}")

        mix_out_spec = self.ilrma(mix_spec)
        mix_out = self.stft.inverse(mix_out_spec[0, :, :, :].unsqueeze(0))

        return mix_out

def load_audio_sf(path):
    # data 是 numpy array, float64 或 float32
    # samplerate 是 int
    data, samplerate = sf.read(path, dtype='float32')

    # Soundfile 读出来通常是 [Time, Channels]
    # PyTorch 处理信号通常习惯 [Batch, Channels, Time] 或 [Channels, Time]
    # 这里需要转置一下
    tensor = torch.from_numpy(data)

    if tensor.ndim == 2:
        tensor = tensor.t() # 转置为 [Channels, Time]

    return tensor, samplerate

def save_audio_sf(path, tensor, samplerate):
    # tensor 形状 [Channels, Time] 或 [Time]
    data = tensor.cpu().numpy()

    if data.ndim == 2:
        data = data.T  # 转置为 [Time, Channels]

    sf.write(path, data, samplerate)

if __name__ == "__main__":
    mix_fname = "/Users/kolor/myWork/data/地铁-0626.wav"
    mix, sr = load_audio_sf(mix_fname)
    model = ILRMA_GRAPH(n_iter=30)

    mix_out = model(mix)

    save_audio_sf("mix_out.wav", mix_out[0], sr)