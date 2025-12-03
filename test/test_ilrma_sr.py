import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from src.audio import STFT, load_audio_sf, save_audio_sf
from src.bss import ILRMA, ILRMA_V2

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

        # mix_out_spec = self.ilrma(mix_spec)
        # mix_out = self.stft.inverse(mix_out_spec)

        in_spec = torch.view_as_complex(mix_spec).permute(1, 2, 0)
        bf_spec = torch.einsum('TFC,FC->TF', in_spec, torch.from_numpy(sv.conj()).to(torch.complex64))
        bf_spec = bf_spec / mix.shape[0]  # Normalize by number of channels
        mix_out = self.stft.inverse(torch.view_as_real(bf_spec).unsqueeze(0))
        return mix_out

def calculate_steering_vector(mic_positions, source_angle_deg, n_fft=512, fs=16000):
    c = 343  # Speed of sound in m/s
    freqs = np.linspace(0, fs/2, n_fft//2 + 1)
    # standard mathematical angle (0 is East, 90 is North)
    azimuth = np.deg2rad(source_angle_deg)
    doa = np.array([np.cos(azimuth), np.sin(azimuth), 0])  # 2D plane, z=0

    center = np.mean(mic_positions, axis=0)
    centered_mics = mic_positions - center  # Center the array

    delays = -np.dot(centered_mics, doa) / c
    # exp(-j * 2*pi * f * tau)
    steering_vector = np.exp(-1j * 2 *np.pi * freqs[:, None] * delays[None, :])

    return steering_vector

def generate_mixture(source, source_angle=0, interference=None, interference_angle=60, sr=16000):
    room_dim = [6, 5, 3]
    array_center = np.array([3, 2.5, 1.5])
    mic_distance = 0.02
    dist_from_array = 1.5

    room = pra.ShoeBox(room_dim, fs=16000, max_order=12, absorption=0.2)

    mic_pos = np.c_[
        array_center + np.array([-mic_distance / 2, 0, 0]),
        array_center + np.array([mic_distance / 2, 0, 0])
    ].T

    mic_array = pra.MicrophoneArray(mic_pos.T, room.fs)
    room.add_microphone_array(mic_array)

    source_loc = array_center + [
        dist_from_array * np.cos(np.deg2rad(source_angle)),
        dist_from_array * np.sin(np.deg2rad(source_angle)),
        0
    ]
    room.add_source(source_loc, signal=source)

    if interference is not None:
        interference_loc = array_center + [
            dist_from_array * np.cos(np.deg2rad(interference_angle)),
            dist_from_array * np.sin(np.deg2rad(interference_angle)),
            0
        ]
        room.add_source(interference_loc, signal=interference)

    room.simulate()
    mix = room.mic_array.signals.T  # (n_samples, n_channels)
    sv = calculate_steering_vector(mic_pos, source_angle, n_fft=512, fs=sr)
    return mix, sv

if __name__ == "__main__":
    # mix_fname = "/Users/kolor/myWork/data/地铁-0626.wav"
    # mix_fname = "../noise_test_2ch.wav"
    # mix_fname = "../train_low_snr.wav"
    # mix_fname = "../train_high_snr.wav"
    source_fname = "/Users/kolor/myWork/speech_process/data/didine.wav"
    interference_fname = "/Users/kolor/myWork/speech_process/data/interference.wav"

    # mix, sr = load_audio_sf(mix_fname, n_channels=2)
    # print(f"Loaded {mix.shape[1] / sr:.2f} seconds of audio at {sr} Hz")
    source = load_audio_sf(source_fname, n_channels=1)[0]
    interference = load_audio_sf(interference_fname, n_channels=1)[0]
    source_angle = 0
    interference_angle = 60
    sr = 16000
    mix, sv = generate_mixture(
        source, source_angle=source_angle, interference=interference, interference_angle=interference_angle,
        sr=sr
    )
    mix = torch.from_numpy(mix.T).float()

    save_audio_sf("generated_mix.wav", mix, sr)
    print(f"Steering vector: {sv}")

    model = ILRMA_GRAPH(frame_shift=256, n_iter=100, n_components=2, k_NMF_bases=8)

    fname = Path(source_fname).stem
    out_path = f"mix_out_{fname}_ILRMA_SR_.wav"
    print(f"Saving output to {out_path}")

    # # Need to run model again to get outp ut since profile returns dict
    with torch.no_grad():
        mix_out = model(mix)
    save_audio_sf(out_path, mix_out, sr)