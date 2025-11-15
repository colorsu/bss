import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.audio import STFT, load_audio_sf, save_audio_sf
from src.bss import AUX_IVA_ISS


class AUX_IVA_GRAPH(torch.nn.Module):
    """AUX-IVA ISS computational graph for source separation.

    This mirrors ILRMA_GRAPH but uses AUX_IVA_ISS as the separator.
    """

    def __init__(self, n_iter=20, frame_shift=256, contrast_func="laplace"):
        super().__init__()
        self.n_iter = n_iter
        self.frame_shift = frame_shift
        self.stft = STFT(win_len=frame_shift * 2, shift_len=frame_shift)
        self.iva = AUX_IVA_ISS(n_iter=n_iter, contrast_func=contrast_func)
        print(
            f"Initialized AUX_IVA_GRAPH with n_iter={n_iter}, "
            f"frame_shift={frame_shift}, contrast_func={contrast_func}"
        )

    def forward(self, mix):
        # mix: (K, T)
        mix_spec = self.stft.transform(mix)
        # mix_spec: (K, T, F, 2)
        print(f"Mix spectrogram shape: {mix_spec.shape}")

        sep_spec = self.iva(mix_spec)
        sep = self.stft.inverse(sep_spec)
        return sep


if __name__ == "__main__":
    # For convenience, reuse the same mixture as ILRMA test if present
    mix_fname = "../train_high_snr.wav"

    mix, sr = load_audio_sf(mix_fname, n_channels=2)
    print(f"Loaded {mix.shape[1] / sr:.2f} seconds of audio at {sr} Hz")

    model = AUX_IVA_GRAPH(n_iter=30, frame_shift=256, contrast_func="gaussian")

    fname = mix_fname.split("/")[-1].split(".")[0]
    out_path = f"mix_out_iva_{fname}.wav"
    print(f"Saving output to {out_path}")

    # Need to run model again to get output since profile returns dict
    with torch.no_grad():
        mix_out = model(mix)
    save_audio_sf(out_path, mix_out, sr)
