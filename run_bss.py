#!/usr/bin/env python3
"""Hydra-based BSS entry point.

Usage:
    python run_bss.py audio.input_file=/path/to/audio.wav algorithm=ilrma_v2
    python run_bss.py algorithm=aux_iva_iss algorithm.params.n_iter=50

Examples:
    # List available algorithms
    python run_bss.py --help

    # Show resolved config
    python run_bss.py algorithm=ilrma --cfg job

    # Override parameters
    python run_bss.py audio.input_file=test.wav algorithm=aux_iva_iss_online \\
                      algorithm.params.alpha=0.99
"""

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import soundfile as sf
import soxr

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bss import get_bss, list_algorithms


def load_audio(cfg: DictConfig) -> tuple[torch.Tensor, int]:
    """Load and preprocess audio file."""
    if cfg.audio.input_file is None:
        raise ValueError("audio.input_file must be specified")
    
    audio_path = Path(cfg.audio.input_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Use soundfile to load audio
    data, sr = sf.read(str(audio_path))
    # Convert to torch tensor and ensure (channels, samples) format
    waveform = torch.from_numpy(data).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T  # soundfile returns (samples, channels), we need (channels, samples)
    
    # Resample if needed using soxr
    if sr != cfg.audio.sample_rate:
        # soxr expects (samples, channels) format
        data_resampled = soxr.resample(waveform.numpy().T, sr, cfg.audio.sample_rate)
        waveform = torch.from_numpy(data_resampled.T).float()
        sr = cfg.audio.sample_rate
    
    # Limit channels if specified
    if cfg.audio.n_channels > 0 and waveform.shape[0] > cfg.audio.n_channels:
        waveform = waveform[:cfg.audio.n_channels]
    
    return waveform, sr


def compute_stft(waveform: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    """Compute STFT and return in BSS format (M, T, F, 2)."""
    from audio import STFT
    stft = STFT(win_len=cfg.audio.n_fft, shift_len=cfg.audio.hop_length)
    # waveform: (M, samples) -> STFT expects (batch, samples)
    X = stft.transform(waveform)  # returns (M, T, F, 2)
    return X


def compute_istft(Y: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    """Compute inverse STFT from BSS format (N, T, F, 2)."""
    from audio import STFT
    stft = STFT(win_len=cfg.audio.n_fft, shift_len=cfg.audio.hop_length)
    # Y: (N, T, F, 2) -> inverse returns (N, samples)
    waveform = stft.inverse(Y)
    return waveform


def save_output(waveform: torch.Tensor, sr: int, cfg: DictConfig, input_path: Path):
    """Save separated audio as a single multi-channel file."""
    from audio import save_audio_sf
    
    output_dir = Path(cfg.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stem = input_path.stem
    algo_name = cfg.algorithm.name
    postfix = cfg.output.output_postfix
    
    filename = f"{stem}_{algo_name}"
    if postfix:
        filename += f"_{postfix}"
    filename += ".wav"
    
    output_path = output_dir / filename
    # Save all channels in one file: waveform is (N, samples)
    save_audio_sf(str(output_path), waveform, sr)
    print(f"Saved: {output_path}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for BSS processing."""
    print("=" * 60)
    print("BSS - Blind Source Separation")
    print("=" * 60)
    
    # Print available algorithms
    print(f"Available algorithms: {list_algorithms()}")
    print(f"Selected algorithm: {cfg.algorithm.name}")
    print(f"Parameters: {OmegaConf.to_yaml(cfg.algorithm.params)}")
    
    # Check if input file is provided
    if cfg.audio.input_file is None:
        print("\nNo input file specified. Use: audio.input_file=/path/to/audio.wav")
        print("Example: python run_bss.py audio.input_file=test.wav algorithm=ilrma_v2")
        return
    
    # Load audio
    print(f"\nLoading: {cfg.audio.input_file}")
    waveform, sr = load_audio(cfg)
    print(f"Audio shape: {waveform.shape}, Sample rate: {sr}")
    
    # Compute STFT
    print("\nComputing STFT...")
    X = compute_stft(waveform, cfg)
    print(f"STFT shape: {X.shape}")  # (M, T, F, 2)
    
    # Get algorithm
    algo_params = OmegaConf.to_container(cfg.algorithm.params, resolve=True)
    algorithm = get_bss(cfg.algorithm.name, **algo_params)
    print(f"\nRunning {cfg.algorithm.name}...")
    
    # Run separation
    with torch.no_grad():
        Y = algorithm(X)
    print(f"Output shape: {Y.shape}")  # (N, T, F, 2)
    
    # Compute inverse STFT
    print("\nComputing inverse STFT...")
    separated = compute_istft(Y, cfg)
    print(f"Separated waveform shape: {separated.shape}")
    
    # Save output
    if cfg.output.save_audio:
        save_output(separated, sr, cfg, Path(cfg.audio.input_file))
    
    print("\nDone!")


if __name__ == "__main__":
    main()
