"""
Test script for STFT utility.

This script tests the STFT transform, inverse transform, and reconstruction quality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.audio import STFT


def test_stft_reconstruction():
    """Test STFT forward and inverse transform for reconstruction accuracy."""
    print("=" * 60)
    print("Test 1: STFT Reconstruction")
    print("=" * 60)

    # Create synthetic signal
    sample_rate = 16000
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Synthesize a signal with multiple frequencies
    freq1, freq2, freq3 = 440, 880, 1320  # A4, A5, E6 notes
    signal = (np.sin(2 * np.pi * freq1 * t) +
              0.5 * np.sin(2 * np.pi * freq2 * t) +
              0.3 * np.sin(2 * np.pi * freq3 * t))

    # Convert to torch tensor with batch dimension
    signal_torch = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0)

    print(f"Input signal shape: {signal_torch.shape}")
    print(f"Signal range: [{signal_torch.min():.4f}, {signal_torch.max():.4f}]")

    # Initialize STFT
    stft = STFT(win_len=1024, shift_len=512)

    # Transform
    spec = stft.transform(signal_torch)
    print(f"STFT output shape: {spec.shape}")
    print(f"Spectrogram range: [{spec.min():.4f}, {spec.max():.4f}]")

    # # Plot spectrogram
    # magnitude = torch.sqrt(spec[0, :, :, 0]**2 + spec[0, :, :, 1]**2)  # Compute magnitude
    # magnitude_db = 20 * torch.log10(magnitude + 1e-8)  # Convert to dB scale

    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(magnitude_db.T.numpy(), aspect='auto', origin='lower', cmap='viridis')
    # plt.colorbar(label='Magnitude (dB)')
    # plt.xlabel('Time Frame')
    # plt.ylabel('Frequency Bin')
    # plt.title('STFT Spectrogram (Magnitude)')

    # # Plot phase
    # phase = torch.atan2(spec[0, :, :, 1], spec[0, :, :, 0])
    # plt.subplot(1, 2, 2)
    # plt.imshow(phase.T.numpy(), aspect='auto', origin='lower', cmap='hsv')
    # plt.colorbar(label='Phase (radians)')
    # plt.xlabel('Time Frame')
    # plt.ylabel('Frequency Bin')
    # plt.title('STFT Spectrogram (Phase)')

    # plt.tight_layout()
    # plt.savefig('/home/luoruihong/hdd_r2/myWork/bss/test/stft_spectrogram.png', dpi=150)
    # print("Spectrogram saved to: test/stft_spectrogram.png")
    # plt.close()

    # Inverse transform
    reconstructed = stft.inverse(spec)
    print(f"Reconstructed signal shape: {reconstructed.shape}")
    print(f"Reconstructed range: [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")

    # Calculate reconstruction error
    # Trim to same length
    min_len = min(signal_torch.shape[-1], reconstructed.shape[-1])
    original_trimmed = signal_torch[..., :min_len]
    reconstructed_trimmed = reconstructed[..., :min_len]

    mse = torch.mean((original_trimmed - reconstructed_trimmed) ** 2).item()
    snr = 10 * np.log10(torch.mean(original_trimmed ** 2).item() / (mse + 1e-10))

    print(f"\nReconstruction Quality:")
    print(f"  MSE: {mse:.6f}")
    print(f"  SNR: {snr:.2f} dB")

    assert snr > 50, f"Poor reconstruction: SNR = {snr:.2f} dB"
    print("✓ Test passed: Good reconstruction quality")
    print()


def test_stft_forward():
    """Test STFT forward pass (transform + inverse in one call)."""
    print("=" * 60)
    print("Test 2: STFT Forward Pass")
    print("=" * 60)

    # Create a simple sine wave
    sample_rate = 16000
    duration = 1.0
    freq = 1000  # 1 kHz
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * freq * t)
    signal_torch = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0)

    print(f"Input signal shape: {signal_torch.shape}")

    # Initialize and run forward pass
    stft = STFT(win_len=512, shift_len=256)
    output = stft(signal_torch)

    print(f"Output signal shape: {output.shape}")

    # Check reconstruction
    min_len = min(signal_torch.shape[-1], output.shape[-1])
    mse = torch.mean((signal_torch[..., :min_len] - output[..., :min_len]) ** 2).item()
    snr = 10 * np.log10(torch.mean(signal_torch[..., :min_len] ** 2).item() / (mse + 1e-10))

    print(f"SNR: {snr:.2f} dB")

    assert snr > 50, f"Poor reconstruction in forward pass: SNR = {snr:.2f} dB"
    print("✓ Test passed: Forward pass works correctly")
    print()


def test_batch_processing():
    """Test STFT with batched input."""
    print("=" * 60)
    print("Test 3: Batch Processing")
    print("=" * 60)

    batch_size = 4
    signal_length = 16000

    # Create batch of random signals
    signals = torch.randn(batch_size, signal_length)
    print(f"Batch input shape: {signals.shape}")

    # Initialize STFT
    stft = STFT()

    # Transform
    spec = stft.transform(signals)
    print(f"Batch STFT shape: {spec.shape}")

    # Inverse
    reconstructed = stft.inverse(spec)
    print(f"Batch reconstructed shape: {reconstructed.shape}")

    # Check each item in batch
    min_len = min(signals.shape[-1], reconstructed.shape[-1])
    for i in range(batch_size):
        mse = torch.mean((signals[i, :min_len] - reconstructed[i, :min_len]) ** 2).item()
        snr = 10 * np.log10(torch.mean(signals[i, :min_len] ** 2).item() / (mse + 1e-10))
        assert snr > 50, f"Poor reconstruction for batch item {i}: SNR = {snr:.2f} dB"

    print("✓ Test passed: Batch processing works correctly")
    print()


def test_different_parameters():
    """Test STFT with different window and hop sizes."""
    print("=" * 60)
    print("Test 4: Different Parameters")
    print("=" * 60)

    signal = torch.randn(1, 8000)

    configs = [
        (512, 256),
        (1024, 512),
        (2048, 1024),
        (1024, 256),  # 75% overlap
    ]

    for win_len, shift_len in configs:
        print(f"Testing win_len={win_len}, shift_len={shift_len}")
        stft = STFT(win_len=win_len, shift_len=shift_len)

        spec = stft.transform(signal)
        reconstructed = stft.inverse(spec)

        min_len = min(signal.shape[-1], reconstructed.shape[-1])
        mse = torch.mean((signal[..., :min_len] - reconstructed[..., :min_len]) ** 2).item()
        snr = 10 * np.log10(torch.mean(signal[..., :min_len] ** 2).item() / (mse + 1e-10))

        print(f"  Spec shape: {spec.shape}, SNR: {snr:.2f} dB")
        assert snr > 50, f"Poor reconstruction: SNR = {snr:.2f} dB"

    print("✓ Test passed: All parameter configurations work correctly")
    print()


def test_cuda_support():
    """Test STFT with CUDA if available."""
    print("=" * 60)
    print("Test 5: CUDA Support")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        print()
        return

    signal = torch.randn(2, 16000)
    stft = STFT()

    # Move to GPU
    signal_gpu = signal.cuda()
    print(f"Input device: {signal_gpu.device}")

    # Transform on GPU
    spec_gpu = stft.transform(signal_gpu)
    print(f"STFT device: {spec_gpu.device}")
    assert spec_gpu.is_cuda, "STFT output should be on CUDA"

    # Inverse on GPU
    reconstructed_gpu = stft.inverse(spec_gpu)
    print(f"Reconstructed device: {reconstructed_gpu.device}")
    assert reconstructed_gpu.is_cuda, "Reconstructed output should be on CUDA"

    # Check reconstruction quality
    min_len = min(signal_gpu.shape[-1], reconstructed_gpu.shape[-1])
    mse = torch.mean((signal_gpu[..., :min_len] - reconstructed_gpu[..., :min_len]) ** 2).item()
    snr = 10 * np.log10(torch.mean(signal_gpu[..., :min_len] ** 2).item() / (mse + 1e-10))

    print(f"GPU SNR: {snr:.2f} dB")
    assert snr > 50, f"Poor reconstruction on GPU: SNR = {snr:.2f} dB"

    print("✓ Test passed: CUDA support works correctly")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STFT Utility Test Suite")
    print("=" * 60 + "\n")

    try:
        test_stft_reconstruction()
        test_stft_forward()
        test_batch_processing()
        test_different_parameters()
        test_cuda_support()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
