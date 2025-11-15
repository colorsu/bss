import torch
import numpy as np
import soundfile as sf


def load_audio_sf(path, n_channels=None, seconds=None):
    """Load an audio file using soundfile and return a PyTorch tensor.

    Args:
        path (str): Path to the audio file.
        n_channels (int, optional): Number of channels to keep. If given and the
            file has more channels, it will be truncated; if it has fewer,
            a :class:`ValueError` is raised.
        seconds (float, optional): Duration in seconds to load. If given, only
            the first `seconds` of audio will be loaded.

    Returns:
        tuple[torch.Tensor, int]: ``(audio, sample_rate)`` where ``audio`` has
        shape ``[channels, time]`` or ``[time]``.
    """

    # Determine how many frames to read
    if seconds is not None:
        info = sf.info(path)
        frames = int(seconds * info.samplerate)
        data, samplerate = sf.read(path, frames=frames, dtype="float32")
    else:
        data, samplerate = sf.read(path, dtype="float32")

    tensor = torch.from_numpy(data)

    # Soundfile: [time, channels]; we prefer [channels, time]
    if tensor.ndim == 2:
        tensor = tensor.t()

    if n_channels is not None:
        if tensor.ndim == 1:
            # Mono but multi-channel requested
            if n_channels != 1:
                raise ValueError(
                    f"Input audio has 1 channel, but {n_channels} were requested."
                )
        else:
            if tensor.shape[0] < n_channels:
                raise ValueError(
                    f"Input audio has {tensor.shape[0]} channels, but {n_channels} were requested."
                )
            elif tensor.shape[0] > n_channels:
                tensor = tensor[:n_channels, :]

    return tensor, samplerate


def save_audio_sf(path, tensor, samplerate):
    """Save a PyTorch tensor to an audio file using soundfile.

    Args:
        path (str): Output file path.
        tensor (torch.Tensor): Audio tensor with shape ``[channels, time]`` or
            ``[time]``.
        samplerate (int): Sampling rate in Hz.
    """

    data = tensor.detach().cpu().numpy()

    if data.ndim == 2:
        # Convert back to [time, channels]
        data = data.T

    sf.write(path, data, samplerate)


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
