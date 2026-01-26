"""Base class for Blind Source Separation algorithms."""

from abc import ABC, abstractmethod
from typing import Literal

import torch


class BSSBase(torch.nn.Module, ABC):
    """Abstract base class for all BSS algorithms.

    All BSS algorithms should inherit from this class and implement:
    - forward(): The main separation method
    - is_online: Property indicating if the algorithm is online/streaming
    - algorithm_family: Property indicating the algorithm family

    Args:
        ref_mic (int): Reference microphone for back-projection (1-indexed, 0 for none).
    """

    def __init__(self, ref_mic: int = 0):
        super().__init__()
        self._ref_mic = ref_mic

    @property
    @abstractmethod
    def is_online(self) -> bool:
        """Whether this algorithm processes data online (frame-by-frame)."""
        pass

    @property
    @abstractmethod
    def algorithm_family(self) -> Literal["iva", "ilrma", "rcscme"]:
        """The algorithm family this belongs to."""
        pass

    @property
    def ref_mic(self) -> int:
        """Reference microphone index (1-indexed, 0 for none)."""
        return self._ref_mic

    @ref_mic.setter
    def ref_mic(self, value: int):
        self._ref_mic = value

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Perform blind source separation.

        Args:
            X: Mixture STFT, shape (M, T, F, 2) where last dim is real/imag.
                M = number of microphones/channels
                T = number of time frames
                F = number of frequency bins

        Returns:
            Y: Separated STFT, shape (N, T, F, 2) where N is number of sources.
        """
        pass

    def reset(self) -> None:
        """Reset internal state (important for online algorithms).

        Override this method in online algorithms to clear any accumulated
        state like covariance matrices, demixing matrices, etc.
        """
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"family={self.algorithm_family}, "
            f"online={self.is_online}, "
            f"ref_mic={self.ref_mic})"
        )
