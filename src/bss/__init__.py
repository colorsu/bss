"""Blind Source Separation (BSS) algorithms package.

This package provides implementations of various BSS algorithms including:
- IVA (Independent Vector Analysis) algorithms
- ILRMA (Independent Low-Rank Matrix Analysis) algorithms
- RCSCME algorithms

Usage:
    from bss import ILRMA, AUX_IVA_ISS, get_bss, list_algorithms

    # Direct instantiation
    algo = ILRMA(n_components=2, n_iter=100)

    # Or use registry
    algo = get_bss("ILRMA", n_components=2, n_iter=100)
"""

# Base class and registry
from .base import BSSBase
from .registry import get_bss, list_algorithms, get_algorithm_info, register_bss

# IVA algorithms
from .iva import (
    IVA_NG,
    AUX_IVA_ISS,
    AUX_IVA_ISS_ONLINE,
    AUX_OVER_IVA,
    AUX_OVER_IVA_ONLINE,
)

# ILRMA algorithms
from .ilrma import (
    ILRMA,
    ILRMA_V2,
    ILRMA_SR,
    ILRMA_REALTIME,
    ILRMA_NOISY,
)

# RCSCME algorithms
from .rcscme import RCSCME

# Utilities
from .utils import nmf_update, select_target_index, contrast_weights

__all__ = [
    # Base
    "BSSBase",
    # Registry
    "get_bss",
    "list_algorithms",
    "get_algorithm_info",
    "register_bss",
    # IVA
    "IVA_NG",
    "AUX_IVA_ISS",
    "AUX_IVA_ISS_ONLINE",
    "AUX_OVER_IVA",
    "AUX_OVER_IVA_ONLINE",
    # ILRMA
    "ILRMA",
    "ILRMA_V2",
    "ILRMA_SR",
    "ILRMA_REALTIME",
    "ILRMA_NOISY",
    # RCSCME
    "RCSCME",
    # Utils
    "nmf_update",
    "select_target_index",
    "contrast_weights",
]
