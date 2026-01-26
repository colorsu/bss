"""Algorithm registry for BSS algorithms.

Provides a decorator-based registration system for dynamic algorithm selection.
"""

from typing import Any, Dict, List, Type

from .base import BSSBase

# Global registry
_BSS_REGISTRY: Dict[str, Type[BSSBase]] = {}


def register_bss(name: str):
    """Decorator to register a BSS algorithm.

    Usage:
        @register_bss("ILRMA")
        class ILRMA(BSSBase):
            ...

    Args:
        name: Unique name for the algorithm (used in configs).
    """
    def decorator(cls: Type[BSSBase]) -> Type[BSSBase]:
        if name in _BSS_REGISTRY:
            raise ValueError(f"Algorithm '{name}' is already registered")
        _BSS_REGISTRY[name] = cls
        return cls
    return decorator


def get_bss(name: str, **kwargs) -> BSSBase:
    """Get a BSS algorithm instance by name.

    Args:
        name: Registered algorithm name.
        **kwargs: Arguments to pass to the algorithm constructor.

    Returns:
        Instantiated algorithm.

    Raises:
        KeyError: If algorithm is not registered.
    """
    if name not in _BSS_REGISTRY:
        available = ", ".join(_BSS_REGISTRY.keys())
        raise KeyError(f"Algorithm '{name}' not found. Available: {available}")
    return _BSS_REGISTRY[name](**kwargs)


def list_algorithms() -> List[str]:
    """List all registered algorithm names."""
    return list(_BSS_REGISTRY.keys())


def get_algorithm_info() -> Dict[str, Dict[str, Any]]:
    """Get detailed info about all registered algorithms.

    Returns:
        Dict mapping algorithm name to info dict with keys:
        - 'class': The algorithm class
        - 'family': Algorithm family ('iva', 'ilrma', etc.)
        - 'online': Whether it's an online algorithm
    """
    info = {}
    for name, cls in _BSS_REGISTRY.items():
        # Create a dummy instance to get properties (if possible)
        try:
            # Try to get class attributes if they're defined as class-level
            info[name] = {
                "class": cls,
                "module": cls.__module__,
            }
        except Exception:
            info[name] = {"class": cls}
    return info
