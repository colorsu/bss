import torch
import numpy as np

def spatial_covariance_matrix(X):
    # X: (M, J, I, 2) real/imag
    M, J, I, _ = X.shape
    X_c = torch.view_as_complex(X.permute(2, 0, 1, 3))  # (I, M, J)
    X_H = X_c.permute(0, 2, 1).conj() # (I, J, M)
    # (I, M, J) @ (I, J, M) -> (I, M, M)
    Rxx = torch.matmul(X_c, X_H) / J # (I, M, M)
    return Rxx

def steering_vector_eigen_decomp(Rxx, ref_mic=0, eps=1e-8):
    """Compute steering vector using eigenvalue decomposition.

    Args:
        Rxx (torch.Tensor): Spatial covariance matrix of shape (I, M, M), where
            I is number of frequency bins and M is number of channels.
        ref_mic (int): Index of the reference microphone to normalize against.
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Steering vector of shape (I, M).
    """
    # Eigenvalue decomposition
    eigvals, eigvecs = torch.linalg.eigh(Rxx)

    # Find the index of the largest eigenvalue
    steering_vector = eigvecs[..., -1] # (I, M)

    # Corresponding eigenvector
    ref_channel = steering_vector[:, ref_mic].unsqueeze(1) # (I, 1)

    # Normalize the steering vector
    steering_vector = steering_vector / (ref_channel + eps)

    return steering_vector