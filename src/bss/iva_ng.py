"""Independent Low-Rank Matrix Analysis (ILRMA) - Vectorized Version."""

import torch
from .utils import nmf_update


class IVA_NG(torch.nn.Module):
    """IVA implement based on Natural gradient.

    Args:
        n_components (int): Number of sources (must equal number of channels). Default: 2
        k_NMF_bases (int): Number of NMF basis functions. Default: 8
        n_iter (int): Number of iterations. Default: 30
        nmf_eps (float): Small constant for NMF stability. Default: 1e-20
        ip_eps (float): Small constant for IP update stability. Default: 1e-20
        ref_mic (int): Reference microphone for back-projection (1-indexed, None to disable). Default: None
    """

    def __init__(self, n_components=2, n_iter=30, learning_rate=0.1, ip_eps: float = 1e-20, ref_mic=None):
        super().__init__()
        self.n_components = n_components
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.ip_eps = ip_eps
        self.ref_mic = ref_mic

    def forward(self, X):
        # X: (M, J, I, 2) real/imag
        M, J, I, _ = X.shape
        N = self.n_components
        assert M == N, "IVA_NG assumes number of sources equals number of channels (M == N)."

        X_c = torch.view_as_complex(X.permute(2, 0, 1, 3))  # (I, M, J)

        # Initialize W_i as identity for each frequency bin
        W = torch.eye(N).repeat(I, 1, 1).to(X_c) # (I, N, M)

        eye_M = torch.eye(N, dtype=torch.complex64, device=X.device)

        for _ in range(self.n_iter):
            for n in range(N):
                # Separation
                Y = torch.matmul(W, X_c) # (I, N, N) x (I, N, J) -> (I, N, J)

                # "Glue"
                source_model = torch.sum(torch.abs(Y)**2, dim=0) # (N, J)

                # Contrast function
                inv_contrast = 1.0 / (torch.sqrt(source_model) + self.ip_eps)
                inv_contrast = inv_contrast.unsqueeze(0)  # (1, N, J)

                # score function
                phi = Y * inv_contrast # (I, N, J)
                Y_conj_T = Y.conj().permute(0, 2, 1) # (I, J, N)

                # covariance matrix
                cov = torch.matmul(phi, Y_conj_T) / J # (I, N, N)

                gradient_direction = eye_M.unsqueeze(0) - cov # (I, N, N)

                delta_W = torch.matmul(gradient_direction, W) # (I, N, N) x (I, N, M) -> (I, N, M)

                W = W + self.learning_rate * delta_W

        # Final output with latest W
        Y = torch.zeros_like(X_c)  # (I, N, J)
        for f in range(I):
            X_f = X_c[f, :, :]  # (N, J)
            Y[f] = W[f] @ X_f  # (N, N) x (N, J) -> (N, J)

        # Minimal-distortion back-projection (optional)
        if self.ref_mic is not None and self.ref_mic > 0 and self.ref_mic <= N:
            ref = self.ref_mic - 1  # convert 1-based to 0-based
            for f in range(I):
                # Af: estimate of mixing matrix = pinv(W_f)
                Af = torch.linalg.pinv(W[f])  # (N, N)
                scale = Af[ref, :]  # (N,)
                Y[f] = scale.unsqueeze(1) * Y[f]  # (N, 1) * (N, J) -> (N, J)

        # Return Y in original axis order: (M, J, I, 2)
        Y_out = Y.permute(1, 2, 0)  # (N, J, I)
        Y_out = torch.view_as_real(Y_out).contiguous()  # (N, J, I, 2)
        return Y_out
