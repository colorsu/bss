"""Independent Low-Rank Matrix Analysis with Spatial Regularization (ILRMA-SR)."""

import torch

from ..base import BSSBase
from ..registry import register_bss
from ..utils import nmf_update


@register_bss("ILRMA_SR")
class ILRMA_SR(torch.nn.Module):
    """ILRMA vectorized/batched implementation for improved performance.

    This version performs frequency-wise operations in a vectorized manner
    for better computational efficiency compared to the per-frequency loop in ILRMA.

    Args:
        n_components (int): Number of sources (must equal number of channels). Default: 2
        k_NMF_bases (int): Number of NMF basis functions. Default: 8
        n_iter (int): Number of iterations. Default: 30
        nmf_eps (float): Small constant for NMF stability. Default: 1e-20
        ip_eps (float): Small constant for IP update stability. Default: 1e-20
    """

    def __init__(self, n_components=2, k_NMF_bases=8, n_iter=30,
                 nmf_eps: float = 1e-20, ip_eps: float = 1e-20):
        super().__init__()
        self.n_components = n_components
        self.k_NMF_bases = k_NMF_bases
        self.n_iter = n_iter
        self.nmf_eps = nmf_eps
        self.ip_eps = ip_eps

    def forward(self, X, sv):
        # X: (M, J, I, 2) real/imag
        M, J, I, _ = X.shape
        N = self.n_components
        K = self.k_NMF_bases
        assert M == N, "ILRMA_V2 assumes number of sources equals number of channels (M == N)."

        X_c = torch.view_as_complex(X.permute(2, 0, 1, 3))  # (I, M, J)

        # Initialize W_i as identity for each frequency bin
        W = torch.stack([
            torch.eye(N, M, dtype=torch.complex64, device=X.device)
            for _ in range(I)
        ], dim=0)  # (I, N, M)

        # Initial separated spectrograms
        Y = torch.einsum('inm,imj->inj', W, X_c)  # (I, N, J)

        # NMF parameters T(i,k,n), V(k,j,n)
        T = torch.rand(I, K, N, device=X.device) * (1 - 1e-8) + 1e-8
        V = torch.rand(K, J, N, device=X.device) * (1 - 1e-8) + 1e-8

        eye_M = torch.eye(M, dtype=torch.complex64, device=X.device)

        for _ in range(self.n_iter):
            for n in range(N):
                # --- 1. NMF update ---
                Y_hat_n = Y[:, n, :]  # (I, J)
                T[:, :, n], V[:, :, n], Rn = nmf_update(
                    T[:, :, n], V[:, :, n], Y_hat_n, eps=self.nmf_eps
                )
                # Rn: (I, J)

                # --- 2. IP update for all frequencies ---
                # Compute X_sqrt_weighted = X_i / sqrt(r_ijn)
                inv_Rn = 1.0 / (Rn + self.ip_eps)  # (I, J)
                X_sqrt_weighted = X_c * torch.sqrt(inv_Rn).unsqueeze(1)  # (I, M, J)

                # D_in(i) = 1/J * X_sqrt_weighted(i) X_sqrt_weighted(i)^H, shape (I, M, M)
                D_in = torch.matmul(
                    X_sqrt_weighted,
                    X_sqrt_weighted.conj().transpose(1, 2)
                ) / J  # (I, M, M)

                # Regularize for numerical stability
                D_reg = D_in + self.ip_eps * eye_M.unsqueeze(0)

                # A_i = W_i D_reg_i, shapes (I, M, M)
                A = torch.matmul(W, D_reg)  # (I, M, M) because M == N

                # e_n: nth standard basis vector (column), broadcast over I
                e_n = eye_M[:, n].view(1, M, 1).expand(I, -1, -1)  # (I, M, 1)

                # Solve A_i b_in(i) = e_n
                b_in = torch.linalg.solve(A, e_n)  # (I, M, 1)

                # denom_i = b_in(i)^H D_reg(i) b_in(i)
                b_H = b_in.conj().transpose(1, 2)  # (I, 1, M)
                denom = torch.matmul(torch.matmul(b_H, D_reg), b_in)  # (I, 1, 1)
                denom = torch.sqrt(denom.real + self.ip_eps)  # real positive scalar per frequency

                # Update nth row of W_i: w_in^H = b_in^H / sqrt(b_in^H D_in b_in)
                w_in = (b_in / denom).squeeze(-1).conj()  # (I, M)
                W[:, n, :] = w_in

            # Recompute Y for next iteration
            Y = torch.einsum('inm,imj->inj', W, X_c)

        # Return Y in original axis order: (M, J, I, 2)
        Y_out = Y.permute(1, 2, 0)  # (N, J, I)
        Y_out = torch.view_as_real(Y_out).contiguous()  # (N, J, I, 2)
        return Y_out
