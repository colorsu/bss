"""Independent Low-Rank Matrix Analysis (ILRMA)."""

import torch
from .utils import nmf_update


class ILRMA(torch.nn.Module):
    """Independent Low-Rank Matrix Analysis for blind source separation.

    ILRMA models each source with a low-rank NMF-based spectral model
    and jointly estimates demixing filters and source models.

    Args:
        n_components (int): Number of sources (must equal number of channels). Default: 2
        k_NMF_bases (int): Number of NMF basis functions. Default: 8
        n_iter (int): Number of iterations. Default: 30
        nmf_eps (float): Small constant for NMF stability. Default: 1e-20
        ip_eps (float): Small constant for IP update stability. Default: 1e-20
    """

    def __init__(self, n_components=2, k_NMF_bases=8, n_iter=30,
                 nmf_eps: float = 1e-20, ip_eps: float = 1e-20):
        super(ILRMA, self).__init__()
        self.n_components = n_components
        self.k_NMF_bases = k_NMF_bases
        self.n_iter = n_iter
        self.nmf_eps = nmf_eps
        self.ip_eps = ip_eps

    def forward(self, X):
        M, J, I, _ = X.shape  # M channels, J time frames, I frequency bins, complex dimension
        N = self.n_components

        assert M == N, "ILRMA assumes number of sources equals number of channels (M == N)."
        K = self.k_NMF_bases
        X = torch.view_as_complex(X.permute(2, 0, 1, 3))  # (I, M, J)
        W = torch.stack([
            torch.eye(N, M, dtype=torch.complex64, device=X.device)
            for _ in range(I)
        ], dim=0)  # (I, N, M)
        Y = torch.einsum('inm,imj->inj', W, X)
        T = torch.rand(I, K, N, device=X.device) * (1 - 1e-8) + 1e-8  # uniform in [1e-8, 1)
        V = torch.rand(K, J, N, device=X.device) * (1 - 1e-8) + 1e-8  # uniform in [1e-8, 1)

        eye_M = torch.eye(M, dtype=torch.complex64, device=X.device)

        for l in range(self.n_iter):
            for n in range(N):
                # Extract Y_hat_n: magnitude spectrogram for source n
                Y_hat_n = Y[:, n, :]  # (I, J)
                T[:, :, n], V[:, :, n], Rn = nmf_update(
                    T[:, :, n], V[:, :, n], Y_hat_n, eps=self.nmf_eps
                )
                e_n = eye_M[:, n]
                for i in range(I):
                    # per-frequency NMF variance for source n: Rn[i] shape (J,)
                    inv_Rn_i = 1.0 / (Rn[i, :] + self.ip_eps)  # (J,)

                    # X_sqrt_weighted(i) = X(i) * sqrt(1 / Rn_i)
                    X_sqrt_weighted_i = X[i, :, :] * torch.sqrt(inv_Rn_i).unsqueeze(0)  # (M, J)

                    # D_in(i) = 1/J * X_sqrt_weighted(i) X_sqrt_weighted(i)^H
                    D_in = X_sqrt_weighted_i @ X_sqrt_weighted_i.conj().T / J  # (M, M)

                    # Regularize for numerical stability
                    D_reg = D_in + self.ip_eps * eye_M

                    # A_i = W_i D_reg_i
                    A = W[i, :, :] @ D_reg  # (N, M) x (M, M) -> (N, M)

                    # Solve A_i^H b = e_n; keep it equivalent to the original scalar form that solved A b = e_n
                    b_in = torch.linalg.solve(A, e_n)  # (M,)

                    # denom_i = b_in^H D_reg b_in
                    denom = b_in.conj().unsqueeze(0) @ (D_reg @ b_in.unsqueeze(1))  # (1,1)
                    denom = torch.sqrt(denom.real + self.ip_eps)

                    W[i, n, :] = (b_in / denom).conj()

                # check if nan occurs
                if torch.isnan(W).any():
                    raise ValueError("NaN occurred in W matrix during ILRMA iterations.")
            Y = torch.einsum('inm,imj->inj', W, X)

        return torch.view_as_real(Y.permute(1, 2, 0).contiguous())  # (N, J, I)
