import torch
import numpy as np

def nmf_update(Tn, Vn, Y_n, eps: float = 1e-20):
    """Shared NMF update used by ILRMA variants.

    Args:
        Tn: (I, K) nonnegative NMF basis matrix.
        Vn: (K, J) nonnegative NMF activation matrix.
        Y_n: (I, J) complex or real spectrogram for source n.
        eps: small constant for numerical stability.

    Returns:
        Tn_new, Vn_new, Rn_new where Rn_new = Tn_new @ Vn_new (I, J).
    """
    Rn = Tn @ Vn  # (I, J)
    mag_R = Rn.abs()
    Y_d_R = Y_n.abs() ** 2 / (mag_R ** 2 + eps)
    Rn_d = 1.0 / (mag_R + eps)

    Tn_num = Y_d_R @ Vn.T
    Tn_den = Rn_d @ Vn.T + eps
    Tn_new = Tn * torch.sqrt(Tn_num / Tn_den)

    Rn = Tn_new @ Vn
    mag_R_new = Rn.abs()
    Rn_d_new = 1.0 / (mag_R_new + eps)

    Vn_num = Tn_new.T @ Y_d_R
    Vn_den = Tn_new.T @ Rn_d_new + eps
    Vn_new = Vn * torch.sqrt(Vn_num / Vn_den)

    Rn_new = Tn_new @ Vn_new
    return Tn_new, Vn_new, Rn_new


class ILRMA(torch.nn.Module):
    def __init__(self, n_components=2, k_NMF_bases=8, n_iter=30,
                 nmf_eps: float = 1e-20, ip_eps: float = 1e-20):
        super(ILRMA, self).__init__()
        self.n_components = n_components
        self.k_NMF_bases = k_NMF_bases
        self.n_iter = n_iter
        self.nmf_eps = nmf_eps
        self.ip_eps = ip_eps

    def forward(self, X):
        M, J, I, _ = X.shape # M channels, J time frames, I frequency bins,  complex dimension
        N = self.n_components

        assert M == N, "ILRMA assumes number of sources equals number of channels (M == N)."
        K = self.k_NMF_bases
        X = torch.view_as_complex(X.permute(2,0,1,3)) # (I, M, J)
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

        return torch.view_as_real(Y.permute(1,2,0).contiguous())  # (N, J, I)

class ILRMA_V2(torch.nn.Module):
    def __init__(self, n_components=2, k_NMF_bases=8, n_iter=30,
                 nmf_eps: float = 1e-20, ip_eps: float = 1e-20):
        super().__init__()
        self.n_components = n_components
        self.k_NMF_bases = k_NMF_bases
        self.n_iter = n_iter
        self.nmf_eps = nmf_eps
        self.ip_eps = ip_eps

    def forward(self, X):
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
