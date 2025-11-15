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


class AUX_IVA_ISS(torch.nn.Module):
    def __init__(self, n_iter: int = 20, ref_mic: int = 0,
                 contrast_func: str = "laplace",
                 tol: float = 1e-5,
                 eps_r: float = 1e-10,
                 reg: float = 1e-6):
        super().__init__()
        self.n_iter = n_iter
        self.ref_mic = ref_mic
        self.contrast_func = contrast_func
        self.tol = tol
        self.eps_r = eps_r
        self.reg = reg

    def _contrast_weights(self, r: torch.Tensor) -> torch.Tensor:
        """Compute contrast weights g'(r)/r for different contrast functions.

        Args:
            r: (...,) nonnegative norms.

        Returns:
            (...,) weights.
        """
        eps_r = self.eps_r
        r_safe = torch.clamp(r, min=eps_r)
        if self.contrast_func == "laplace":
            # g(r) = r -> g'(r)/r = 1/r
            return 1.0 / r_safe
        elif self.contrast_func == "gaussian":
            # g(r) = r^2 -> g'(r)/r = 2
            return torch.full_like(r_safe, 2.0)
        elif self.contrast_func == "logcosh":
            # g(r) = log(cosh(r)) -> g'(r)/r = tanh(r)/r
            return torch.tanh(r_safe) / r_safe
        elif self.contrast_func == "exp":
            # g(r) = -exp(-r^2/2) -> g'(r)/r = exp(-r^2/2)
            return torch.exp(-(r_safe ** 2) / 2.0)
        elif self.contrast_func == "pow1.5":
            # g(r) = r^1.5 -> g'(r)/r = 1.5/sqrt(r)
            return 1.5 / torch.sqrt(r_safe)
        elif self.contrast_func == "pow0.5":
            # g(r) = r^0.5 -> g'(r)/r = 0.5/r^1.5
            return 0.5 / (r_safe ** 1.5)
        else:
            raise ValueError(f"Unknown contrast function: {self.contrast_func}")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """AUX-IVA (offline) with iterative source steering (ISS).

        Args:
            X: mixture STFT, shape (K, T, F, 2) real/imag.

        Returns:
            Y: separated STFT, shape (K, T, F, 2) real/imag.
        """
        # Match ILRMA style: assume X real/imag last, transpose to (F, K, T)
        # MATLAB: X: [K, F, T]; here we accept (K, T, F, 2).
        K, T, F, _ = X.shape

        # Convert to complex tensor with shape (F, K, T) for easier mapping
        X_c = torch.view_as_complex(X).permute(2, 0, 1).contiguous()

        device = X_c.device
        dtype = torch.complex64

        # Demixing matrices per frequency: W_f in C^{K x K}
        # We keep W as (F, K, K) to mirror ILRMA's per-frequency structure
        W = torch.stack([
            torch.eye(K, dtype=dtype, device=device)
            for _ in range(F)
        ], dim=0)  # (F, K, K)

        eps = torch.finfo(torch.float32).eps

        for _ in range(self.n_iter):
            W_old = W.clone()

            # 1) Y = W * X per frequency, Y: (F, K, T)
            Y = torch.zeros_like(X_c)
            for f in range(F):
                # X_f: (K, T); W_f: (K, K)
                X_f = X_c[f, :, :]  # (K, T)
                Y[f, :, :] = W[f] @ X_f

            # 2) Cross-frequency norms r_{k,t} = sqrt(sum_f |Y_{k,f,t}|^2)
            # Y: (F, K, T) -> abs^2 sum over F -> (K, T)
            Y_mag2 = (Y.abs() ** 2).sum(dim=0)  # (K, T)
            r = torch.sqrt(torch.clamp(Y_mag2, min=self.eps_r))  # (K, T)

            # 2b) Weights w_t(k,t)
            w_t = self._contrast_weights(r)  # (K, T)

            # 3) Build weighted covariance V_{k,f} and update w_{k,f} via ISS
            for f in range(F):
                X_f = X_c[f, :, :]  # (K, T)

                # V has shape (K, K, K) storing V_kf for each source k
                V = torch.zeros(K, K, K, dtype=dtype, device=device)

                for k in range(K):
                    # Avoid explicit diag: scale columns by weights
                    # w_t[k]: (T,) -> (1, T) to scale each time frame
                    Xw = X_f * w_t[k].unsqueeze(0)  # (K, T)
                    V_kf = (Xw @ X_f.conj().T) / T  # (K, K)
                    V_kf = V_kf + self.reg * torch.eye(K, dtype=dtype, device=device)
                    V[k] = V_kf

                # ISS updates for each source k at this frequency
                for k in range(K):
                    w_k = W[f, k, :].clone().unsqueeze(0)  # (1, K)
                    V_kf = V[k]

                    vk = torch.zeros(K, dtype=dtype, device=device)
                    for m in range(K):
                        w_m = W[f, m, :].unsqueeze(0)  # (1, K)
                        V_mf = V[m]

                        denom = (w_k @ V_mf @ w_k.conj().T).real
                        denom = torch.clamp(denom, min=self.eps_r)

                        if m != k:
                            num = w_m @ V_mf @ w_k.conj().T
                            vk[m] = num / denom
                        else:
                            vk[m] = 1.0 - 1.0 / torch.sqrt(denom)

                    # Rank-1 ISS update
                    W[f] = W[f] - vk.unsqueeze(1) @ w_k

            # Relative change for convergence check
            diff = (W - W_old).view(-1)
            rel_change = diff.norm() / torch.clamp(W_old.view(-1).norm(), min=eps)
            if rel_change.item() < self.tol:
                break

        # Final output with latest W
        Y = torch.zeros_like(X_c)
        for f in range(F):
            X_f = X_c[f, :, :]
            Y[f] = W[f] @ X_f

        # Minimal-distortion back-projection (optional)
        if self.ref_mic is not None and self.ref_mic > 0 and self.ref_mic <= K:
            ref = self.ref_mic - 1  # convert 1-based (MATLAB) to 0-based
            for f in range(F):
                # Af: estimate of mixing matrix = pinv(W_f)
                Af = torch.linalg.pinv(W[f])  # (K, K)
                scale = Af[ref, :]  # (K,)
                Y[f] = scale.unsqueeze(1) * Y[f]

        # Return in (K, T, F, 2) like ILRMA (sources x time x freq x complex)
        Y_out = Y.permute(1, 2, 0).contiguous()  # (K, T, F)
        Y_out = torch.view_as_real(Y_out)
        return Y_out