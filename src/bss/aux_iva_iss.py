"""Auxiliary-function-based Independent Vector Analysis with Iterative Source Steering."""

import torch
from .utils import contrast_weights


class AUX_IVA_ISS(torch.nn.Module):
    """Auxiliary-function-based Independent Vector Analysis with Iterative Source Steering.

    AUX-IVA uses auxiliary function optimization for IVA, which separates sources
    by exploiting dependencies across frequency bins. The ISS variant applies
    iterative source steering updates for improved convergence.

    Args:
        n_iter (int): Number of iterations. Default: 20
        ref_mic (int): Reference microphone for back-projection (1-based, 0 for none). Default: 0
        contrast_func (str): Contrast function for independence measure. Options:
            "laplace", "gaussian", "logcosh", "exp", "pow1.5", "pow0.5", "power". Default: "laplace"
        gamma (float): Exponent for "power" contrast function. Default: 1.0
        tol (float): Convergence tolerance. Default: 1e-5
        eps_r (float): Small constant for numerical stability in norms. Default: 1e-10
        reg (float): Regularization parameter for covariance matrices. Default: 1e-6
    """

    def __init__(self, n_iter: int = 20, ref_mic: int = 0,
                 contrast_func: str = "laplace",
                 gamma: float = 1.0,
                 tol: float = 1e-5,
                 eps_r: float = 1e-10,
                 reg: float = 1e-6):
        super().__init__()
        self.n_iter = n_iter
        self.ref_mic = ref_mic
        self.contrast_func = contrast_func
        self.gamma = gamma
        self.tol = tol
        self.eps_r = eps_r
        self.reg = reg


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
            w_t = contrast_weights(r, self.contrast_func, self.gamma, self.eps_r)  # (K, T)

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
