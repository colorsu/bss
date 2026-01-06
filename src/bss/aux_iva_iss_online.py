"""Online AuxIVA-ISS.

Implements the per-time-frame (online) AuxIVA-ISS update shown in the provided
algorithm figure.

This processes a full STFT sequence but updates the demixing matrices in a
causal, frame-by-frame manner using a forgetting factor.
"""

import torch
from .utils import contrast_weights

# Debug switch: Set to True for vectorized (fast), False for loop version (debug)
USE_VECTORIZED_ISS = True   


class AUX_IVA_ISS_ONLINE(torch.nn.Module):
    """Online Auxiliary-function-based IVA with ISS.

    Args:
        n_iter (int): Number of ISS iterations per time frame (N_iter in the figure).
        alpha (float): Forgetting factor α for the online covariance update (0 < α < 1).
        ref_mic (int): Reference mic for minimal-distortion back-projection (1-based, 0 for none).
        contrast_func (str): Contrast function; defines φ(r)=g'(r)/r.
            Options: "laplace", "gaussian", "logcosh", "exp", "pow1.5", "pow0.5", "power".
        gamma (float): Exponent for "power" contrast function. Default: 1.0.
        proj_back_type (str): Projection back method: "mdp", "scale_constraint", "none".
        eps_r (float): Small constant for numerical stability. Default: 1e-10.
        reg (float): Diagonal regularization added inside covariance updates. Default: 1e-6.
    """

    def __init__(
        self,
        n_iter: int = 2,
        alpha: float = 0.96,
        ref_mic: int = 0,
        contrast_func: str = "laplace",
        gamma: float = 1.0,
        proj_back_type: str = "mdp",
        eps_r: float = 1e-10,
        reg: float = 1e-6,
    ):
        """Initialize Online AuxIVA-ISS."""
        super().__init__()
        if n_iter < 1:
            raise ValueError("n_iter must be >= 1")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.n_iter = n_iter
        self.alpha = alpha
        self.ref_mic = ref_mic
        self.contrast_func = contrast_func
        self.gamma = gamma
        self.proj_back_type = proj_back_type
        self.eps_r = eps_r
        self.reg = reg

    def _iss_update_loop(self, W, U, x_t, K, F, dtype, device):
        """ISS update using explicit loops (for debugging)."""
        for k in range(K):
            for f in range(F):
                w_k = W[f, k, :].unsqueeze(0)  # (1, K)
                vk = torch.zeros(K, dtype=dtype, device=device)

                for m in range(K):
                    w_m = W[f, m, :].unsqueeze(0)  # (1, K)
                    U_mf = U[m, f]

                    denom = (w_k @ U_mf @ w_k.conj().T).real
                    denom = torch.clamp(denom, min=self.eps_r)

                    if m != k:
                        num = w_m @ U_mf @ w_k.conj().T
                        vk[m] = num / denom
                    else:
                        vk[m] = 1.0 - 1.0 / torch.sqrt(denom)

                # Rank-1 ISS update
                W[f] = W[f] - vk.unsqueeze(1) @ w_k
        return W

    def _iss_update_vectorized(self, W, U, x_t, K, F, dtype, device):
        """ISS update using vectorized operations (fast)."""
        for k in range(K):
            w_k = W[:, k, :]  # (F, K)

            # Compute w_k @ U_m @ w_k^H for all m, all f
            # U: (K, F, K, K), w_k: (F, K) -> w_k.conj(): (F, K)
            # temp[m, f, i] = sum_j U[m, f, i, j] * w_k[f, j].conj()
            temp = torch.einsum('mfij,fj->mfi', U, w_k.conj())  # (K, F, K)
            
            # denom[m, f] = sum_i w_k[f, i] * temp[m, f, i]
            denom = torch.einsum('fi,mfi->mf', w_k, temp).real  # (K, F)
            denom = torch.clamp(denom, min=self.eps_r)

            # Compute num = w_m @ U_m @ w_k^H for all m, all f
            # W: (F, K, K), temp: (K, F, K)
            # num[m, f] = sum_i W[f, m, i] * temp[m, f, i]
            num = torch.einsum('fmi,mfi->mf', W, temp)  # (K, F)

            # Compute v_k for all m, all f
            vk = num / denom  # (K, F) - off-diagonal elements
            
            # Diagonal element: m == k
            vk[k, :] = 1.0 - 1.0 / torch.sqrt(denom[k, :])

            # Rank-1 ISS update for all frequencies: W = W - v_k @ w_k^T
            # vk: (K, F) -> (F, K, 1), w_k: (F, K) -> (F, 1, K)
            W = W - vk.T.unsqueeze(2) * w_k.unsqueeze(1)

        return W

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Online AuxIVA-ISS.

        Args:
            X: mixture STFT, shape (K, T, F, 2) real/imag.

        Returns:
            Y: separated STFT, shape (K, T, F, 2) real/imag.
        """
        K, T, F, _ = X.shape

        # Convert to complex with shape (F, K, T)
        X_c = torch.view_as_complex(X).permute(2, 0, 1).contiguous()

        device = X_c.device
        dtype = X_c.dtype if X_c.is_complex() else torch.complex64

        # Demixing matrices per frequency (F, K, K)
        W = torch.eye(K, dtype=dtype, device=device).unsqueeze(0).repeat(F, 1, 1)

        # Online covariance matrices U_{k,f} (K, F, K, K)
        eye = torch.eye(K, dtype=dtype, device=device)
        U = eye.unsqueeze(0).unsqueeze(0).repeat(K, F, 1, 1)

        Y = torch.zeros_like(X_c)  # (F, K, T)
        alpha = self.alpha

        for t in range(T):
            x_t = X_c[:, :, t]  # (F, K)

            # First compute initial y_t and update covariance U ONCE per frame
            y_t = torch.einsum('fkj,fj->fk', W, x_t)  # (F, K)

            # r_{k,t} = sqrt(sum_f |y_{k,f,t}|^2)
            r_t = torch.sqrt(torch.clamp((y_t.abs() ** 2).sum(dim=0), min=self.eps_r))  # (K,)
            phi_t = contrast_weights(r_t, self.contrast_func, self.gamma, self.eps_r)  # (K,)

            # Update U_{k,f} with forgetting factor ONCE per frame (vectorized)
            # xxH[f] = x_t[f] @ x_t[f]^H -> (F, K, K)
            xxH = torch.einsum('fi,fj->fij', x_t, x_t.conj())  # (F, K, K)
            for k in range(K):
                U[k] = alpha * U[k] + (1.0 - alpha) * phi_t[k] * xxH

            # ISS iterations (update W multiple times using fixed U)
            for _ in range(self.n_iter):
                if USE_VECTORIZED_ISS:
                    W = self._iss_update_vectorized(W, U, x_t, K, F, dtype, device)
                else:
                    W = self._iss_update_loop(W, U, x_t, K, F, dtype, device)

            # Store separated output for this frame
            Y[:, :, t] = torch.einsum('fkj,fj->fk', W, x_t)

            # Projection back (optional, per-frame for online use)
            if self.ref_mic is not None and self.ref_mic > 0 and self.ref_mic <= K:
                ref = self.ref_mic - 1  # convert 1-based (MATLAB) to 0-based

                if self.proj_back_type == "mdp":
                    # Minimal Distortion Principle
                    for f in range(F):
                        Af = torch.linalg.pinv(W[f])  # (K, K)
                        scale = Af[ref, :]  # (K,)
                        Y[f, :, t] = scale * Y[f, :, t]

                elif self.proj_back_type == "scale_constraint":
                    # Scale constraint: D_inv = inv(D); d(m) = max(D_inv(:,m)); Demix = diag(d)*D
                    # This scales each source by the max element in corresponding column of W^{-1}
                    for f in range(F):
                        W_inv = torch.linalg.inv(W[f])  # (K, K)
                        # Get max absolute value for each column (source)
                        d = W_inv.abs().max(dim=0).values  # (K,)
                        # Apply scaling to separated signals
                        Y[f, :, t] = d * Y[f, :, t]
                        # Also update demixing matrix W to maintain consistency
                        W[f] = torch.diag(d) @ W[f]

                # else: proj_back_type == "none" - do nothing

        # Return in (K, T, F, 2) like ILRMA (sources x time x freq x complex)
        Y_out = Y.permute(1, 2, 0).contiguous()  # (K, T, F)
        Y_out = torch.view_as_real(Y_out)
        return Y_out
