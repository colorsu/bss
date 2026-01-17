"""Offline OverIVA with Iterative Projection (IP).

Implements batch (offline) OverIVA update using the IP algorithm.
This is for over-determined blind source separation where M > K
(more microphones than target sources).

Reference:
    R. Scheibler and N. Ono, "Independent Vector Analysis with more Microphones
    than Sources", arXiv, 2019. https://arxiv.org/abs/1905.07880
"""

import torch

from .utils import contrast_weights

# Debug switch: Set to True for vectorized (fast), False for loop version (debug)
USE_VECTORIZED_UPDATE = True


class AUX_OVER_IVA(torch.nn.Module):
    """Offline Auxiliary-function-based over-determined IVA.

    Args:
        num_targets (int): Number of target sources K (must be < num_mics M). Default: 1.
        n_iter (int): Number of iterations. Default: 20.
        ref_mic (int): Reference mic for minimal-distortion back-projection (1-based, 0 for none).
        contrast_func (str): Contrast function; defines φ(r)=g'(r)/r.
            Options: "laplace", "gaussian", "logcosh", "exp", "pow1.5", "pow0.5", "power".
        gamma (float): Exponent for "power" contrast function. Default: 1.0.
        proj_back_type (str): Projection back method: "mdp", "scale_constraint", "none".
        tol (float): Convergence tolerance. Default: 1e-5.
        eps_r (float): Small constant for numerical stability. Default: 1e-10.
        reg (float): Diagonal regularization added inside covariance updates. Default: 1e-6.
    """

    def __init__(
        self,
        num_targets: int = 1,
        n_iter: int = 20,
        ref_mic: int = 0,
        contrast_func: str = "laplace",
        gamma: float = 1.0,
        proj_back_type: str = "mdp",
        tol: float = 1e-5,
        eps_r: float = 1e-10,
        reg: float = 1e-6,
    ):
        """Initialize Offline OverIVA."""
        super().__init__()
        self.num_targets = num_targets
        self.n_iter = n_iter
        self.ref_mic = ref_mic
        self.contrast_func = contrast_func
        self.gamma = gamma
        self.proj_back_type = proj_back_type
        self.tol = tol
        self.eps_r = eps_r
        self.reg = reg

    def _update_loop(self, W_hat, C, V, M, K, F, dtype, device):
        """
        Loop-based OverIVA IP update.
        
        Reference formula for J (from pyroomacoustics):
            tmp = W @ Cx
            J = solve(tmp[:, :K], tmp[:, K:])^H
        """
        # Add regularization to V for numerical stability
        reg_mat = self.reg * torch.eye(M, dtype=dtype, device=device)
        
        for k in range(K):
            e_k = torch.zeros(M, dtype=dtype, device=device)
            e_k[k] = 1.0
            for f in range(F):
                # IP update step: w = (W_hat @ V)^{-1} e_k
                V_reg = V[k, f] + reg_mat  # Add regularization
                temp = W_hat[f, :, :] @ V_reg
                w_kf = torch.linalg.solve(temp, e_k)

                # Normalize: w = w / sqrt(w^H V w)
                denom = torch.sqrt((w_kf.conj() @ V_reg @ w_kf).real + self.eps_r)
                w_kf = w_kf / denom

                # CRITICAL: Conjugate when assigning to row (w is column vector, row needs w^H)
                W_hat[f, k, :] = w_kf.conj()
                
            # Update J after updating w_k (orthogonal constraint)
            for f in range(F):
                W_f = W_hat[f, 0:K, :]  # (K, M)
                C_f = C[f]  # (M, M)
                tmp = W_f @ C_f  # (K, M)
                # J = solve(tmp[:, :K], tmp[:, K:])^H
                J_f = torch.linalg.solve(tmp[:, :K], tmp[:, K:]).conj().T  # (M-K, K)
                W_hat[f, K:, 0:K] = J_f
                
        return W_hat

    def _update_vectorized(self, W_hat, C, V, M, K, F, dtype, device):
        """
        Vectorized OverIVA IP update across all frequencies.
        
        Args:
            W_hat: (F, M, M) - Full demixing matrix
            C:     (F, M, M) - Input covariance
            V:     (K, F, M, M) - Weighted covariance for each target source
        """
        # Add regularization to V for numerical stability
        reg_mat = self.reg * torch.eye(M, dtype=dtype, device=device).reshape(1, 1, M, M)
        V_reg = V + reg_mat  # (K, F, M, M)
        
        # Loop over K sources (OverIVA requires serial source updates)
        for k in range(K):
            # --- A. Update w_kf (IP Step) ---
            
            # Compute (W_hat @ V_reg[k])
            temp = W_hat @ V_reg[k]  # (F, M, M)
            
            # Solve for w: (W_hat @ V) @ w = e_k
            e_k = torch.zeros(F, M, dtype=dtype, device=device)
            e_k[:, k] = 1.0
            w_vec = torch.linalg.solve(temp, e_k)  # (F, M)
            
            # Normalize: w = w / sqrt(w^H V w)
            w_vec_unsq = w_vec.unsqueeze(1)  # (F, 1, M)
            denom_mat = w_vec_unsq.conj() @ V_reg[k] @ w_vec.unsqueeze(2)
            denom = torch.sqrt(denom_mat.squeeze(-1).squeeze(-1).real + self.eps_r)  # (F,)
            
            w_new = w_vec / denom.unsqueeze(1)
            W_hat[:, k, :] = w_new.conj()

        # --- B. Update J (Orthogonal Constraint Step) ---
        W_f = W_hat[:, :K, :]  # (F, K, M)
        tmp = W_f @ C  # (F, K, M)
        
        # J = solve(tmp[:, :, :K], tmp[:, :, K:])^H
        J = torch.linalg.solve(tmp[:, :, :K], tmp[:, :, K:]).mH  # (F, M-K, K)
        W_hat[:, K:, :K] = J

        return W_hat

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Offline OverIVA.

        Args:
            X: mixture STFT, shape (M, T, F, 2) real/imag.

        Returns:
            Y: separated STFT, shape (K, T, F, 2) real/imag.
        """
        M, T, F, _ = X.shape
        K = self.num_targets
        if K >= M:
            raise ValueError("num_targets must be less than num_mics for over-determined IVA.")

        # Convert to complex with shape (F, M, T)
        X_c = torch.view_as_complex(X).permute(2, 0, 1).contiguous()

        device = X_c.device
        dtype = X_c.dtype if X_c.is_complex() else torch.complex64

        eps = torch.finfo(torch.float32).eps

        # Demixing matrices per frequency (F, M, M)
        # W_hat = [W; U] where U = [J, -I_{M-K}]
        W_hat = torch.eye(M, dtype=dtype, device=device).unsqueeze(0).repeat(F, 1, 1)
        # Set bottom-right block to -I_{M-K} (as per reference implementation)
        W_hat[:, K:, K:] = -torch.eye(M - K, dtype=dtype, device=device)

        for iteration in range(self.n_iter):
            W_old = W_hat.clone()

            # 1) Compute Y = W @ X for target sources
            # W = first K rows of W_hat (target sources demixing vectors)
            W = W_hat[:, 0:K, :]  # (F, K, M)
            Y = torch.einsum('fkm,fmt->fkt', W, X_c)  # (F, K, T)

            # 2) Compute cross-frequency norms r_{k,t} = sqrt(sum_f |Y_{k,f,t}|^2)
            Y_mag2 = (Y.abs() ** 2).sum(dim=0)  # (K, T)
            r = torch.sqrt(torch.clamp(Y_mag2, min=self.eps_r))  # (K, T)

            # 3) Compute contrast weights
            w_t = contrast_weights(r, self.contrast_func, self.gamma, self.eps_r)  # (K, T)

            # 4) Compute weighted covariance matrices V_{k,f} and input covariance C_f
            # V[k,f] = (1/T) * sum_t phi_k(t) * x_t * x_t^H
            # C[f] = (1/T) * sum_t x_t * x_t^H
            
            # Input covariance C: (F, M, M)
            C = torch.einsum('fmt,fnt->fmn', X_c, X_c.conj()) / T  # (F, M, M)
            
            # Weighted covariance V: (K, F, M, M)
            # Expand w_t to weight X_c: w_t is (K, T), X_c is (F, M, T)
            V = torch.zeros(K, F, M, M, dtype=dtype, device=device)
            for k in range(K):
                # X_c * w_t[k] -> (F, M, T) weighted by (T,)
                Xw = X_c * w_t[k].unsqueeze(0).unsqueeze(0)  # (F, M, T)
                V[k] = torch.einsum('fmt,fnt->fmn', Xw, X_c.conj()) / T  # (F, M, M)

            # 5) IP update
            if USE_VECTORIZED_UPDATE:
                W_hat = self._update_vectorized(W_hat, C, V, M, K, F, dtype, device)
            else:
                W_hat = self._update_loop(W_hat, C, V, M, K, F, dtype, device)

            # 6) Convergence check
            diff = (W_hat - W_old).view(-1)
            rel_change = diff.norm() / torch.clamp(W_old.view(-1).norm(), min=eps)
            if rel_change.item() < self.tol:
                break

        # Compute final output
        W = W_hat[:, 0:K, :]  # (F, K, M)
        Y = torch.einsum('fkm,fmt->fkt', W, X_c)  # (F, K, T)

        # Projection back (optional)
        if self.ref_mic is not None and self.ref_mic > 0 and self.ref_mic <= M:
            ref = self.ref_mic - 1  # convert 1-based (MATLAB) to 0-based

            if self.proj_back_type == "mdp":
                # Minimal Distortion Principle
                # For over-determined case, use pseudo-inverse of W (K x M)
                for f in range(F):
                    Af = torch.linalg.pinv(W[f])  # (M, K)
                    scale = Af[ref, :]  # (K,)
                    Y[f] = scale.unsqueeze(1) * Y[f]

            elif self.proj_back_type == "scale_constraint":
                # Scale constraint
                for f in range(F):
                    W_pinv = torch.linalg.pinv(W[f])  # (M, K)
                    # Get max absolute value for each column (source)
                    d = W_pinv.abs().max(dim=0).values  # (K,)
                    # Apply scaling to separated signals
                    Y[f] = d.unsqueeze(1) * Y[f]

            # else: proj_back_type == "none" - do nothing

        # Return in (K, T, F, 2) like ILRMA (sources x time x freq x complex)
        Y_out = Y.permute(1, 2, 0).contiguous()  # (K, T, F)
        Y_out = torch.view_as_real(Y_out)
        return Y_out
