"""Online OverIVA with Iterative Projection (IP).

Implements the per-time-frame (online) OverIVA update using the IP algorithm.
This is for over-determined blind source separation where M > K
(more microphones than target sources).

This processes a full STFT sequence but updates the demixing matrices in a
causal, frame-by-frame manner using a forgetting factor.

Reference:
    R. Scheibler and N. Ono, "Independent Vector Analysis with more Microphones
    than Sources", arXiv, 2019. https://arxiv.org/abs/1905.07880
"""

import torch

from .utils import contrast_weights

# Debug switch: Set to True for vectorized (fast), False for loop version (debug)
USE_VECTORIZED_UPDATE = True


class AUX_OVER_IVA_ONLINE(torch.nn.Module):
    """Online Auxiliary-function-based over-determined IVA.

    Args:
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
        num_targets: int = 1,
        n_iter: int = 1,
        alpha: float = 0.96,
        ref_mic: int = 0,
        contrast_func: str = "laplace",
        gamma: float = 1.0,
        proj_back_type: str = "mdp",
        scale_smooth_factor: float = 0.9,
        eps_r: float = 1e-10,
        reg: float = 1e-6,
    ):
        """Initialize Online OverIVA.
        
        Args:
            scale_smooth_factor: Smoothing factor for MDP scale (0 < factor < 1).
                Higher values = slower adaptation, more stable amplitude.
                Set to 0.0 to disable smoothing.
        """
        super().__init__()
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.num_targets = num_targets
        self.n_iter = n_iter
        self.alpha = alpha
        self.ref_mic = ref_mic
        self.contrast_func = contrast_func
        self.gamma = gamma
        self.proj_back_type = proj_back_type
        self.scale_smooth_factor = scale_smooth_factor
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
            # CRITICAL: Conjugate when assigning to row (w is column vector, row needs w^H)
            W_hat[:, k, :] = w_new.conj()

        # --- B. Update J (Orthogonal Constraint Step) ---
        W_f = W_hat[:, :K, :]  # (F, K, M)
        tmp = W_f @ C  # (F, K, M)
        
        # J = solve(tmp[:, :, :K], tmp[:, :, K:])^H
        J = torch.linalg.solve(tmp[:, :, :K], tmp[:, :, K:]).mH  # (F, M-K, K)
        W_hat[:, K:, :K] = J

        return W_hat

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Online OverIVA.

        Args:
            X: mixture STFT, shape (K, T, F, 2) real/imag.

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

        # Demixing matrices per frequency (F, M, M)
        # W_hat = [W; U] where U = [J, -I_{M-K}]
        W_hat = torch.eye(M, dtype=dtype, device=device).unsqueeze(0).repeat(F, 1, 1)
        # Set bottom-right block to -I_{M-K} (as per reference implementation)
        W_hat[:, K:, K:] = -torch.eye(M - K, dtype=dtype, device=device)

        # Online covariance matrices V_{k,f} (K, F, M, M) - only for K target sources
        eye = torch.eye(M, dtype=dtype, device=device)
        V = eye.unsqueeze(0).unsqueeze(0).repeat(K, F, 1, 1)

        # Online Input signal's covariance matrix C_{f} (F, M, M)
        C = torch.eye(M, dtype=dtype, device=device).unsqueeze(0).repeat(F, 1, 1)

        # Output: only K target sources
        Y = torch.zeros(F, K, T, dtype=dtype, device=device)
        alpha = self.alpha
        
        # Running scale buffer for MDP projection back smoothing
        running_scale = [None] * F

        for t in range(T):
            x_t = X_c[:, :, t]  # (F, M)

            # W = first K rows of W_hat (target sources demixing vectors)
            W = W_hat[:, 0:K, :]  # (F, K, M)
            # Compute y_t for target sources
            y_t = torch.einsum('fkj,fj->fk', W, x_t)  # (F, K)

            # r_{k,t} = sqrt(sum_f |y_{k,f,t}|^2)
            r_t = torch.sqrt(torch.clamp((y_t.abs() ** 2).sum(dim=0), min=self.eps_r))  # (K,)
            phi_t = contrast_weights(r_t, self.contrast_func, self.gamma, self.eps_r)  # (K,)

            # Update covariance matrices
            # xxH[f] = x_t[f] @ x_t[f]^H -> (F, M, M)
            xxH = torch.einsum('fi,fj->fij', x_t, x_t.conj())  # (F, M, M)

            C = alpha * C + (1.0 - alpha) * xxH
            for k in range(K):
                V[k] = alpha * V[k] + (1.0 - alpha) * phi_t[k] * xxH

            for _ in range(self.n_iter):
                if USE_VECTORIZED_UPDATE:
                    W_hat = self._update_vectorized(W_hat, C, V, M, K, F, dtype, device)
                else:
                    W_hat = self._update_loop(W_hat, C, V, M, K, F, dtype, device)

            # Store separated output for this frame (only target sources)
            W = W_hat[:, 0:K, :]  # (F, K, M)
            Y[:, :, t] = torch.einsum('fkj,fj->fk', W, x_t)  # (F, K)

            # Projection back (optional, per-frame for online use)
            if self.ref_mic is not None and self.ref_mic > 0 and self.ref_mic <= M:
                ref = self.ref_mic - 1  # convert 1-based (MATLAB) to 0-based

                if self.proj_back_type == "mdp":
                    # Minimal Distortion Principle
                    # For over-determined case, use pseudo-inverse of W (K x M)
                    for f in range(F):
                        Af = torch.linalg.pinv(W[f])  # (M, K)
                        current_scale = Af[ref, :]  # (K,)
                        
                        # Scale smoothing to avoid amplitude fluctuations
                        if self.scale_smooth_factor > 0:
                            if running_scale[f] is None:
                                running_scale[f] = current_scale
                            else:
                                running_scale[f] = (self.scale_smooth_factor * running_scale[f] 
                                                    + (1.0 - self.scale_smooth_factor) * current_scale)
                            Y[f, :, t] = running_scale[f] * Y[f, :, t]
                        else:
                            Y[f, :, t] = current_scale * Y[f, :, t]

                elif self.proj_back_type == "scale_constraint":
                    # Scale constraint
                    for f in range(F):
                        W_pinv = torch.linalg.pinv(W[f])  # (M, K)
                        # Get max absolute value for each column (source)
                        d = W_pinv.abs().max(dim=0).values  # (K,)
                        # Apply scaling to separated signals
                        Y[f, :, t] = d * Y[f, :, t]

                # else: proj_back_type == "none" - do nothing

        # Return in (K, T, F, 2) like ILRMA (sources x time x freq x complex)
        Y_out = Y.permute(1, 2, 0).contiguous()  # (K, T, F)
        Y_out = torch.view_as_real(Y_out)
        return Y_out
