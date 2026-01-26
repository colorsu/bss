"""NoisyILRMA: Diffuse-Noise-Aware Independent Low-Rank Matrix Analysis.

Reference:
    K. Nishida, N. Takamune, R. Ikeshita, D. Kitamura, H. Saruwatari, and T. Nakatani,
    "NoisyILRMA: Diffuse-Noise-Aware Independent Low-Rank Matrix Analysis
    for Fast Blind Source Extraction," ICASSP 2020.
"""

import torch

from ..registry import register_bss


@register_bss("ILRMA_NOISY")
class ILRMA_NOISY(torch.nn.Module):
    """NoisyILRMA: Diffuse-Noise-Aware ILRMA for fast blind source extraction.
    
    This implementation extends ILRMA by modeling both target source and diffuse
    noise explicitly. The key assumption is that noise leaks into the target
    channel with a frequency-dependent weight λ.

    Features source model switching (Section D of paper):
    - Phase 1 (iter < switch_iter): NMF-based variance model with GEVD W update
    - Phase 2 (iter >= switch_iter): Fix W, switch to unconstrained r_s/r_n with
      inverse-gamma prior for finer per-T-F estimation (Eq. 23-24)
    
    This switching mechanism improves SAR by eliminating the "over-smoothing"
    artifact from NMF's low-rank approximation.

    Model:
        - Target channel y_1 ~ N(0, r_s + r_n * λ)
        - Noise channels y_n ~ N(0, r_n) for n = 2, ..., M

    Reference:
        K. Nishida, N. Takamune, R. Ikeshita, D. Kitamura, H. Saruwatari, and T. Nakatani,
        "NoisyILRMA: Diffuse-Noise-Aware Independent Low-Rank Matrix Analysis
        for Fast Blind Source Extraction," ICASSP 2020.

    Args:
        n_components (int): Number of sources (= number of channels). Default: 2
        k_NMF_bases (int): Number of NMF basis functions. Default: 8
        n_iter (int): Total number of iterations. Default: 30
        switch_iter (int): Iteration to switch from NMF to direct r update.
            Set to n_iter to disable switching (pure NMF mode). Default: 20
        alpha (float): Inverse-gamma prior shape (α). Small values = less regularization. Default: 0.1
        beta (float): Inverse-gamma prior scale (β). Set to 0 for ML estimation. Default: 0.0
        eps (float): Small constant for numerical stability. Default: 1e-10
    """

    def __init__(self, n_components=2, k_NMF_bases=8, n_iter=30, switch_iter=20,
                 alpha: float = 0.1, beta: float = 0.0, eps: float = 1e-10):
        super().__init__()
        self.n_components = n_components
        self.k_NMF_bases = k_NMF_bases
        self.n_iter = n_iter
        self.switch_iter = switch_iter
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, X):
        """
        Args:
            X: (M, J, I, 2) real/imag STFT spectrogram

        Returns:
            S_hat: (M, J, I, 2) estimated target signal (replicated across channels)
        """
        M, J, I, _ = X.shape
        N = self.n_components
        K = self.k_NMF_bases
        eps = self.eps
        alpha = self.alpha
        beta = self.beta
        
        assert M == N, f"ILRMA_NOISY requires M == N. Got M={M}, N={N}"

        # Convert to complex: (I, M, J)
        X_c = torch.view_as_complex(X.permute(2, 0, 1, 3).contiguous())

        # Initialize demixing matrix W: (I, M, M)
        W = torch.stack([
            torch.eye(M, dtype=torch.complex64, device=X.device)
            for _ in range(I)
        ], dim=0)

        # Initial separation: Y = W^H @ X, shape (I, M, J)
        Y = torch.einsum('imn,inj->imj', W.conj().transpose(1, 2), X_c)

        # Initialize NMF parameters
        # Target source: T_s (I, K), V_s (K, J)
        T_s = torch.rand(I, K, device=X.device) * 0.1 + eps
        V_s = torch.rand(K, J, device=X.device) * 0.1 + eps
        
        # Noise: T_n (I, K), V_n (K, J)
        T_n = torch.rand(I, K, device=X.device) * 0.1 + eps
        V_n = torch.rand(K, J, device=X.device) * 0.1 + eps
        
        # Noise weight per frequency: lambda_n (I,)
        lambda_n = torch.ones(I, device=X.device)
        
        # Direct variance parameters (initialized from NMF at switch point)
        r_s = torch.clamp(T_s @ V_s, min=eps)  # (I, J)
        r_n = torch.clamp(T_n @ V_n, min=eps)  # (I, J)

        # Identity matrix for regularization
        eye_M = torch.eye(M, dtype=torch.complex64, device=X.device)

        for iteration in range(self.n_iter):
            # === Step 1: Compute variances ===
            if iteration < self.switch_iter:
                # [Phase 1] NMF mode: r = T @ V
                r_s = torch.clamp(T_s @ V_s, min=eps)
                r_n = torch.clamp(T_n @ V_n, min=eps)
            # else: [Phase 2] r_s and r_n inherited from previous iteration
            
            # Target channel total variance: Λ_s = r_s + r_n * λ
            lambda_expand = lambda_n.unsqueeze(1)  # (I, 1)
            Lambda_s = r_s + r_n * lambda_expand
            Lambda_s = torch.clamp(Lambda_s, min=eps)

            # === Step 2: Compute separated signal powers ===
            Y_1 = Y[:, 0, :]  # (I, J) - target channel
            Y_noise = Y[:, 1:, :]  # (I, M-1, J) - noise channels
            
            power_y1 = torch.abs(Y_1) ** 2  # (I, J)
            power_noise = torch.sum(torch.abs(Y_noise) ** 2, dim=1)  # (I, J)

            # Precompute inverse terms
            inv_Lambda = 1.0 / Lambda_s
            inv_Lambda2 = inv_Lambda ** 2

            # ============================================
            # Branch: NMF Phase vs Switching Phase
            # ============================================
            if iteration < self.switch_iter:
                # ------------------------------------------
                # [Phase 1] NMF Updates (Eq. 17-21) + GEVD
                # ------------------------------------------
                
                # --- Update T_s (target basis) ---
                T_s_num = (power_y1 * inv_Lambda2) @ V_s.T
                T_s_den = inv_Lambda @ V_s.T + eps
                T_s = T_s * torch.sqrt(T_s_num / T_s_den)
                T_s = torch.clamp(T_s, min=eps)
                
                # Recompute r_s and Lambda_s
                r_s = torch.clamp(T_s @ V_s, min=eps)
                Lambda_s = torch.clamp(r_s + r_n * lambda_expand, min=eps)
                inv_Lambda = 1.0 / Lambda_s
                inv_Lambda2 = inv_Lambda ** 2

                # --- Update V_s (target activation) ---
                V_s_num = T_s.T @ (power_y1 * inv_Lambda2)
                V_s_den = T_s.T @ inv_Lambda + eps
                V_s = V_s * torch.sqrt(V_s_num / V_s_den)
                V_s = torch.clamp(V_s, min=eps)
                
                # Recompute variances
                r_s = torch.clamp(T_s @ V_s, min=eps)
                Lambda_s = torch.clamp(r_s + r_n * lambda_expand, min=eps)
                inv_Lambda = 1.0 / Lambda_s
                inv_Lambda2 = inv_Lambda ** 2
                inv_r_n = 1.0 / r_n
                inv_r_n2 = inv_r_n ** 2

                # --- Update T_n (noise basis, Eq. 19) ---
                T_n_num_target = (lambda_expand * power_y1 * inv_Lambda2) @ V_n.T
                T_n_den_target = (lambda_expand * inv_Lambda) @ V_n.T
                T_n_num_noise = (power_noise * inv_r_n2) @ V_n.T
                T_n_den_noise = ((M - 1) * inv_r_n) @ V_n.T
                
                T_n_num = T_n_num_target + T_n_num_noise
                T_n_den = T_n_den_target + T_n_den_noise + eps
                T_n = T_n * torch.sqrt(T_n_num / T_n_den)
                T_n = torch.clamp(T_n, min=eps)
                
                # Recompute r_n
                r_n = torch.clamp(T_n @ V_n, min=eps)
                Lambda_s = torch.clamp(r_s + r_n * lambda_expand, min=eps)
                inv_Lambda = 1.0 / Lambda_s
                inv_Lambda2 = inv_Lambda ** 2
                inv_r_n = 1.0 / r_n
                inv_r_n2 = inv_r_n ** 2

                # --- Update V_n (noise activation, Eq. 20) ---
                V_n_num_target = T_n.T @ (lambda_expand * power_y1 * inv_Lambda2)
                V_n_den_target = T_n.T @ (lambda_expand * inv_Lambda)
                V_n_num_noise = T_n.T @ (power_noise * inv_r_n2)
                V_n_den_noise = (M - 1) * T_n.T @ inv_r_n
                
                V_n_num = V_n_num_target + V_n_num_noise
                V_n_den = V_n_den_target + V_n_den_noise + eps
                V_n = V_n * torch.sqrt(V_n_num / V_n_den)
                V_n = torch.clamp(V_n, min=eps)
                
                # Recompute variances for lambda update
                r_n = torch.clamp(T_n @ V_n, min=eps)
                r_s = torch.clamp(T_s @ V_s, min=eps)
                Lambda_s = torch.clamp(r_s + r_n * lambda_expand, min=eps)
                inv_Lambda = 1.0 / Lambda_s
                inv_Lambda2 = inv_Lambda ** 2

                # --- Update lambda_n (Eq. 21) ---
                lambda_num = torch.sum(r_n * power_y1 * inv_Lambda2, dim=1)
                lambda_den = torch.sum(r_n * inv_Lambda, dim=1) + eps
                lambda_n = lambda_n * torch.sqrt(lambda_num / lambda_den)
                lambda_n = torch.clamp(lambda_n, min=eps)

                # --- Update W via GEVD (Eq. 14-16) ---
                Lambda_s = torch.clamp(r_s + r_n * lambda_n.unsqueeze(1), min=eps)
                inv_Lambda_s = 1.0 / Lambda_s
                inv_r_n = 1.0 / r_n
                
                X_weighted_s = X_c * torch.sqrt(inv_Lambda_s).unsqueeze(1)
                X_weighted_n = X_c * torch.sqrt(inv_r_n).unsqueeze(1)
                
                G_s = torch.matmul(X_weighted_s, X_weighted_s.conj().transpose(1, 2)) / J
                G_n = torch.matmul(X_weighted_n, X_weighted_n.conj().transpose(1, 2)) / J
                
                G_s = G_s + eps * eye_M.unsqueeze(0)
                G_n = G_n + eps * eye_M.unsqueeze(0)

                L_s = torch.linalg.cholesky(G_s)
                L_s_inv = torch.linalg.inv(L_s)
                C = torch.matmul(torch.matmul(L_s_inv, G_n), L_s_inv.conj().transpose(1, 2))
                eigenvalues, eigenvectors = torch.linalg.eigh(C)
                V_gevd = torch.matmul(L_s_inv.conj().transpose(1, 2), eigenvectors)
                
                sort_idx = torch.argsort(eigenvalues, dim=1, descending=False)
                
                for m in range(M):
                    idx = sort_idx[:, m]
                    h_m = V_gevd[torch.arange(I, device=X.device), :, idx]
                    
                    if m == 0:
                        h_m_col = h_m.unsqueeze(2)
                        denom = torch.matmul(torch.matmul(h_m_col.conj().transpose(1, 2), G_s), h_m_col)
                        denom = torch.sqrt(torch.clamp(denom.real, min=eps)).squeeze()
                        W[:, :, m] = h_m / denom.unsqueeze(1)
                    else:
                        h_m_col = h_m.unsqueeze(2)
                        denom = torch.matmul(torch.matmul(h_m_col.conj().transpose(1, 2), G_n), h_m_col)
                        denom = torch.sqrt(torch.clamp(denom.real, min=eps)).squeeze()
                        W[:, :, m] = h_m / denom.unsqueeze(1)

                # Recompute Y for next iteration
                Y = torch.einsum('imn,inj->imj', W.conj().transpose(1, 2), X_c)

            else:
                # ------------------------------------------
                # [Phase 2] Source Model Switching (Eq. 23-24)
                # W is FIXED, update r_s and r_n directly
                # ------------------------------------------
                
                inv_r_s = 1.0 / (r_s + eps)
                inv_r_s2 = inv_r_s ** 2
                inv_r_n = 1.0 / (r_n + eps)
                inv_r_n2 = inv_r_n ** 2
                
                # --- Update r_s (Eq. 23) ---
                # r_s <- r_s * sqrt((|y_1|^2/Lambda^2 + beta/r_s^2) / (1/Lambda + (alpha+1)/r_s))
                num_term1 = power_y1 * inv_Lambda2
                num_term2 = beta * inv_r_s2
                den_term1 = inv_Lambda
                den_term2 = (alpha + 1) * inv_r_s
                
                r_s = r_s * torch.sqrt((num_term1 + num_term2) / (den_term1 + den_term2 + eps))
                r_s = torch.clamp(r_s, min=eps)
                
                # Recompute Lambda_s
                Lambda_s = torch.clamp(r_s + r_n * lambda_expand, min=eps)
                inv_Lambda = 1.0 / Lambda_s
                inv_Lambda2 = inv_Lambda ** 2
                
                # --- Update r_n (Eq. 24) ---
                # r_n <- r_n * sqrt((λ|y_1|^2/Lambda^2 + Σ|y_n|^2/r_n^2) / (λ/Lambda + (M-1)/r_n))
                num_part1 = lambda_expand * power_y1 * inv_Lambda2
                num_part2 = power_noise * inv_r_n2
                den_part1 = lambda_expand * inv_Lambda
                den_part2 = (M - 1) * inv_r_n
                
                r_n = r_n * torch.sqrt((num_part1 + num_part2) / (den_part1 + den_part2 + eps))
                r_n = torch.clamp(r_n, min=eps)

                # --- Update lambda_n (Eq. 21, same form) ---
                Lambda_s = torch.clamp(r_s + r_n * lambda_expand, min=eps)
                inv_Lambda = 1.0 / Lambda_s
                inv_Lambda2 = inv_Lambda ** 2
                
                lambda_num = torch.sum(r_n * power_y1 * inv_Lambda2, dim=1)
                lambda_den = torch.sum(r_n * inv_Lambda, dim=1) + eps
                lambda_n = lambda_n * torch.sqrt(lambda_num / lambda_den)
                lambda_n = torch.clamp(lambda_n, min=eps)
                
                # NOTE: W is FIXED, so Y doesn't change. Skip Y recomputation.

        # === Final reconstruction with Wiener filtering (Eq. 22) ===
        Lambda_s = torch.clamp(r_s + r_n * lambda_n.unsqueeze(1), min=eps)
        
        # Wiener gain: r_s / Lambda_s
        wiener_gain = r_s / Lambda_s  # (I, J)
        
        # Steering vector: first column of W^{-H}
        try:
            W_inv = torch.linalg.inv(W)
            a_s = W_inv[:, 0, :].conj()  # (I, M)
        except RuntimeError:
            a_s = torch.zeros(I, M, device=X.device, dtype=X_c.dtype)
            a_s[:, 0] = 1.0
        
        # Separated target signal
        Y_1 = Y[:, 0, :]  # (I, J)
        
        # Reconstruct: s_hat = a_s * wiener_gain * y_1
        S_hat = a_s.unsqueeze(2) * (wiener_gain * Y_1).unsqueeze(1)

        # Convert to output format: (M, J, I, 2)
        S_out = S_hat.permute(1, 2, 0)  # (M, J, I)
        S_out = torch.view_as_real(S_out).contiguous()  # (M, J, I, 2)
        
        return S_out
