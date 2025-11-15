import torch
import numpy as np


class ILRMA(torch.nn.Module):
    def __init__(self, n_components=2, k_NMF_bases=8, n_iter=30):
        super(ILRMA, self).__init__()
        self.n_components = n_components
        self.k_NMF_bases = k_NMF_bases
        self.n_iter = n_iter

    def NMF(self, Tn, Vn, Y_n):
        # Tn: (I, K), Vn: (K, J), Y_n: (I, J)
        Rn = Tn @ Vn # (I, J)
        Y_d_R = Y_n.abs() ** 2 / (Rn.abs() ** 2 + 1e-20)
        Rn_d = 1 / Rn.abs()
        Tn_new = Tn * torch.sqrt((Y_d_R @ Vn.T) / (Rn_d @ Vn.T))
        Rn = Tn_new @ Vn
        Vn_new = Vn * torch.sqrt((Tn_new.T @ Y_d_R) / (Tn_new.T @ Rn_d))
        Rn_new = Tn_new @ Vn_new
        return Tn_new, Vn_new, Rn_new

    def forward(self, X):
        M, J, I, _ = X.shape # channels, frequency bins, time frames, complex dimension
        N = self.n_components
        K = self.k_NMF_bases
        X = torch.view_as_complex(X.permute(2,0,1,3)) # (I, M, J)
        W = torch.stack([torch.eye(N, M) for _ in range(I)], dim=0)  # (I, N, M)
        Y = torch.einsum('inm,imj->inj', W, X)
        T = torch.rand(I, K, N) * (1 - 1e-8) + 1e-8  # uniform in [1e-8, 1)
        V = torch.rand(K, J, N) * (1 - 1e-8) + 1e-8  # uniform in [1e-8, 1)
        for l in range(self.n_iter):
            for n in range(N):
                # Extract Y_hat_n: magnitude spectrogram for source n
                Y_hat_n = Y[:, n, :]  # (I, J)
                T[:, :, n], V[:, :, n], Rn = self.NMF(T[:, :, n], V[:, :, n], Y_hat_n)
                e_n = torch.eye(M)[:, n]
                for i in range(I):
                    X_in_hat = X[i, :, :] / Rn[i, :].unsqueeze(0)  # (M, J)
                    D_in = X_in_hat @ X[i, :, :].T / J # (M, M)
                    b_in = torch.linalg.inv(W[i, :, :] @ D_in) @ e_n
                    W[i, :, n] = b_in.T / torch.sqrt((b_in.T @ D_in @ b_in).item())

            Y = torch.einsum('inm,imj->inj', W, X)
        return Y