import torch
from src.bss import ILRMA_V2
from src.bss.utils import select_target_index

class RCSCME(torch.nn.Module):
    def __init__(self, n_components=2, k_NMF_bases=8, n_iter=30, sr=16000, n_fft=512):
        super(RCSCME, self).__init__()
        self.n_components = n_components
        self.k_NMF_bases = k_NMF_bases
        self.n_iter = n_iter
        self.sr = sr
        self.n_fft = n_fft
        self.ilrma_v2 = ILRMA_V2(n_components=n_components, k_NMF_bases=k_NMF_bases, n_iter=n_iter)

    def forward(self, X):
        # X: (M, J, I, 2) - mixture spectrogram
        Y = self.ilrma_v2(X)
        Y_target_index, Y_target = select_target_index(Y, sr=self.sr, n_fft=self.n_fft)
        return Y, Y_target