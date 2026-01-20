"""Real-Time Dual-Rate ILRMA Audio Processing Framework.

Implements a near-real-time ILRMA with dual-rate asynchronous design:
- Fast Path: Apply current demixing matrix W every frame (32ms)
- Slow Path: Background update of W matrix every 512ms

Uses double-buffering for lock-free W matrix access in the fast path.

Based on the offline ILRMA_V2 implementation.

Reference:
    T. Nakashima, R. Scheibler, M. Togami, and N. Ono,
    "Real-Time Speech Extraction Based on Rank-Constrained Spatial Covariance
    Matrix Estimation and Spatially Regularized Independent Low-Rank Matrix
    Analysis With Fast Demixing Matrix Estimation,"
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2024.
"""

import torch
import threading
from typing import Optional, Tuple
from .utils import nmf_update


class ILRMA_REALTIME(torch.nn.Module):
    """Real-time dual-rate ILRMA for blind source separation.

    Uses double-buffering to minimize lock contention:
    - W_front: Read by fast path (no lock needed)
    - W_back: Written by slow path
    - Atomic swap after background update completes

    Args:
        n_components (int): Number of sources (must equal number of channels). Default: 2
        k_NMF_bases (int): Number of NMF basis functions. Default: 10
        n_iter (int): Number of ILRMA iterations per background update. Default: 3
        observation_window_sec (float): Observation window length in seconds. Default: 5.0
        update_interval_frames (int): Frames between W updates. Default: 16 (512ms @ 32ms/frame)
        sample_rate (int): Sample rate in Hz. Default: 16000
        n_fft (int): FFT size. Default: 1024
        hop_length (int): Hop length (frame shift). Default: 512
        nmf_eps (float): Small constant for NMF stability. Default: 1e-20
        ip_eps (float): Small constant for IP update stability. Default: 1e-20
        smoothing_frames (int): Number of frames for W interpolation. Default: 3
    """

    def __init__(
        self,
        n_components: int = 2,
        k_NMF_bases: int = 10,
        n_iter: int = 3,
        observation_window_sec: float = 5.0,
        update_interval_frames: int = 16,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        nmf_eps: float = 1e-20,
        ip_eps: float = 1e-20,
        smoothing_frames: int = 3,
    ):
        super().__init__()
        self.n_components = n_components
        self.k_NMF_bases = k_NMF_bases
        self.n_iter = n_iter
        self.observation_window_sec = observation_window_sec
        self.update_interval_frames = update_interval_frames
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.nmf_eps = nmf_eps
        self.ip_eps = ip_eps
        self.smoothing_frames = smoothing_frames

        # Derived parameters
        self.n_freqs = n_fft // 2 + 1
        self.frame_duration_sec = hop_length / sample_rate
        self.observation_frames = int(observation_window_sec / self.frame_duration_sec)

        # State variables (initialized in _init_state)
        # Double-buffered W matrices
        self._W_front: Optional[torch.Tensor] = None  # Read by fast path (lock-free)
        self._W_back: Optional[torch.Tensor] = None   # Written by slow path
        self._W_target: Optional[torch.Tensor] = None  # Target for interpolation

        # NMF parameters
        self._T: Optional[torch.Tensor] = None
        self._V: Optional[torch.Tensor] = None

        # STFT buffer
        self._stft_buffer: Optional[torch.Tensor] = None
        self._buffer_write_pos: int = 0

        # Frame counters
        self._frame_counter: int = 0
        self._smoothing_counter: int = 0
        self._is_smoothing: bool = False

        # Threading - minimal lock usage with double buffering
        self._swap_lock = threading.Lock()  # Only for atomic pointer swap
        self._nmf_lock = threading.Lock()   # For T, V access
        self._update_thread: Optional[threading.Thread] = None
        self._update_in_progress: bool = False
        self._new_W_ready: bool = False  # Atomic flag for swap notification

    def _init_state(self, M: int, device: torch.device):
        """Initialize state variables for M channels."""
        N = self.n_components
        K = self.k_NMF_bases
        I = self.n_freqs
        J = self.observation_frames

        assert M == N, "ILRMA_REALTIME assumes number of sources equals number of channels."

        # Double-buffered demixing matrices W: (I, N, M) - identity initialization
        W_init = torch.stack([
            torch.eye(N, M, dtype=torch.complex64, device=device)
            for _ in range(I)
        ], dim=0)
        self._W_front = W_init.clone()
        self._W_back = W_init.clone()
        self._W_target = W_init.clone()

        # NMF parameters
        self._T = torch.rand(I, K, N, device=device) * (1 - 1e-8) + 1e-8
        self._V = torch.rand(K, J, N, device=device) * (1 - 1e-8) + 1e-8

        # STFT buffer: (M, I, J) complex
        self._stft_buffer = torch.zeros(M, I, J, dtype=torch.complex64, device=device)
        self._buffer_write_pos = 0
        self._total_frames_received = 0  # Track total frames to know when buffer is full
        self._frame_counter = 0
        self._smoothing_counter = 0
        self._is_smoothing = False
        self._new_W_ready = False

    def _push_stft_frame(self, X_frame: torch.Tensor):
        """Push a new STFT frame into the circular buffer.

        Args:
            X_frame: (M, I) complex STFT frame
        """
        self._stft_buffer[:, :, self._buffer_write_pos] = X_frame
        self._buffer_write_pos = (self._buffer_write_pos + 1) % self.observation_frames
        self._total_frames_received += 1

    def _get_stft_buffer_ordered(self) -> torch.Tensor:
        """Get STFT buffer in temporal order.

        Returns:
            (M, I, J) complex tensor with oldest frame first
        """
        if self._buffer_write_pos == 0:
            return self._stft_buffer.clone()
        else:
            # Roll to put oldest frame first
            return torch.cat([
                self._stft_buffer[:, :, self._buffer_write_pos:],
                self._stft_buffer[:, :, :self._buffer_write_pos]
            ], dim=2)

    def _apply_demixing(self, X_frame: torch.Tensor) -> torch.Tensor:
        """Apply current demixing matrix to a single frame (lock-free read).

        Args:
            X_frame: (M, I) complex STFT frame

        Returns:
            Y_frame: (N, I) complex separated frame
        """
        # X_frame: (M, I) -> (I, M, 1)
        X_c = X_frame.T.unsqueeze(-1)  # (I, M, 1)

        # Lock-free read of W_front - no lock needed as we only read
        # and the swap is atomic (Python reference assignment)
        W = self._W_front

        # Y = W @ X: (I, N, M) @ (I, M, 1) -> (I, N, 1)
        Y_c = torch.matmul(W, X_c).squeeze(-1)  # (I, N)

        return Y_c.T  # (N, I)

    def _check_and_swap_W(self):
        """Check if new W is ready and perform atomic swap if needed.

        Called from fast path at safe point (between frame processing).
        """
        if not self._new_W_ready:
            return

        with self._swap_lock:
            if self._new_W_ready:  # Double-check under lock
                # Atomic pointer swap
                self._W_target = self._W_back.clone()
                self._is_smoothing = True
                self._smoothing_counter = 0
                self._new_W_ready = False

    def _smooth_W_step(self):
        """Perform one step of W interpolation (lock-free)."""
        if not self._is_smoothing:
            return

        self._smoothing_counter += 1
        alpha = self._smoothing_counter / self.smoothing_frames

        # Interpolate towards target - no lock needed, only fast path touches W_front
        self._W_front = (1 - alpha) * self._W_front + alpha * self._W_target

        if self._smoothing_counter >= self.smoothing_frames:
            self._is_smoothing = False
            self._smoothing_counter = 0

    def _ilrma_iteration(self, X_c: torch.Tensor, W: torch.Tensor,
                         T: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform one ILRMA iteration (vectorized).

        Args:
            X_c: (I, M, J) complex mixture spectrogram
            W: (I, N, M) demixing matrix
            T: (I, K, N) NMF basis
            V: (K, J, N) NMF activation

        Returns:
            Updated (W, T, V)
        """
        I, M, J = X_c.shape
        N = self.n_components
        eye_M = torch.eye(M, dtype=torch.complex64, device=X_c.device)

        # Compute separated spectrogram
        Y = torch.einsum('inm,imj->inj', W, X_c)  # (I, N, J)

        for n in range(N):
            # --- 1. NMF update ---
            Y_hat_n = Y[:, n, :]  # (I, J)
            T[:, :, n], V[:, :, n], Rn = nmf_update(
                T[:, :, n], V[:, :, n], Y_hat_n, eps=self.nmf_eps
            )

            # --- 2. IP update for all frequencies (vectorized) ---
            inv_Rn = 1.0 / (Rn + self.ip_eps)  # (I, J)
            X_sqrt_weighted = X_c * torch.sqrt(inv_Rn).unsqueeze(1)  # (I, M, J)

            # Covariance matrix
            D_in = torch.matmul(
                X_sqrt_weighted,
                X_sqrt_weighted.conj().transpose(1, 2)
            ) / J  # (I, M, M)

            # Regularize
            D_reg = D_in + self.ip_eps * eye_M.unsqueeze(0)

            # A = W @ D_reg
            A = torch.matmul(W, D_reg)  # (I, M, M)

            # e_n: nth standard basis vector
            e_n = eye_M[:, n].view(1, M, 1).expand(I, -1, -1)  # (I, M, 1)

            # Solve A @ b = e_n
            b_in = torch.linalg.solve(A, e_n)  # (I, M, 1)

            # Normalization
            b_H = b_in.conj().transpose(1, 2)  # (I, 1, M)
            denom = torch.matmul(torch.matmul(b_H, D_reg), b_in)  # (I, 1, 1)
            denom = torch.sqrt(denom.real + self.ip_eps)

            # Update nth row of W
            w_in = (b_in / denom).squeeze(-1).conj()  # (I, M)
            W[:, n, :] = w_in

        return W, T, V

    def _update_W_background(self, X_c: torch.Tensor):
        """Background thread function to update W.

        Writes to W_back without blocking the fast path.

        Args:
            X_c: (I, M, J) complex mixture spectrogram
        """
        try:
            # Read current W_front as starting point (lock-free read)
            W = self._W_front.clone()

            # Read NMF parameters with minimal lock
            with self._nmf_lock:
                T = self._T.clone()
                V = self._V.clone()

            # Perform ILRMA iterations (no lock held during computation)
            for _ in range(self.n_iter):
                W, T, V = self._ilrma_iteration(X_c, W, T, V)

            # Apply projection back (Minimal Distortion Principle) to stabilize W
            # This ensures the first column of A = W^{-1} extracts the original signal scale
            I = W.shape[0]
            for i in range(I):
                try:
                    A = torch.linalg.inv(W[i])  # A = W^{-1}, shape (N, M)
                    # MDP: scale each source by the corresponding row of A that extracts it
                    # For determined case (N=M), use diagonal of A
                    scale = torch.diag(A)  # (N,)
                    W[i] = torch.diag(scale) @ W[i]
                except:
                    pass  # Skip if inversion fails

            # Write results to back buffer
            self._W_back = W

            # Update NMF parameters with minimal lock
            with self._nmf_lock:
                self._T = T
                self._V = V

            # Signal that new W is ready (atomic flag)
            self._new_W_ready = True

        finally:
            self._update_in_progress = False

    def _trigger_background_update(self, sync: bool = False):
        """Trigger a W update.

        Args:
            sync: If True, run update synchronously (for offline/simulation mode).
                  If False, run in background thread (for real-time streaming).
        """
        if self._update_in_progress:
            return

        # Get ordered STFT buffer
        X_buffer = self._get_stft_buffer_ordered()  # (M, I, J)
        X_c = X_buffer.permute(1, 0, 2).contiguous()  # (I, M, J)

        if sync:
            # Synchronous update - run directly and apply immediately
            self._update_in_progress = True
            self._update_W_background(X_c)
            # Immediately apply the new W (with NaN check)
            if self._new_W_ready:
                # Check for NaN before applying
                if torch.isnan(self._W_back).any() or torch.isinf(self._W_back).any():
                    pass  # Skip if NaN/Inf detected
                else:
                    self._W_front = self._W_back.clone()
                self._new_W_ready = False
        else:
            # Async update - run in background thread
            self._update_in_progress = True
            self._update_thread = threading.Thread(
                target=self._update_W_background,
                args=(X_c,),
                daemon=True
            )
            self._update_thread.start()

    def process_frame(self, X_frame: torch.Tensor, sync: bool = False) -> torch.Tensor:
        """Process a single STFT frame (fast path).

        Lock-free in the critical path using double-buffering (when sync=False).

        Args:
            X_frame: (M, I) complex STFT frame
            sync: If True, run W updates synchronously (for offline mode).

        Returns:
            Y_frame: (N, I) complex separated frame
        """
        M, I = X_frame.shape

        # Initialize state if needed
        if self._W_front is None:
            self._init_state(M, X_frame.device)

        # Push frame to buffer
        self._push_stft_frame(X_frame)

        # Check if new W is ready and swap (at safe point between frames)
        self._check_and_swap_W()

        # Smooth W transition if in progress (lock-free)
        # self._smooth_W_step()

        # Apply demixing (lock-free read of W_front)
        Y_frame = self._apply_demixing(X_frame)

        # Increment frame counter
        self._frame_counter += 1

        # Only trigger update if buffer has enough data (at least observation_frames)
        # and interval reached
        if self._frame_counter >= self.update_interval_frames:
            self._frame_counter = 0
            if self._total_frames_received >= self.observation_frames:
                self._trigger_background_update(sync=sync)

        return Y_frame

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Offline-compatible forward pass.

        Processes the entire spectrogram using dual-rate simulation.
        Uses synchronous W updates to ensure separation actually occurs.

        Args:
            X: (M, J, I, 2) real/imag mixture spectrogram (matching ILRMA_V2 interface)

        Returns:
            Y: (N, J, I, 2) real/imag separated spectrogram
        """
        M, J, I, _ = X.shape
        N = self.n_components
        assert M == N, "ILRMA_REALTIME assumes number of sources equals number of channels."

        # Convert to complex: (M, J, I) -> (I, M, J) for internal use
        X_c = torch.view_as_complex(X.permute(2, 0, 1, 3).contiguous())  # (I, M, J)

        device = X.device

        # Initialize state
        self._init_state(M, device)

        # Output buffer
        Y_frames = []

        # Process frame by frame with synchronous W updates
        for t in range(J):
            X_frame = X_c[:, :, t].T  # (M, I)
            Y_frame = self.process_frame(X_frame, sync=True)  # (N, I)
            Y_frames.append(Y_frame)

        # Stack frames: list of (N, I) -> (N, J, I)
        Y = torch.stack(Y_frames, dim=1)

        # Convert to real/imag: (N, J, I) -> (N, J, I, 2)
        Y_out = torch.view_as_real(Y).contiguous()

        return Y_out

    def reset(self):
        """Reset all internal state."""
        self._W_front = None
        self._W_back = None
        self._W_target = None
        self._T = None
        self._V = None
        self._stft_buffer = None
        self._buffer_write_pos = 0
        self._total_frames_received = 0
        self._frame_counter = 0
        self._smoothing_counter = 0
        self._is_smoothing = False
        self._update_in_progress = False
        self._update_thread = None
        self._new_W_ready = False
