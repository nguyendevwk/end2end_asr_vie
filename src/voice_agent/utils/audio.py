"""Audio processing utilities."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from voice_agent.core import SAMPLE_RATE, AudioError


def pcm_to_numpy(
    audio_bytes: bytes,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Convert PCM S16LE bytes to numpy float32 array.

    Args:
        audio_bytes: PCM S16LE audio bytes
        sample_rate: Sample rate (unused, for API compatibility)

    Returns:
        Numpy array of float32 samples normalized to [-1, 1]

    Raises:
        AudioError: If conversion fails
    """
    if len(audio_bytes) == 0:
        return np.array([], dtype=np.float32)

    if len(audio_bytes) % 2 != 0:
        raise AudioError("PCM S16LE audio must have even number of bytes")

    try:
        # Convert S16LE to float32
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio_np.astype(np.float32) / 32768.0
    except Exception as e:
        raise AudioError(f"Failed to convert PCM to numpy: {e}") from e


def numpy_to_pcm(
    audio_np: np.ndarray,
) -> bytes:
    """
    Convert numpy float32 array to PCM S16LE bytes.

    Args:
        audio_np: Numpy array of float32 samples [-1, 1]

    Returns:
        PCM S16LE audio bytes

    Raises:
        AudioError: If conversion fails
    """
    if len(audio_np) == 0:
        return b""

    try:
        # Clip and convert to S16LE
        audio_clipped = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        return audio_int16.tobytes()
    except Exception as e:
        raise AudioError(f"Failed to convert numpy to PCM: {e}") from e


def resample(
    audio_np: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Resample audio to target sample rate.

    Uses polyphase filtering (fast, high quality for speech) with
    FFT fallback for non-integer ratios.

    Args:
        audio_np: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array

    Raises:
        AudioError: If resampling fails
    """
    if orig_sr == target_sr:
        return audio_np

    if len(audio_np) == 0:
        return audio_np

    try:
        import math

        import scipy.signal

        # Polyphase is ~10x faster than FFT resample for integer ratios
        # e.g. 24000 -> 16000 = 3:2 (down=3, up=2)
        gcd = math.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd

        # Polyphase works well for small integer ratios (typical SR conversions)
        if up <= 100 and down <= 100:
            return scipy.signal.resample_poly(audio_np, up, down).astype(np.float32)

        # Fallback to FFT method for exotic ratios
        num_samples = int(len(audio_np) * target_sr / orig_sr)
        return scipy.signal.resample(audio_np, num_samples).astype(np.float32)
    except Exception as e:
        raise AudioError(f"Failed to resample audio: {e}") from e


def get_audio_duration_ms(
    audio_bytes: bytes,
    sample_rate: int = SAMPLE_RATE,
) -> float:
    """
    Calculate audio duration in milliseconds.

    Args:
        audio_bytes: PCM S16LE audio bytes
        sample_rate: Sample rate

    Returns:
        Duration in milliseconds
    """
    if len(audio_bytes) == 0:
        return 0.0

    # PCM S16LE: 2 bytes per sample
    num_samples = len(audio_bytes) // 2
    return (num_samples / sample_rate) * 1000


def load_audio_file(
    file_path: str,
    target_sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """
    Load audio file and convert to target sample rate.

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate

    Returns:
        Tuple of (audio_array, sample_rate)

    Raises:
        AudioError: If loading fails
    """
    try:
        audio, sr = sf.read(file_path, dtype="float32")

        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != target_sr:
            audio = resample(audio, sr, target_sr)

        return audio, target_sr
    except Exception as e:
        raise AudioError(f"Failed to load audio file: {e}") from e


def save_audio_file(
    file_path: str,
    audio_np: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """
    Save numpy audio array to file.

    Args:
        file_path: Output file path
        audio_np: Audio array
        sample_rate: Sample rate

    Raises:
        AudioError: If saving fails
    """
    try:
        sf.write(file_path, audio_np, sample_rate)
    except Exception as e:
        raise AudioError(f"Failed to save audio file: {e}") from e


def audio_to_wav_bytes(
    audio_np: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """
    Convert numpy array to WAV bytes.

    Args:
        audio_np: Audio array
        sample_rate: Sample rate

    Returns:
        WAV file bytes
    """
    buffer = io.BytesIO()
    sf.write(buffer, audio_np, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


class AudioBuffer:
    """
    Efficient audio buffer with fixed maximum duration.

    Uses deque for O(1) append and popleft operations.
    """

    __slots__ = ("_buffer", "_dropped", "_sample_rate", "_total_bytes")

    def __init__(
        self,
        max_duration_ms: int,
        sample_rate: int = SAMPLE_RATE,
        chunk_ms: int = 100,
    ) -> None:
        """
        Initialize audio buffer.

        Args:
            max_duration_ms: Maximum buffer duration in milliseconds
            sample_rate: Audio sample rate
            chunk_ms: Nominal chunk duration (only for max_chunks estimate;
                actual duration is computed from real bytes, since WebSocket
                chunks vary, e.g. browser sends 2048 samples = 128ms)
        """
        from collections import deque

        self._sample_rate = sample_rate

        # Calculate max chunks
        max_chunks = max_duration_ms // chunk_ms
        self._buffer: deque[bytes] = deque(maxlen=max_chunks)
        self._total_bytes = 0
        self._dropped = 0  # chunks evicted by overflow (backpressure signal)

    def add(self, chunk: bytes) -> None:
        """Add audio chunk to buffer."""
        # deque with maxlen silently drops oldest; keep byte counter accurate
        if len(self._buffer) == self._buffer.maxlen:
            oldest = self._buffer[0]
            self._total_bytes -= len(oldest)
            self._dropped += 1
        self._buffer.append(chunk)
        self._total_bytes += len(chunk)

    def get_all(self) -> bytes:
        """Get all buffered audio as single bytes object."""
        return b"".join(self._buffer)

    def get_numpy(self) -> np.ndarray:
        """Get all buffered audio as numpy array."""
        return pcm_to_numpy(self.get_all(), self._sample_rate)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._total_bytes = 0
        self._dropped = 0

    @property
    def dropped_chunks(self) -> int:
        """Chunks evicted by overflow since last clear."""
        return self._dropped

    @property
    def duration_ms(self) -> float:
        """Current buffer duration in milliseconds (computed from real bytes)."""
        num_samples = self._total_bytes // 2  # PCM S16LE
        return (num_samples / self._sample_rate) * 1000 if self._sample_rate else 0.0

    @property
    def chunk_count(self) -> int:
        """Number of chunks in buffer."""
        return len(self._buffer)

    def __len__(self) -> int:
        """Number of chunks in buffer."""
        return len(self._buffer)

    def __bool__(self) -> bool:
        """True if buffer has data."""
        return len(self._buffer) > 0
