"""Audio preprocessor for input audio normalization and cleaning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from voice_agent.core import SAMPLE_RATE, AudioError
from voice_agent.utils import get_logger, pcm_to_numpy, numpy_to_pcm, Timer

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Audio preprocessing configuration."""

    # Filters
    high_pass_hz: float = 80.0      # Remove low freq rumble
    low_pass_hz: float = 7500.0     # Remove high freq noise
    
    # Normalization
    normalize: bool = True
    target_db: float = -20.0        # Target RMS level
    
    # Noise gate
    noise_gate_enabled: bool = True
    noise_gate_threshold_db: float = -45.0
    
    # DC offset removal
    remove_dc: bool = True
    
    # Sample rate
    sample_rate: int = SAMPLE_RATE


class AudioPreprocessor:
    """
    Audio preprocessor for cleaning and normalizing input audio.
    
    Pipeline:
        1. DC offset removal
        2. High-pass filter (remove rumble)
        3. Low-pass filter (remove high freq noise)
        4. Noise gate (suppress low-level noise)
        5. Normalization (consistent volume)
    
    Attributes:
        config: Preprocessing configuration
    """

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        """
        Initialize preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self._config = config or PreprocessConfig()
        self._hp_filter: np.ndarray | None = None
        self._lp_filter: np.ndarray | None = None
        self._initialized = False

    def _init_filters(self) -> None:
        """Initialize filter coefficients."""
        if self._initialized:
            return

        from scipy import signal

        sr = self._config.sample_rate
        nyq = sr / 2

        # High-pass filter
        if self._config.high_pass_hz > 0:
            hp_normalized = self._config.high_pass_hz / nyq
            self._hp_b, self._hp_a = signal.butter(
                4, hp_normalized, btype="high"
            )

        # Low-pass filter
        if self._config.low_pass_hz > 0 and self._config.low_pass_hz < nyq:
            lp_normalized = self._config.low_pass_hz / nyq
            self._lp_b, self._lp_a = signal.butter(
                4, lp_normalized, btype="low"
            )

        self._initialized = True
        logger.debug("preprocessor_initialized")

    def process(self, audio: bytes) -> bytes:
        """
        Process audio bytes through preprocessing pipeline.
        
        Args:
            audio: PCM S16LE audio bytes
            
        Returns:
            Processed PCM S16LE audio bytes
            
        Raises:
            AudioError: If processing fails
        """
        if len(audio) == 0:
            return audio

        try:
            with Timer() as t:
                # Convert to numpy
                audio_np = pcm_to_numpy(audio)
                
                # Process
                processed = self.process_numpy(audio_np)
                
                # Convert back
                result = numpy_to_pcm(processed)

            logger.debug(
                "preprocess_complete",
                latency_ms=round(t.elapsed_ms, 2),
                input_size=len(audio),
            )

            return result

        except Exception as e:
            logger.error("preprocess_failed", error=str(e))
            raise AudioError(f"Preprocessing failed: {e}") from e

    def process_numpy(self, audio_np: np.ndarray) -> np.ndarray:
        """
        Process numpy audio array through preprocessing pipeline.
        
        Args:
            audio_np: Float32 audio array [-1, 1]
            
        Returns:
            Processed float32 audio array
        """
        if len(audio_np) == 0:
            return audio_np

        self._init_filters()

        from scipy import signal

        processed = audio_np.copy()

        # 1. DC offset removal
        if self._config.remove_dc:
            processed = processed - np.mean(processed)

        # 2. High-pass filter
        if self._config.high_pass_hz > 0 and hasattr(self, "_hp_b"):
            processed = signal.filtfilt(self._hp_b, self._hp_a, processed)

        # 3. Low-pass filter
        if self._config.low_pass_hz > 0 and hasattr(self, "_lp_b"):
            processed = signal.filtfilt(self._lp_b, self._lp_a, processed)

        # 4. Noise gate
        if self._config.noise_gate_enabled:
            processed = self._apply_noise_gate(processed)

        # 5. Normalization
        if self._config.normalize:
            processed = self._normalize(processed)

        return processed.astype(np.float32)

    def _apply_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise gate to suppress low-level noise."""
        threshold_linear = 10 ** (self._config.noise_gate_threshold_db / 20)
        
        # Calculate envelope (simple RMS-based)
        frame_size = int(self._config.sample_rate * 0.01)  # 10ms frames
        
        if len(audio) < frame_size:
            # Too short, check whole signal
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < threshold_linear:
                return np.zeros_like(audio)
            return audio

        # Frame-based gating
        result = audio.copy()
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i + frame_size]
            rms = np.sqrt(np.mean(frame ** 2))
            if rms < threshold_linear:
                # Soft gate (reduce by 20dB instead of hard mute)
                result[i:i + frame_size] *= 0.1

        return result

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to target dB level."""
        if len(audio) == 0:
            return audio

        # Calculate current RMS
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-10:
            return audio

        # Target RMS
        target_rms = 10 ** (self._config.target_db / 20)

        # Scale
        gain = target_rms / rms
        
        # Limit gain to prevent over-amplification
        gain = min(gain, 10.0)  # Max 20dB gain

        normalized = audio * gain

        # Soft clip to prevent clipping
        normalized = np.tanh(normalized)

        return normalized


@dataclass(frozen=True, slots=True)
class PostprocessConfig:
    """Audio postprocessing configuration."""

    # Output normalization
    normalize: bool = True
    target_db: float = -16.0    # Slightly louder for playback
    
    # Soft limiter
    limiter_enabled: bool = True
    limiter_threshold_db: float = -3.0
    
    # Fade in/out
    fade_in_ms: float = 5.0
    fade_out_ms: float = 10.0
    
    # Sample rate
    sample_rate: int = SAMPLE_RATE


class AudioPostprocessor:
    """
    Audio postprocessor for TTS output preparation.
    
    Pipeline:
        1. Normalization
        2. Soft limiting (prevent clipping)
        3. Fade in/out (smooth transitions)
    
    Attributes:
        config: Postprocessing configuration
    """

    def __init__(self, config: PostprocessConfig | None = None) -> None:
        """
        Initialize postprocessor.
        
        Args:
            config: Postprocessing configuration
        """
        self._config = config or PostprocessConfig()

    def process(self, audio: bytes) -> bytes:
        """
        Process TTS output audio.
        
        Args:
            audio: PCM S16LE audio bytes
            
        Returns:
            Processed PCM S16LE audio bytes
        """
        if len(audio) == 0:
            return audio

        try:
            with Timer() as t:
                audio_np = pcm_to_numpy(audio)
                processed = self.process_numpy(audio_np)
                result = numpy_to_pcm(processed)

            logger.debug(
                "postprocess_complete",
                latency_ms=round(t.elapsed_ms, 2),
                input_size=len(audio),
            )

            return result

        except Exception as e:
            logger.error("postprocess_failed", error=str(e))
            # Return original on error
            return audio

    def process_numpy(self, audio_np: np.ndarray) -> np.ndarray:
        """
        Process numpy audio array.
        
        Args:
            audio_np: Float32 audio array
            
        Returns:
            Processed float32 audio array
        """
        if len(audio_np) == 0:
            return audio_np

        processed = audio_np.copy()

        # 1. Normalization
        if self._config.normalize:
            processed = self._normalize(processed)

        # 2. Soft limiting
        if self._config.limiter_enabled:
            processed = self._soft_limit(processed)

        # 3. Fade in/out
        processed = self._apply_fades(processed)

        return processed.astype(np.float32)

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize to target level."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-10:
            return audio

        target_rms = 10 ** (self._config.target_db / 20)
        gain = target_rms / rms
        gain = min(gain, 5.0)  # Limit gain

        return audio * gain

    def _soft_limit(self, audio: np.ndarray) -> np.ndarray:
        """Apply soft limiter to prevent harsh clipping."""
        threshold = 10 ** (self._config.limiter_threshold_db / 20)
        
        # Soft knee compression above threshold
        above_threshold = np.abs(audio) > threshold
        if np.any(above_threshold):
            # Apply soft saturation
            audio = np.sign(audio) * (
                threshold + (1 - threshold) * np.tanh(
                    (np.abs(audio) - threshold) / (1 - threshold)
                )
            ) * above_threshold + audio * ~above_threshold

        return audio

    def _apply_fades(self, audio: np.ndarray) -> np.ndarray:
        """Apply fade in and fade out."""
        sr = self._config.sample_rate
        
        # Fade in
        fade_in_samples = int(sr * self._config.fade_in_ms / 1000)
        if fade_in_samples > 0 and len(audio) > fade_in_samples:
            fade_in = np.linspace(0, 1, fade_in_samples)
            audio[:fade_in_samples] *= fade_in

        # Fade out
        fade_out_samples = int(sr * self._config.fade_out_ms / 1000)
        if fade_out_samples > 0 and len(audio) > fade_out_samples:
            fade_out = np.linspace(1, 0, fade_out_samples)
            audio[-fade_out_samples:] *= fade_out

        return audio


# === Singleton instances ===

_preprocessor: AudioPreprocessor | None = None
_postprocessor: AudioPostprocessor | None = None


def get_preprocessor(config: PreprocessConfig | None = None) -> AudioPreprocessor:
    """Get or create preprocessor singleton."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = AudioPreprocessor(config)
    return _preprocessor


def get_postprocessor(config: PostprocessConfig | None = None) -> AudioPostprocessor:
    """Get or create postprocessor singleton."""
    global _postprocessor
    if _postprocessor is None:
        _postprocessor = AudioPostprocessor(config)
    return _postprocessor
