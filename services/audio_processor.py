import librosa
import numpy as np
from pydub import AudioSegment
import tempfile
import os
from typing import List, Tuple

class AudioProcessor:
    """Handles audio file processing and feature extraction"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Load audio file and convert to numpy array
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio data as numpy array
        """
        try:
            # Load with librosa (handles most formats)
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            # Fallback: convert with pydub first
            audio_segment = AudioSegment.from_file(file_path)
            
            # Export to temporary wav file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                audio_segment.export(tmp.name, format='wav')
                tmp_path = tmp.name
            
            try:
                audio, sr = librosa.load(tmp_path, sr=self.sample_rate)
                return audio
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    
    def get_duration(self, audio: np.ndarray) -> float:
        """
        Get duration of audio in seconds
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Duration in seconds
        """
        return len(audio) / self.sample_rate
    
    def extract_intensity(self, audio: np.ndarray, frame_length=2048, hop_length=512) -> np.ndarray:
        """
        Extract intensity/energy from audio over time
        
        Args:
            audio: Audio data
            frame_length: Size of analysis window
            hop_length: Number of samples between windows
            
        Returns:
            Array of intensity values
        """
        # Calculate RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        # Normalize to 0-100 scale
        rms_normalized = (rms / np.max(rms)) * 100 if np.max(rms) > 0 else rms
        return rms_normalized
    
    def extract_pitch(self, audio: np.ndarray, fmin=80, fmax=400) -> np.ndarray:
        """
        Extract pitch/fundamental frequency from audio
        
        Args:
            audio: Audio data
            fmin: Minimum frequency to detect (Hz)
            fmax: Maximum frequency to detect (Hz)
            
        Returns:
            Array of pitch values in Hz
        """
        # Use piptrack to estimate pitch
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=self.sample_rate,
            fmin=fmin,
            fmax=fmax
        )
        
        # Extract the most prominent pitch at each time frame
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            pitch_values.append(pitch if pitch > 0 else 0)
        
        return np.array(pitch_values)
    
    def detect_silence(self, audio: np.ndarray, threshold_db=-40) -> List[Tuple[float, float]]:
        """
        Detect silent segments in audio
        
        Args:
            audio: Audio data
            threshold_db: Silence threshold in dB
            
        Returns:
            List of (start_time, end_time) tuples for silent segments
        """
        # Calculate frame-wise RMS energy
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms)
        
        # Find silent frames
        silent_frames = rms_db < threshold_db
        
        # Convert frame indices to time
        times = librosa.frames_to_time(
            np.arange(len(silent_frames)),
            sr=self.sample_rate,
            hop_length=hop_length
        )
        
        # Group consecutive silent frames
        silent_segments = []
        in_silence = False
        start_time = 0
        
        for i, is_silent in enumerate(silent_frames):
            if is_silent and not in_silence:
                start_time = times[i]
                in_silence = True
            elif not is_silent and in_silence:
                silent_segments.append((start_time, times[i]))
                in_silence = False
        
        # Close last segment if needed
        if in_silence:
            silent_segments.append((start_time, times[-1]))
        
        return silent_segments
    
    def normalize_audio(self, audio: np.ndarray, target_db=-20.0) -> np.ndarray:
        """
        Normalize audio to target loudness
        
        Args:
            audio: Audio data
            target_db: Target loudness in dB
            
        Returns:
            Normalized audio
        """
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio**2))
        
        # Calculate target RMS from dB
        target_rms = 10 ** (target_db / 20)
        
        # Apply gain
        gain = target_rms / rms if rms > 0 else 1
        return audio * gain