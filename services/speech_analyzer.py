import speech_recognition as sr
from typing import List, Dict
import re
import numpy as np

class SpeechAnalyzer:
    def transcribe(self, file_path: str) -> str:
        """
        Transcribe audio file to text
        """
        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio)
            return transcript
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Could not request results"

    def analyze_pace(self, transcript: str, duration: float) -> List[Dict]:
        """
        Analyze speaking pace over time
        """
        words = transcript.split()
        word_count = len(words)

        if duration <= 0 or word_count == 0:
            return [{"time": 0, "words_per_minute": 0}]

        # Create pace points at regular intervals
        num_points = min(10, max(1, int(duration / 10)))  # Up to 10 points, at least 1
        pace_points = []

        for i in range(num_points):
            time_point = (i * duration) / (num_points - 1) if num_points > 1 else 0
            # Estimate words per minute at this time point
            # This is a simplified approximation - in a real implementation,
            # you'd use word timestamps from the transcription
            words_per_minute = (word_count / duration) * 60
            pace_points.append({"time": round(time_point, 1), "words_per_minute": round(words_per_minute, 1)})

        return pace_points

    def detect_fillers(self, transcript: str) -> List[Dict]:
        """
        Detect filler words
        """
        filler_words = ["um", "uh", "like", "you know", "so"]
        fillers = []
        for word in filler_words:
            count = len(re.findall(r'\b' + re.escape(word) + r'\b', transcript.lower()))
            if count > 0:
                fillers.append({"word": word, "count": count})
        return fillers

    def analyze_emphasis(self, audio_data, duration: float) -> List[Dict]:
        """
        Analyze emphasis points based on audio intensity
        """
        # Use audio processor to extract intensity
        intensity = self.audio_processor.extract_intensity(audio_data)

        # Convert to time points
        times = np.linspace(0, duration, len(intensity))

        # Find peaks in intensity as emphasis points
        emphasis_points = []
        threshold = np.mean(intensity) + np.std(intensity)  # Above average + std

        for i, (time, intens) in enumerate(zip(times, intensity)):
            if intens > threshold:
                emphasis_points.append({"time": float(time), "intensity": float(intens / 100)})  # Normalize to 0-1

        # If no emphasis found, return a default
        if not emphasis_points:
            emphasis_points = [{"time": duration / 2, "intensity": 0.5}]

        return emphasis_points
