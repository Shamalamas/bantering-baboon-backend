from pydantic import BaseModel
from typing import List, Dict, Any

class FillerWord(BaseModel):
    word: str
    count: int

class PacePoint(BaseModel):
    time: float
    words_per_minute: float

class EmphasisPoint(BaseModel):
    time: float
    intensity: float

class AnalysisResponse(BaseModel):
    duration: float
    wordCount: int
    avgPace: float
    paceData: List[PacePoint]
    fillerWords: List[FillerWord]
    emphasisData: List[EmphasisPoint]
    totalFillers: int
    transcript: str
