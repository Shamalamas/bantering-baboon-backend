from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import os
import tempfile
from typing import Dict, List
import json

from services.audio_processor import AudioProcessor
from services.speech_analyzer import SpeechAnalyzer
from services.pdf_generator import PDFGenerator
from models.schemas import AnalysisResponse, FillerWord, PacePoint, EmphasisPoint

app = FastAPI(
    title="SpeakPace API",
    description="Speech analysis API for Bantering Baboon",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
audio_processor = AudioProcessor()
speech_analyzer = SpeechAnalyzer()
pdf_generator = PDFGenerator()

@app.get("/")
async def root():
    return {
        "message": "🎙️ Welcome to SpeakPace API - Bantering Baboon",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/api/analyze",
            "report": "/api/report"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "SpeakPace API"}

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze uploaded audio file for speech patterns
    """
    # Validate file type
    allowed_types = ["audio/mpeg", "audio/wav", "audio/mp3", "audio/m4a", "audio/x-m4a"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )

    # Read file contents
    contents = await file.read()

    # Validate file size (max 50MB)
    max_file_size = 50 * 1024 * 1024  # 50MB
    if len(contents) > max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {max_file_size // (1024*1024)}MB"
        )

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Process audio
        audio_data = audio_processor.load_audio(tmp_path)
        duration = audio_processor.get_duration(audio_data)

        # Perform speech analysis
        transcript = speech_analyzer.transcribe(tmp_path)
        pace_data = speech_analyzer.analyze_pace(transcript, duration)
        filler_words = speech_analyzer.detect_fillers(transcript)
        emphasis_data = speech_analyzer.analyze_emphasis(audio_data, duration)

        # Calculate statistics
        word_count = len(transcript.split())
        avg_pace = (word_count / duration) * 60 if duration > 0 else 0
        total_fillers = sum(fw['count'] for fw in filler_words)

        # Create response
        response = AnalysisResponse(
            duration=duration,
            wordCount=word_count,
            avgPace=round(avg_pace, 1),
            paceData=pace_data,
            fillerWords=filler_words,
            emphasisData=emphasis_data,
            totalFillers=total_fillers,
            transcript=transcript
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/api/report")
async def generate_report(analysis_data: AnalysisResponse):
    """
    Generate PDF report from analysis data
    """
    try:
        pdf_path = pdf_generator.create_report(analysis_data.dict())
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename="speakpace_report.pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
