"""
MouthMap Inference Server
==========================
Production-ready FastAPI server for lip reading inference.

Uses ONNX Runtime for optimized inference on CPU.

Usage:
    uvicorn server.main:app --host 0.0.0.0 --port 8000

API Endpoints:
    POST /predict - Upload video and get transcription
    GET /health   - Health check
    GET /model/info - Model metadata
"""

import os
import time
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort


# =============================================================================
# Configuration
# =============================================================================

# Path to the ONNX model
# Note: FP16 model can cause segfault on some CPUs with BiLSTM layers
# Use baseline FP32 for maximum compatibility
MODEL_PATH = os.getenv(
    "MODEL_PATH", 
    "models/onnx/lipnet_baseline.onnx"
)

# Video preprocessing constants (must match training)
FRAME_COUNT = 75
FRAME_HEIGHT = 46
FRAME_WIDTH = 140

# Character vocabulary for CTC decoding
VOCAB = " abcdefghijklmnopqrstuvwxyz'?!123456789"


# =============================================================================
# Response Models
# =============================================================================

class PredictionResponse(BaseModel):
    """Response for prediction endpoint."""
    transcription: str
    confidence: float
    latency_ms: float
    model_version: str


class HealthResponse(BaseModel):
    """Response for health check."""
    status: str
    model_loaded: bool
    model_size_mb: float


class ModelInfoResponse(BaseModel):
    """Response for model info endpoint."""
    input_name: str
    input_shape: list
    output_shape: list
    providers: list


# =============================================================================
# Model Manager
# =============================================================================

class ModelManager:
    """Manages ONNX model lifecycle."""
    
    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.model_path: str = ""
        self.model_size_mb: float = 0.0
        
    def load(self, model_path: str):
        """Load ONNX model with optimizations."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Configure session options for performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # Adjust based on CPU
        
        # Create inference session (CPU only for portability)
        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.model_path = model_path
        self.model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data."""
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        outputs = self.session.run(None, {self.input_name: input_data})
        return outputs[0]


# Global model manager
model_manager = ModelManager()


# =============================================================================
# Video Preprocessing
# =============================================================================

def preprocess_video(video_path: str) -> np.ndarray:
    """
    Extract and preprocess frames from video.
    
    Steps:
        1. Read video frames
        2. Convert to grayscale
        3. Resize to (FRAME_HEIGHT, FRAME_WIDTH)
        4. Normalize to [0, 1]
        5. Pad/truncate to FRAME_COUNT frames
    
    Returns:
        np.ndarray of shape (1, FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH, 1)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < FRAME_COUNT:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to expected dimensions
        resized = cv2.resize(gray, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        frames.append(normalized)
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError("Could not read any frames from video")
    
    # Pad with last frame if video is too short
    while len(frames) < FRAME_COUNT:
        frames.append(frames[-1])
    
    # Truncate if too long
    frames = frames[:FRAME_COUNT]
    
    # Reshape to (1, 75, 46, 140, 1)
    return np.array(frames).reshape(
        1, FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH, 1
    ).astype(np.float32)


# =============================================================================
# CTC Decoding
# =============================================================================

def decode_ctc_greedy(predictions: np.ndarray) -> tuple:
    """
    Greedy CTC decoding.
    
    Args:
        predictions: Shape (1, FRAME_COUNT, vocab_size)
    
    Returns:
        (transcription, confidence)
    """
    # Get most likely character at each timestep
    char_indices = np.argmax(predictions[0], axis=-1)
    confidences = np.max(predictions[0], axis=-1)
    
    # Merge repeated characters and remove blanks (index 0)
    result = []
    result_confidences = []
    prev = None
    
    for idx, conf in zip(char_indices, confidences):
        if idx != prev and idx != 0:  # 0 is blank
            if idx - 1 < len(VOCAB):
                result.append(VOCAB[idx - 1])
                result_confidences.append(float(conf))
        prev = idx
    
    transcription = ''.join(result).strip()
    avg_confidence = float(np.mean(result_confidences)) if result_confidences else 0.0
    
    return transcription, avg_confidence


# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model_manager.load(MODEL_PATH)
        print(f"Model loaded successfully! Size: {model_manager.model_size_mb:.2f} MB")
    except Exception as e:
        print(f"Warning: Failed to load model: {e}")
    yield
    print("Shutting down...")


app = FastAPI(
    title="MouthMap Lip Reading API",
    description="Production inference API for visual speech recognition",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to docs."""
    return {"message": "MouthMap API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for deployment monitoring."""
    return HealthResponse(
        status="healthy" if model_manager.session else "degraded",
        model_loaded=model_manager.session is not None,
        model_size_mb=model_manager.model_size_mb
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model metadata."""
    if model_manager.session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    session = model_manager.session
    return ModelInfoResponse(
        input_name=session.get_inputs()[0].name,
        input_shape=list(session.get_inputs()[0].shape),
        output_shape=list(session.get_outputs()[0].shape),
        providers=session.get_providers()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(video: UploadFile = File(...)):
    """
    Process uploaded video and return transcription.
    
    Accepts: video file (mp4, avi, webm, mpg)
    Returns: transcription with confidence and latency metrics
    """
    if model_manager.session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    allowed_types = ["video/mp4", "video/avi", "video/webm", "video/mpeg"]
    if video.content_type and not any(t in video.content_type for t in ["video", "application/octet-stream"]):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type: {video.content_type}. Expected video file."
        )
    
    # Save uploaded file temporarily
    suffix = os.path.splitext(video.filename)[1] if video.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        start_time = time.perf_counter()
        
        # Preprocess video
        input_data = preprocess_video(tmp_path)
        
        # Run inference
        predictions = model_manager.predict(input_data)
        
        # Decode output
        transcription, confidence = decode_ctc_greedy(predictions)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return PredictionResponse(
            transcription=transcription,
            confidence=confidence,
            latency_ms=round(latency_ms, 1),
            model_version="v1.0-fp16"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
