# inference/deployment/api.py
import os
import json
import tempfile
import uuid
from typing import Dict, List, Optional, Union

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
from pydantic import BaseModel

from ..inference import DeepfakeDetector
from ..ensemble_inference import EnsembleDetector
from ..video_inference import VideoDetector
from .utils import load_deployment_config


class PredictionResponse(BaseModel):
    """Model for prediction response"""
    prediction: str
    probability: float
    confidence: float
    faces: int
    processing_time: float


class VideoAnalysisResponse(BaseModel):
    """Model for video analysis response"""
    prediction: str
    confidence: str
    real_frames: Optional[int] = None
    fake_frames: Optional[int] = None
    real_percent: Optional[float] = None
    fake_percent: Optional[float] = None
    frames_processed: int
    frames_with_faces: int
    processing_time: float
    analysis_video_id: Optional[str] = None


def create_api(config_path: str) -> FastAPI:
    """
    Create a FastAPI application for deepfake detection
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        FastAPI application
    """
    # Load configuration
    config = load_deployment_config(config_path)
    
    # Create API
    app = FastAPI(
        title="Deepfake Detection API",
        description="API for detecting deepfakes in images and videos",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Create output directory for results
    output_dir = config.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create static files directory for serving results
    static_dir = os.path.join(output_dir, "static")
    os.makedirs(static_dir, exist_ok=True)
    
    # Mount static files directory
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # Create detector based on configuration
    if "ensemble" in config and config.get("use_ensemble", False):
        # Create ensemble detector
        detector = EnsembleDetector(config_path=config_path)
    else:
        # Create single model detector
        detector = DeepfakeDetector(
            model_type=config["model"]["type"],
            model_path=config["model"]["checkpoint"],
            device=config.get("device", None),
            face_detector=config.get("face_detector", True),
            config=config
        )
    
    # Create video detector
    video_detector = VideoDetector(
        detector=detector,
        batch_size=config.get("video", {}).get("batch_size", 16),
        sample_rate=config.get("video", {}).get("sample_rate", 30),
        device=config.get("device", None)
    )
    
    # Define background tasks
    background_tasks = {}
    
    # Define routes
    @app.get("/", response_class=HTMLResponse)
    async def read_root():
        """Root endpoint"""
        return """
        <html>
            <head>
                <title>Deepfake Detection API</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 40px;
                        line-height: 1.6;
                    }
                    h1 {
                        color: #333;
                    }
                    h2 {
                        color: #555;
                    }
                    code {
                        background-color: #f4f4f4;
                        padding: 2px 5px;
                        border-radius: 3px;
                    }
                    pre {
                        background-color: #f4f4f4;
                        padding: 10px;
                        border-radius: 5px;
                    }
                    .endpoint {
                        margin-bottom: 30px;
                    }
                </style>
            </head>
            <body>
                <h1>Deepfake Detection API</h1>
                <p>This API provides endpoints for detecting deepfakes in images and videos.</p>
                
                <h2>Available Endpoints:</h2>
                
                <div class="endpoint">
                    <h3>1. Image Detection</h3>
                    <code>POST /detect/image</code>
                    <p>Upload an image for deepfake detection.</p>
                    <pre>
curl -X 'POST' \\
  'http://localhost:8000/detect/image' \\
  -H 'accept: application/json' \\
  -H 'Content-Type: multipart/form-data' \\
  -F 'file=@your_image.jpg'
                    </pre>
                </div>
                
                <div class="endpoint">
                    <h3>2. Video Detection</h3>
                    <code>POST /detect/video</code>
                    <p>Upload a video for deepfake detection.</p>
                    <pre>
curl -X 'POST' \\
  'http://localhost:8000/detect/video' \\
  -H 'accept: application/json' \\
  -H 'Content-Type: multipart/form-data' \\
  -F 'file=@your_video.mp4'
                    </pre>
                </div>
                
                <div class="endpoint">
                    <h3>3. Check Video Analysis Status</h3>
                    <code>GET /detect/video/status/{task_id}</code>
                    <p>Check the status of a video analysis task.</p>
                    <pre>
curl -X 'GET' \\
  'http://localhost:8000/detect/video/status/12345'
                    </pre>
                </div>
                
                <div class="endpoint">
                    <h3>4. Get Analysis Video</h3>
                    <code>GET /detect/video/result/{video_id}</code>
                    <p>Get the analysis video result.</p>
                    <pre>
curl -X 'GET' \\
  'http://localhost:8000/detect/video/result/12345'
                    </pre>
                </div>
            </body>
        </html>
        """
    
    @app.post("/detect/image", response_model=PredictionResponse)
    async def detect_image(file: UploadFile = File(...)):
        """
        Detect deepfake in an image
        
        Args:
            file: Uploaded image file
            
        Returns:
            Prediction result
        """
        try:
            # Read image file
            contents = await file.read()
            img_array = np.frombuffer(contents, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # Check if image was loaded successfully
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Start timing
            import time
            start_time = time.time()
            
            # Detect deepfake
            result = detector.predict(img)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Add processing time to result
            result['processing_time'] = processing_time
            
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/detect/image/explain")
    async def explain_image(file: UploadFile = File(...)):
        """
        Generate explanation for deepfake detection in an image
        
        Args:
            file: Uploaded image file
            
        Returns:
            Explanation visualizations
        """
        try:
            # Read image file
            contents = await file.read()
            img_array = np.frombuffer(contents, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # Check if image was loaded successfully
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Generate explanation
            explanation = detector.explain(img)
            
            # Save visualizations
            result_id = str(uuid.uuid4())
            result_dir = os.path.join(static_dir, result_id)
            os.makedirs(result_dir, exist_ok=True)
            
            # Save original image
            orig_path = os.path.join(result_dir, "original.jpg")
            cv2.imwrite(orig_path, cv2.cvtColor(explanation['original'], cv2.COLOR_RGB2BGR))
            
            # Save Grad-CAM visualization if available
            if 'grad_cam' in explanation:
                heatmap_path = os.path.join(result_dir, "grad_cam_heatmap.jpg")
                overlay_path = os.path.join(result_dir, "grad_cam_overlay.jpg")
                
                cv2.imwrite(heatmap_path, cv2.cvtColor(explanation['grad_cam']['heatmap'], cv2.COLOR_RGB2BGR))
                cv2.imwrite(overlay_path, cv2.cvtColor(explanation['grad_cam']['overlay'], cv2.COLOR_RGB2BGR))
            
            # Save attention map visualization if available
            if 'attention' in explanation:
                attn_map_path = os.path.join(result_dir, "attention_map.jpg")
                attn_heatmap_path = os.path.join(result_dir, "attention_heatmap.jpg")
                attn_overlay_path = os.path.join(result_dir, "attention_overlay.jpg")
                
                # Save attention map
                cv2.imwrite(attn_map_path, cv2.applyColorMap(
                    np.uint8(255 * explanation['attention']['map']), cv2.COLORMAP_VIRIDIS))
                
                # Save attention heatmap and overlay
                cv2.imwrite(attn_heatmap_path, cv2.cvtColor(explanation['attention']['heatmap'], cv2.COLOR_RGB2BGR))
                cv2.imwrite(attn_overlay_path, cv2.cvtColor(explanation['attention']['overlay'], cv2.COLOR_RGB2BGR))
            
            # Create response with paths
            response = {
                'result_id': result_id,
                'prediction': explanation['prediction'],
                'visualizations': {
                    'original': f"/static/{result_id}/original.jpg"
                }
            }
            
            if 'grad_cam' in explanation:
                response['visualizations']['grad_cam'] = {
                    'heatmap': f"/static/{result_id}/grad_cam_heatmap.jpg",
                    'overlay': f"/static/{result_id}/grad_cam_overlay.jpg"
                }
            
            if 'attention' in explanation:
                response['visualizations']['attention'] = {
                    'map': f"/static/{result_id}/attention_map.jpg",
                    'heatmap': f"/static/{result_id}/attention_heatmap.jpg",
                    'overlay': f"/static/{result_id}/attention_overlay.jpg"
                }
            
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def process_video_task(task_id: str, video_path: str):
        """Background task for processing video"""
        try:
            # Create output paths
            result_dir = os.path.join(static_dir, task_id)
            os.makedirs(result_dir, exist_ok=True)
            
            output_path = os.path.join(result_dir, "annotated.mp4")
            analysis_path = os.path.join(result_dir, "analysis.mp4")
            
            # Process video
            import time
            start_time = time.time()
            
            # Process video
            results = video_detector.process_video(
                video_path=video_path,
                output_path=output_path,
                max_frames=config.get("video", {}).get("max_frames", None),
                extract_frames=False
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create analysis video
            if config.get("video", {}).get("create_analysis_video", True):
                video_detector.create_analysis_video(
                    video_path=video_path,
                    output_path=analysis_path,
                    temporal_window=config.get("video", {}).get("temporal_window", 5)
                )
            
            # Add processing time to results
            results['processing_time'] = processing_time
            results['analysis_video_id'] = task_id
            
            # Save results to JSON file
            results_path = os.path.join(result_dir, "results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            # Update task status
            background_tasks[task_id] = {
                'status': 'completed',
                'result': results
            }
            
            # Remove temporary video file
            os.unlink(video_path)
        except Exception as e:
            # Update task status with error
            background_tasks[task_id] = {
                'status': 'failed',
                'error': str(e)
            }
            
            # Remove temporary video file
            if os.path.exists(video_path):
                os.unlink(video_path)
    
    @app.post("/detect/video")
    async def detect_video(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...)
    ):
        """
        Detect deepfake in a video
        
        Args:
            background_tasks: Background tasks manager
            file: Uploaded video file
            
        Returns:
            Task ID for checking status
        """
        try:
            # Create task ID
            task_id = str(uuid.uuid4())
            
            # Save uploaded file to temporary location
            _, temp_path = tempfile.mkstemp(suffix='.mp4')
            with open(temp_path, 'wb') as f:
                f.write(await file.read())
            
            # Create result directory
            result_dir = os.path.join(static_dir, task_id)
            os.makedirs(result_dir, exist_ok=True)
            
            # Add task to background tasks
            background_tasks[task_id] = {
                'status': 'processing',
                'file': file.filename
            }
            
            # Start background processing
            background_tasks.add_task(process_video_task, task_id, temp_path)
            
            return {
                'task_id': task_id,
                'status': 'processing',
                'message': 'Video uploaded successfully and is being processed'
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/detect/video/status/{task_id}")
    async def check_video_status(task_id: str):
        """
        Check status of video processing task
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status
        """
        # Check if task exists
        if task_id not in background_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Get task status
        task = background_tasks[task_id]
        
        # Return status
        if task['status'] == 'completed':
            # Return results
            return {
                'task_id': task_id,
                'status': 'completed',
                'result': task['result'],
                'analysis_video_url': f"/detect/video/result/{task_id}"
            }
        elif task['status'] == 'failed':
            # Return error
            return {
                'task_id': task_id,
                'status': 'failed',
                'error': task['error']
            }
        else:
            # Return processing status
            return {
                'task_id': task_id,
                'status': 'processing',
                'message': 'Video is still being processed'
            }
    
    @app.get("/detect/video/result/{video_id}")
    async def get_analysis_video(video_id: str):
        """
        Get analysis video
        
        Args:
            video_id: Video ID
            
        Returns:
            Analysis video file
        """
        # Check if video exists
        video_path = os.path.join(static_dir, video_id, "analysis.mp4")
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Analysis video not found")
        
        # Return video file
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=f"analysis_{video_id}.mp4"
        )
    
    return app


def run_server(app: FastAPI, host: str = "0.0.0.0", port: int = 8000):
    """
    Run FastAPI server
    
    Args:
        app: FastAPI application
        host: Host to bind
        port: Port to bind
    """
    uvicorn.run(app, host=host, port=port)