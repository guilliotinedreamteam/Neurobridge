from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import threading
import asyncio
import json
import time
from pathlib import Path
from loguru import logger

from .config import NeuroBridgeConfig, DatasetConfig
from .data_pipeline import PhonemeInventory
from .speech import PhonemeSynthesizer
from .training import train_and_evaluate
from .signals import SineWaveSignalProvider, ReplaySignalProvider
from .realtime.engine import NeuralEngine

# Global Engine Management (used in lifespan)
engine: Optional[NeuralEngine] = None
engine_task: Optional[asyncio.Task] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, engine_task
    # Startup
    provider = SineWaveSignalProvider(num_channels=128)
    engine = NeuralEngine(provider)
    engine_task = asyncio.create_task(engine.start())
    logger.info("Neural Engine background task started")
    
    yield
    
    # Shutdown
    if engine:
        await engine.stop()
    if engine_task:
        engine_task.cancel()
        try:
            await engine_task
        except asyncio.CancelledError:
            pass
    logger.info("Neural Engine background task stopped")

app = FastAPI(title="NeuroBridge API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainingRequest(BaseModel):
    config_path: str = "neurobridge.config.yaml"
    epochs: int = 10

class SynthesisRequest(BaseModel):
    sequence: str
    output_path: Optional[str] = None

class SystemStatus(BaseModel):
    status: str
    current_task: Optional[str] = None
    engine_running: bool = False

# Thread-safe system state
class StateManager:
    def __init__(self):
        self._state = {"status": "idle", "current_task": None}
        self._lock = threading.Lock()

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return self._state.copy()

    def set_task(self, status: str, task: Optional[str]):
        with self._lock:
            self._state["status"] = status
            self._state["current_task"] = task

state_manager = StateManager()

@app.get("/status", response_model=SystemStatus)
def get_status():
    state = state_manager.get_state()
    # Note: engine is global and managed by lifespan
    state["engine_running"] = engine._running if engine else False
    return state

@app.websocket("/ws/signals")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    if not engine:
        await websocket.close(code=1001, reason="Engine not initialized")
        return

    # Create a subscriber queue
    q = asyncio.Queue(maxsize=10)
    engine.add_subscriber(q)

    try:
        while True:
            # Wait for Holographic Frame from engine
            frame = await q.get()
            await websocket.send_json(frame)
    except Exception as e:
        logger.info(f"WebSocket disconnected or error: {e}")
    finally:
        engine.remove_subscriber(q)

async def run_training_task(config_path: str, epochs: int):
    state_manager.set_task("training", f"Training with {config_path}")
    try:
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            logger.error(f"Config file {cfg_path} not found")
            return
        
        config = NeuroBridgeConfig.from_yaml(cfg_path)
        config.training.max_epochs = epochs
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, train_and_evaluate, config)
        logger.info(f"Training completed successfully for {config_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
    finally:
        state_manager.set_task("idle", None)

@app.post("/train")
async def trigger_training(req: TrainingRequest, background_tasks: BackgroundTasks):
    current_status = state_manager.get_state()["status"]
    if current_status != "idle":
        raise HTTPException(status_code=409, detail="System is busy")
    
    background_tasks.add_task(run_training_task, req.config_path, req.epochs)
    return {"message": "Training started in background"}

@app.post("/synthesize")
def trigger_synthesis(req: SynthesisRequest):
    try:
        # Load default config for speech settings
        cfg_path = Path("neurobridge.config.yaml")
        if not cfg_path.exists():
             raise FileNotFoundError("Default config neurobridge.config.yaml not found")
        
        config = NeuroBridgeConfig.from_yaml(cfg_path)
        inventory = PhonemeInventory(config.dataset.phonemes)
        ids = [int(token) for token in req.sequence.split(",") if token.strip()]
        synthesizer = PhonemeSynthesizer(inventory, config.speech)
        
        audio = synthesizer.synthesize(ids)
        if audio.size == 0:
            return {"message": "No audio generated (empty sequence)"}
            
        output = req.output_path or str(config.speech.export_audio_dir / "output.wav")
        synthesizer.save_wav(audio, Path(output))
        
        return {"message": "Synthesis complete", "output": output}
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
