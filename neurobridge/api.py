from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import threading
import asyncio
import json
import random
import math
from pathlib import Path
from loguru import logger

from .config import NeuroBridgeConfig, DatasetConfig
from .data_pipeline import PhonemeInventory
from .speech import PhonemeSynthesizer
from .training import train_and_evaluate

app = FastAPI(title="NeuroBridge API", version="1.0.0")

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

from .signals import SineWaveSignalProvider, ReplaySignalProvider

state_manager = StateManager()

@app.get("/status", response_model=SystemStatus)
def get_status():
    return state_manager.get_state()

@app.websocket("/ws/signals")
async def websocket_endpoint(websocket: WebSocket, source: Optional[str] = None):
    await websocket.accept()
    
    # Select signal provider
    if source and Path(source).exists():
        provider = ReplaySignalProvider(Path(source))
    else:
        provider = SineWaveSignalProvider()

    try:
        t = 0
        while True:
            data = provider.get_next_sample()
            await websocket.send_json({"timestamp": t, "channels": data})
            t += 1
            await asyncio.sleep(0.05) # 20Hz update rate for demo
    except Exception as e:
        logger.info(f"WebSocket disconnected: {e}")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
