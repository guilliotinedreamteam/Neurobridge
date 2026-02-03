# Physical Architecture: NeuroBridge

## Deployment View

### 1. Backend (Python)
- **Runtime**: Python 3.12+
- **Framework**: FastAPI (Web), TensorFlow (ML), Uvicorn (Server).
- **Communication**: REST API (Status/Config), WebSockets (Signal Streaming).

### 2. Frontend (React)
- **Runtime**: Node.js (Vite)
- **Framework**: React, Tailwind CSS, shadcn/ui.
- **Visualization**: HTML5 Canvas / Recharts for real-time waveforms.

## Directory Structure
```
neurobridge_project/
├── graceful-ferret-soar/
│   ├── neurobridge/          # Python Core
│   │   ├── api.py            # Entry point
│   │   ├── models.py         # Keras Models
│   │   └── ...
│   ├── src/                  # React Source
│   ├── tests/                # Pytest suite
│   ├── data/                 # Raw data (ECoG/Labels)
│   └── artifacts/            # Model weights and audio
```

## Scaling Considerations
- **Concurrency**: FastAPI handles multiple requests; training is offloaded to background threads.
- **Data Volume**: Neural data is high-bandwidth; WebSockets are used to minimize overhead for real-time visualization.
- **GPU Acceleration**: TensorFlow models can utilize NVIDIA GPUs (via CUDA) or Apple Silicon (via Metal/MPS).
