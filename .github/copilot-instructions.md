# Copilot Instructions for NeuroBridge

## Project Overview

NeuroBridge is a full-stack neural interface bridge system that processes and visualizes brain-computer interface (ECoG) data. It combines a **FastAPI Python backend** for real-time signal processing and model training with a **React/TypeScript frontend** for dashboard visualization.

**Key Architecture:**
- **Backend** (`neurobridge/`): FastAPI server handling WebSocket streaming, training, and speech synthesis
- **Frontend** (`src/`): React + TypeScript dashboard with real-time signal visualization
- **Config-driven**: System behavior controlled via `neurobridge.config.yaml`

## Build & Run Commands

### Frontend (React/Vite)
```bash
npm install                 # Install dependencies
npm run dev               # Start dev server (localhost:5173)
npm run build             # Build for production
npm run lint              # Run ESLint
npm run preview           # Preview production build locally
```

### Backend (FastAPI)
```bash
pip install -r requirements.txt    # Install dependencies (requires uvicorn, fastapi, websockets)
python -m neurobridge.api          # Start FastAPI server (localhost:8000)
```

### Full System
```bash
../run_neurobridge.sh     # Start both backend and frontend simultaneously
```

### Testing
```bash
pytest tests/             # Run all tests
pytest tests/test_models.py -v  # Run single test file with verbose output
pytest -k test_name       # Run specific test by name
```

## Architecture & Key Components

### Backend Structure

**Entry Point:** `neurobridge/api.py`
- FastAPI application with CORS middleware
- Key endpoints: `/status`, `/train`, `/synthesize`, `WS /ws/signals` (WebSocket)
- Uses `StateManager` for thread-safe concurrent operations

**Core Modules:**
- **`config.py`** - YAML-based configuration loader (NeuroBridgeConfig, DatasetConfig)
- **`data_pipeline.py`** - ECoG data loading, preprocessing, windowing; PhonemeInventory management
- **`models.py`** - Neural network architectures (transformer/RNN decoders)
- **`training.py`** - Model training and evaluation loop
- **`signals.py`** - Signal providers (SineWaveSignalProvider for simulation, ReplaySignalProvider for data)
- **`speech.py`** - Phoneme synthesis (PhonemeSynthesizer)
- **`realtime/`** - Real-time inference components
  - `engine.py` - Real-time processing engine
  - `decoder.py` - Online decoding logic
  - `buffer.py` - Circular buffer for streaming data
  - `supervisor.py` - State coordination

### Frontend Structure

**Routing:** `src/App.tsx`
- All routes must be added to the Routes component
- Main page: `src/pages/Index.tsx` (edit this to include new components)
- Use React Router for navigation

**Styling & UI:**
- **Tailwind CSS** for all styling (use classes extensively)
- **shadcn/ui** components (all pre-installed; use without additional setup)
- **Lucide React** for icons
- **React Query** for API client state management

**Component Organization:**
- `src/components/` - Reusable UI components
- `src/pages/` - Page-level components
- `src/lib/` - API clients and utilities
- Use `@` alias for imports: `@/components/...`, `@/lib/...`

### Configuration

**Location:** `neurobridge.config.yaml`
- **dataset**: Input directories, sampling rates, window duration, number of features (128 channels)
- **model**: RNN/transformer architecture parameters
- **training**: Batch size, epochs, learning rate, validation/test splits
- **realtime**: Frame duration, smoothing, debounce thresholds
- **speech**: Phoneme synthesis parameters

## Key Conventions

### Python Backend

**Imports:** Use absolute imports from package root
```python
from neurobridge.config import NeuroBridgeConfig
from neurobridge.models import TransformerDecoder
```

**Logging:** Use loguru logger
```python
from loguru import logger
logger.info("Message")
```

**API Requests:** Use Pydantic models for request/response validation
```python
class TrainingRequest(BaseModel):
    config_path: str
    epochs: int
```

**Async Operations:** Use `BackgroundTasks` for async training to avoid blocking
```python
@app.post("/train")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(train_and_evaluate, ...)
```

### React/TypeScript Frontend

**Component Pattern:** Functional components with hooks
```tsx
export default function ComponentName() {
  return <div>...</div>;
}
```

**Styling:** Use Tailwind classes directly; wrap custom styles in new components if shadcn components need overriding

**API Integration:** Use React Query for fetching; API client methods in `src/lib/`

**Page Updates:** Always update `src/pages/Index.tsx` to include new components or they won't be visible

**Routes:** Keep all routes in `src/App.tsx` under the Routes component

## Data Flow

1. **Signal Streaming:** WebSocket (`/ws/signals`) sends 128-channel ECoG data as JSON frames
2. **Training:** POST `/train` triggers async background task using config from YAML
3. **Synthesis:** POST `/synthesize` converts phoneme sequences to audio via PhonemeSynthesizer
4. **Real-time Decoding:** Uses signal providers and realtime engine for online inference

## Development Tips

- **Configuration changes:** Modify `neurobridge.config.yaml` and restart backend
- **Signal simulation:** Use `SineWaveSignalProvider` for testing without real hardware
- **Data replay:** Use `ReplaySignalProvider` to simulate with recorded data
- **Frontend dev:** Changes auto-reload with Vite dev server
- **Backend changes:** Restart FastAPI server to pick up changes
- **Testing models:** Use pytest with `tests/test_models.py` as reference

## Testing Strategy

- **Unit tests:** Test individual modules (config loading, model creation, data pipeline)
- **Test location:** `tests/` directory with same structure as `neurobridge/`
- **Realtime tests:** Specialized tests in `tests/realtime/` for signal processing

## VS Code / Editor Setup

- TypeScript/React files use SWC compiler (fast refresh)
- Path alias `@/` resolves to `src/`
- ESLint config: `eslint.config.js` (fix with `npm run lint`)
- Tailwind IntelliSense recommended for class completion
