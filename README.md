# NeuroBridge

A neural interface bridge system for processing and analyzing brain-computer interface data.

## Overview

NeuroBridge is a sophisticated system designed to facilitate communication between neural interfaces and computational systems. This project provides tools for signal processing, data analysis, and real-time neural data streaming.

## Features

- **Real-time Signal Visualization**: WebSocket-based streaming of 128-channel ECoG data.
- **Modern Dashboard**: React + Tailwind + shadcn/ui based control panel.
- **FastAPI Backend**: Robust Python backend for training and inference.
- **Transformer Architecture**: State-of-the-art decoding models.
- **Modular Design**: Easy to extend for new neural protocols.

## Project Structure

```
giga-apps/graceful-ferret-soar/
├── neurobridge/              # Python Backend
│   ├── api.py                # FastAPI Server
│   ├── models.py             # Transformer & RNN Models
│   ├── data_pipeline.py      # Optimized Data Loading
│   └── ...
├── src/                      # React Frontend
│   ├── components/           # UI Components (Visualizer, Sidebar)
│   ├── pages/                # Dashboard Pages
│   └── lib/                  # API Clients
└── neurobridge.config.yaml   # Configuration
```

## Quick Start (Demo Mode)

We have provided a unified runner script to start both the Backend (FastAPI) and Frontend (Vite) in simulation mode.

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install fastapi uvicorn websockets
   cd graceful-ferret-soar
   npm install
   cd ..
   ```

2. **Run the System:**
   ```bash
   ./run_neurobridge.sh
   ```

3. **Access the Dashboard:**
   Open [http://localhost:5173](http://localhost:5173) in your browser.

## Backend API

- **Status**: `GET /status`
- **Train**: `POST /train` (Starts async training)
- **Synthesize**: `POST /synthesize` (Phoneme to Audio)
- **Stream**: `WS /ws/signals` (Real-time data stream)

## Configuration

Configuration options live inside `neurobridge.config.yaml`. Key fields:
- `dataset`: Input directories and sampling rates.
- `model`: Choose between `rnn` or `transformer` architecture.
- `training`: Hyperparameters.

## License

MIT License.

---

Made with [Giga](https://gigamind.dev/) & Improved by **Ralph Loop**.