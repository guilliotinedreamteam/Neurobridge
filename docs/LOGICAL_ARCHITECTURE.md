# Logical Architecture: NeuroBridge

## System Overview
NeuroBridge is a BCI (Brain-Computer Interface) system that translates ECoG neural signals into synthesized speech.

## Key Components

### 1. Data Pipeline
- **PhonemeInventory**: Manages the mapping between symbolic phoneme labels (e.g., 'aa', 'b') and integer IDs.
- **ECoGDatasetBuilder**: Handles loading of raw signals (.mat, .npz) and rasterizing phoneme labels from CSV files.
- **Preprocessing**: Includes bandpass filtering, notch filtering (60Hz harmonics), and z-score normalization.

### 2. Models
- **Offline Decoder**: A high-capacity model (Transformer or Bidirectional LSTM) used for training on recorded data.
- **Real-time Decoder**: A causal version of the decoder (Unidirectional LSTM) optimized for low-latency inference.
- **TransformerBlock**: Implements Multi-Head Attention for neural signal decoding.

### 3. API & Streaming
- **FastAPI Backend**: Provides endpoints for training status, model management, and speech synthesis.
- **WebSocket Streaming**: Streams 128-channel neural data to the frontend for visualization.
- **SignalProvider**: Abstraction for replaying recorded data or generating simulated waves.

### 4. Speech Synthesis
- **PhonemeSynthesizer**: A procedural synthesizer that converts a sequence of phoneme IDs into audio signals using sine/noise oscillators.

## Data Flow
1. **ECoG Input** -> **Preprocessing** -> **Rolling Window Buffer**
2. **Window Buffer** -> **Real-time Decoder** -> **Phoneme Probabilities**
3. **Smoothed Probabilities** -> **Phoneme Trigger** -> **Speech Synthesizer** -> **Audio Output**
