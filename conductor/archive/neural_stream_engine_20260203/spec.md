# Specification: Neural Stream Engine (Core Pipeline)

## 1. Overview
The "Neural Stream Engine" is a high-performance, asynchronous signal processing core designed to replace the legacy polling loop. It decouples data acquisition (Producer) from inference (Consumer) to guarantee <50ms latency and robust error handling.

## 2. Core Requirements

### 2.1 Performance
- **Latency Guard:** End-to-end latency must be monitored. If latency > 40ms, the engine must drop frames to maintain real-time sync.
- **Throughput:** Must handle 128 channels at 250Hz input rate.
- **Inference Rate:** Must produce output at 20Hz (every 50ms).

### 2.2 Components
- **AsyncRingBuffer:** A thread-safe, lock-protected ring buffer for raw signal data. Must support "overwrite oldest" behavior on overflow.
- **SignalSupervisor:** A health monitoring agent that calculates SNR per channel. Must detect "dead" (flatline) and "noisy" (high variance) channels and mask them out.
- **NeuralEngine:** The orchestrator class managing the Producer and Consumer loops.
- **EventBus:** A Pub/Sub mechanism to broadcast "Holographic Frames" to the API layer without blocking the engine.

### 2.3 Data Protocol ("Holographic Frame")
The engine must emit JSON-serializable frames containing:
- `timestamp`: Acquisition time ($T_0$).
- `latency_ms`: Processing delay ($T_{now} - T_0$).
- `signal_snapshot`: Compressed raw data for UI visualization.
- `health`: System stats (dropped frames, bad channel list).

## 3. Interfaces

### 3.1 NeuralEngine API
```python
class NeuralEngine:
    async def start(self): ...
    async def stop(self): ...
    def get_stats(self) -> dict: ...
    async def subscribe(self) -> AsyncIterator[dict]: ...
```

### 3.2 SignalSupervisor API
```python
class SignalSupervisor:
    def process_window(self, data: np.ndarray): ...
    def get_channel_status(self) -> np.ndarray: ... # Boolean mask
```

## 4. Constraints
- **Python:** 3.10+
- **Asyncio:** Must use `asyncio` for concurrency.
- **No Blocking:** CPU-bound tasks (inference) must run in a separate thread/process if they block the event loop >5ms.
