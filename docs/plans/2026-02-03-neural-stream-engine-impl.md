# Neural Stream Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a high-performance, asyncio-based Producer-Consumer architecture for real-time ECoG decoding with <50ms latency guard and signal health monitoring.

**Architecture:** A `NeuralEngine` class orchestrates a `SignalProducer` (data acquisition) and `DecoderConsumer` (ML inference) connected by an async `RingBuffer`. A `Supervisor` agent monitors signal SNR and auto-rejects bad channels.

**Tech Stack:** Python 3.10+, `asyncio` (queues/tasks), `numpy` (DSP), `loguru` (logging), `FastAPI` (WebSockets).

---

### Task 1: Implement Async Ring Buffer

**Files:**
- Create: `neurobridge/realtime/buffer.py`
- Test: `tests/realtime/test_buffer.py`

**Step 1: Write the failing test**

```python
import pytest
import asyncio
import numpy as np
from neurobridge.realtime.buffer import AsyncRingBuffer

@pytest.mark.asyncio
async def test_ring_buffer_overflow():
    # Buffer size 10, push 15 items. Oldest 5 should be overwritten.
    buf = AsyncRingBuffer(capacity=10, channels=128)
    
    # Push 15 frames
    for i in range(15):
        frame = np.full((128,), i) # Frame of all i's
        await buf.push(frame)
    
    # Check head/tail logic
    assert buf.size() == 10
    
    # Pop oldest (should be 5, not 0)
    data = await buf.pop()
    assert data[0] == 5.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/realtime/test_buffer.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
import asyncio
import numpy as np
from typing import Optional

class AsyncRingBuffer:
    def __init__(self, capacity: int, channels: int):
        self.capacity = capacity
        self.channels = channels
        self._buffer = np.zeros((capacity, channels), dtype=np.float32)
        self._head = 0 # Write pointer
        self._tail = 0 # Read pointer
        self._count = 0
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)

    async def push(self, frame: np.ndarray):
        async with self._lock:
            self._buffer[self._head] = frame
            self._head = (self._head + 1) % self.capacity
            
            if self._count < self.capacity:
                self._count += 1
            else:
                # Overwrite: Advance tail to drop oldest
                self._tail = (self._tail + 1) % self.capacity
            
            self._not_empty.notify()

    async def pop(self) -> np.ndarray:
        async with self._not_empty:
            await self._not_empty.wait_for(lambda: self._count > 0)
            
            data = self._buffer[self._tail].copy()
            self._tail = (self._tail + 1) % self.capacity
            self._count -= 1
            return data

    def size(self) -> int:
        return self._count
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/realtime/test_buffer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add neurobridge/realtime/buffer.py tests/realtime/test_buffer.py
git commit -m "feat(realtime): Implement AsyncRingBuffer with overflow handling"
```

---

### Task 2: Implement Signal Health Supervisor

**Files:**
- Create: `neurobridge/realtime/supervisor.py`
- Test: `tests/realtime/test_supervisor.py`

**Step 1: Write the failing test**

```python
import pytest
import numpy as np
from neurobridge.realtime.supervisor import SignalSupervisor

def test_detect_dead_channel():
    sup = SignalSupervisor(channels=4)
    
    # 3 active channels, 1 dead (flatline)
    active = np.random.normal(0, 1, (100, 3))
    dead = np.zeros((100, 1))
    data = np.hstack([active, dead]) # Shape (100, 4)
    
    sup.process_window(data)
    status = sup.get_channel_status()
    
    assert status[3] == False # Dead
    assert status[0] == True  # Active
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/realtime/test_supervisor.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
import numpy as np
from typing import List, Dict

class SignalSupervisor:
    def __init__(self, channels: int, variance_threshold: float = 1e-6):
        self.channels = channels
        self.var_threshold = variance_threshold
        self.channel_status = np.ones(channels, dtype=bool) # True = Good
        self.snr_history = []

    def process_window(self, data: np.ndarray):
        """
        Analyze a window of data (Time, Channels) for health.
        Updates internal status map.
        """
        # Calculate variance per channel
        variances = np.var(data, axis=0)
        
        # Simple health check: Variance too low (dead) or too high (artifact)
        # For now, just dead check
        self.channel_status = variances > self.var_threshold
        
    def get_channel_status(self) -> np.ndarray:
        return self.channel_status

    def apply_mask(self, frame: np.ndarray) -> np.ndarray:
        """Zeros out bad channels in a single frame"""
        return frame * self.channel_status
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/realtime/test_supervisor.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add neurobridge/realtime/supervisor.py tests/realtime/test_supervisor.py
git commit -m "feat(realtime): Implement SignalSupervisor for dead channel detection"
```

---

### Task 3: Implement Neural Engine Core (Producer/Consumer)

**Files:**
- Create: `neurobridge/realtime/engine.py`
- Test: `tests/realtime/test_engine.py`

**Step 1: Write the failing test**

```python
import pytest
import asyncio
from neurobridge.realtime.engine import NeuralEngine
from neurobridge.signals import SineWaveSignalProvider

@pytest.mark.asyncio
async def test_engine_lifecycle():
    provider = SineWaveSignalProvider(channels=128)
    engine = NeuralEngine(provider)
    
    task = asyncio.create_task(engine.start())
    await asyncio.sleep(0.5) # Let it run
    
    # Check stats
    stats = engine.get_stats()
    assert stats['frames_processed'] > 0
    assert stats['running'] == True
    
    await engine.stop()
    await task
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/realtime/test_engine.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
import asyncio
import time
import numpy as np
from loguru import logger
from .buffer import AsyncRingBuffer
from .supervisor import SignalSupervisor
from ..signals import SignalProvider

class NeuralEngine:
    def __init__(self, provider: SignalProvider):
        self.provider = provider
        self.buffer = AsyncRingBuffer(capacity=2048, channels=128)
        self.supervisor = SignalSupervisor(channels=128)
        self._running = False
        self._frames_processed = 0
        self.latest_frame = None

    async def start(self):
        self._running = True
        logger.info("Neural Engine Started")
        await asyncio.gather(
            self._producer(),
            self._consumer()
        )

    async def stop(self):
        self._running = False
        logger.info("Neural Engine Stopped")

    async def _producer(self):
        """Acquires data at ~250Hz"""
        while self._running:
            start_time = time.time()
            
            # 1. Get Data
            frame = self.provider.get_next_sample()
            
            # 2. Push to Buffer
            await self.buffer.push(frame)
            
            # 3. Maintain Timing (4ms = 250Hz)
            elapsed = time.time() - start_time
            sleep_time = max(0, 0.004 - elapsed)
            await asyncio.sleep(sleep_time)

    async def _consumer(self):
        """Processes data at ~20Hz (50ms)"""
        while self._running:
            # 1. Get latest frame for viz
            if self.buffer.size() > 0:
                self.latest_frame = await self.buffer.pop()
                self._frames_processed += 1
                
                # Apply Health Mask
                clean_frame = self.supervisor.apply_mask(self.latest_frame)
                
                # TODO: Run Decoder Here
            
            await asyncio.sleep(0.05) # 20Hz

    def get_stats(self):
        return {
            "running": self._running,
            "frames_processed": self._frames_processed,
            "buffer_size": self.buffer.size()
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/realtime/test_engine.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add neurobridge/realtime/engine.py tests/realtime/test_engine.py
git commit -m "feat(realtime): Implement NeuralEngine producer-consumer loop"
```

---

### Task 4: Integrate with FastAPI WebSocket

**Files:**
- Modify: `neurobridge/api.py`
- Test: Manual/Integration check (since it requires a running server)

**Step 1: Write integration test**

```python
# tests/test_api_integration.py
from fastapi.testclient import TestClient
from neurobridge.api import app

def test_websocket_connect():
    client = TestClient(app)
    with client.websocket_connect("/ws/signals") as websocket:
        data = websocket.receive_json()
        assert "signal_snapshot" in data
        assert "health" in data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_integration.py -v`
Expected: FAIL (Schema mismatch)

**Step 3: Modify api.py**

Replace the dummy loop with:

```python
# In api.py
from .realtime.engine import NeuralEngine
from .signals import SineWaveSignalProvider

# Global Engine Instance
engine = NeuralEngine(SineWaveSignalProvider(128))

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(engine.start())

@app.on_event("shutdown")
async def shutdown_event():
    await engine.stop()

@app.websocket("/ws/signals")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Broadcast latest state from engine
            if engine.latest_frame is not None:
                payload = {
                    "timestamp": time.time(),
                    "signal_snapshot": engine.latest_frame.tolist(), # Serialize
                    "health": {
                        "frames_processed": engine.get_stats()['frames_processed']
                    }
                }
                await websocket.send_json(payload)
            await asyncio.sleep(0.05)
    except Exception as e:
        logger.info(f"Client disconnected: {e}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_integration.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add neurobridge/api.py tests/test_api_integration.py
git commit -m "feat(api): Connect WebSocket to NeuralStreamEngine"
```
