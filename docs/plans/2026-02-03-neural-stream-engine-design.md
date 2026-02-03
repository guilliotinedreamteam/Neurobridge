# Neural Stream Engine: Breakthrough Architecture

**Date:** 2026-02-03
**Status:** Approved
**Goal:** Achieve <50ms latency, robust error handling, and adaptive resilience for real-time BCI decoding.

## 1. Overview
The "Neural Stream Engine" is a total rewrite of the signal processing core. It replaces the simple polling loop with a multi-threaded, `asyncio`-driven Producer-Consumer architecture. It incorporates "Breakthrough" logic to self-stabilize against noise and latency drift.

## 2. Core Architecture

### 2.1 Producer-Consumer Pipeline
*   **Signal Producer (Thread 1):** Dedicated high-priority thread for data acquisition. Pushes raw neural frames (128ch @ 250Hz) into a `RingBuffer`.
*   **Decoder Consumer (Thread 2 / Async Worker):** Pulls data from the buffer, constructs windows, runs ML inference, and emits events.
*   **Event Bus (Main Loop):** Decouples processing from networking. The WebSocket endpoint subscribes to the bus to broadcast updates.

### 2.2 Latency Guard Protocol
To guarantee the <50ms latency target:
*   **T0 Timestamping:** Every sample is timestamped at acquisition.
*   **Drift Detection:** $Latency = T_{now} - T_{sample}$.
*   **Auto-Drop:** If $Latency > 40ms$, the engine drops the next inference step to catch up. This prioritizes "live" feel over processing every single microsecond of historical data.

## 3. The "Breakthrough" Features

### 3.1 Signal Health Autopilot
A lightweight supervisor agent runs alongside the decoder:
*   **SNR Monitoring:** Continuously calculates Signal-to-Noise Ratio for all 128 channels.
*   **Dynamic Rejection:** If a channel's variance spikes (artifact) or drops to flatline (disconnect), it is zeroed out *before* inference. This prevents the model from decoding noise as speech.
*   **Adaptive Smoothing:**
    *   *High Model Confidence (>90%):* Low smoothing (fast typing).
    *   *Low Model Confidence (<60%):* High smoothing (stable output).

### 3.2 Sliding Window Inference
*   **Window Size:** 200ms (typical for phoneme detection).
*   **Step Size:** 50ms.
*   **Overlap:** 75%.
*   **Result:** 20 updates per second (20Hz), creating a fluid visual experience.

## 4. Data Protocol ("Holographic" Frames)

The WebSocket will transmit a rich, typed JSON object (ProtoBuf ready):

```json
{
  "timestamp": 123456789,
  "latency_ms": 12,
  "signal_snapshot": [[...], ...], // Compressed 128ch snapshot
  "decoder": {
    "phonemes": [
      {"symbol": "aa", "prob": 0.95},
      {"symbol": "n", "prob": 0.02}
    ],
    "confidence": 0.95
  },
  "health": {
    "snr_db": 15.4,
    "dropped_frames": 0,
    "bad_channels": [12, 110]
  }
}
```

## 5. Implementation Strategy
1.  **Refactor `neurobridge/realtime.py`:** Implement the `RingBuffer` and `NeuralEngine` class.
2.  **Implement `Supervisor`:** Add the logic for channel rejection and SNR calculation.
3.  **Upgrade `api.py`:** Replace the dummy loop with the Engine's `EventBus`.
