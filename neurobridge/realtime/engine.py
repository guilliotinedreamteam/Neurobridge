import asyncio
import time
import numpy as np
from typing import List, Set
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
        self._subscribers: Set[asyncio.Queue] = set()

    def add_subscriber(self, queue: asyncio.Queue):
        self._subscribers.add(queue)
        logger.debug(f"Subscriber added. Total: {len(self._subscribers)}")

    def remove_subscriber(self, queue: asyncio.Queue):
        if queue in self._subscribers:
            self._subscribers.remove(queue)
            logger.debug(f"Subscriber removed. Total: {len(self._subscribers)}")

    async def start(self):
        self._running = True
        logger.info("Neural Engine Started")
        try:
            await asyncio.gather(
                self._producer(),
                self._consumer()
            )
        except asyncio.CancelledError:
            logger.info("Neural Engine Tasks Cancelled")
        finally:
            self._running = False

    async def stop(self):
        self._running = False
        logger.info("Neural Engine Stopping...")

    async def _producer(self):
        """Acquires data at ~250Hz"""
        while self._running:
            start_time = time.time()
            
            # 1. Get Data
            frame = self.provider.get_next_sample()
            
            # 2. Push to Buffer with timestamp
            await self.buffer.push(frame, timestamp=time.time())
            
            # 3. Maintain Timing (4ms = 250Hz)
            elapsed = time.time() - start_time
            sleep_time = max(0, 0.004 - elapsed)
            await asyncio.sleep(sleep_time)

    async def _consumer(self):
        """Processes data at ~20Hz (50ms)"""
        while self._running:
            # 1. Get latest frame
            if self.buffer.size() > 0:
                raw_frame, t0 = await self.buffer.pop()
                
                self.latest_frame = raw_frame
                self._frames_processed += 1
                
                # Apply Health Mask
                clean_frame = self.supervisor.apply_mask(raw_frame)
                
                # Create Holographic Frame
                hologram = {
                    "timestamp": t0,
                    "latency_ms": (time.time() - t0) * 1000,
                    "signal_snapshot": clean_frame.tolist(),
                    "health": {
                        "bad_channels": np.where(~self.supervisor.get_channel_status())[0].tolist(),
                        "buffer_size": self.buffer.size()
                    }
                }
                
                # Broadcast to subscribers
                await self._broadcast(hologram)
            
            await asyncio.sleep(0.05) # 20Hz

    async def _broadcast(self, data: dict):
        if not self._subscribers:
            return
            
        for q in self._subscribers:
            try:
                if q.full():
                    q.get_nowait() # Drop oldest if full
                q.put_nowait(data)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")

    def get_stats(self):
        return {
            "running": self._running,
            "frames_processed": self._frames_processed,
            "buffer_size": self.buffer.size(),
            "subscribers": len(self._subscribers)
        }
