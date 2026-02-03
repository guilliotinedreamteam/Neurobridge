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
                # clean_frame = self.supervisor.apply_mask(self.latest_frame)
                
                # TODO: Run Decoder Here
            
            await asyncio.sleep(0.05) # 20Hz

    def get_stats(self):
        return {
            "running": self._running,
            "frames_processed": self._frames_processed,
            "buffer_size": self.buffer.size()
        }
