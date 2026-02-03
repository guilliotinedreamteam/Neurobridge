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
