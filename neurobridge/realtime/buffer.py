import numpy as np
import asyncio

class AsyncRingBuffer:
    def __init__(self, capacity: int, channels: int):
        self.capacity = capacity
        self.channels = channels
        self._buffer = np.zeros((capacity, channels))
        self._timestamps = np.zeros(capacity) # Store T0 here
        self._head = 0
        self._count = 0
        self._lock = asyncio.Lock()

    async def push(self, data: np.ndarray, timestamp: float = 0.0):
        async with self._lock:
            self._buffer[self._head] = data
            self._timestamps[self._head] = timestamp
            self._head = (self._head + 1) % self.capacity
            if self._count < self.capacity:
                self._count += 1

    async def pop(self):
        """Returns (data, timestamp) of the latest item (LIFO behavior for consumer)"""
        async with self._lock:
            if self._count == 0:
                return None, None
            
            # For BCI we usually want the latest data, not the oldest
            # This implementation will act as a stack for 'pop' to get the freshest data
            # Or we can implement 'pop_oldest'
            idx = (self._head - 1) % self.capacity
            data = self._buffer[idx].copy()
            ts = self._timestamps[idx]
            
            # In a real producer-consumer, pop usually removes. 
            # But for viz, we often just want the latest.
            # I'll implement pop as 'remove oldest' (FIFO) to match standard queues.
            
            # FIFO Pop:
            tail = (self._head - self._count) % self.capacity
            data = self._buffer[tail].copy()
            ts = self._timestamps[tail]
            self._count -= 1
            return data, ts

    def size(self):
        return self._count