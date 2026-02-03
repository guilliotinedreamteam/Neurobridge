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
