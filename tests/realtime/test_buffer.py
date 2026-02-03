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
        frame = np.full((128,), float(i))
        await buf.push(frame, timestamp=float(i))
    
    # Check head/tail logic
    assert buf.size() == 10
    
    # Pop oldest (FIFO)
    # 0,1,2,3,4 were overwritten. 5 is oldest.
    data, ts = await buf.pop()
    assert data[0] == 5.0
    assert ts == 5.0