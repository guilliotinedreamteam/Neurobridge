import pytest
import asyncio
from neurobridge.realtime.engine import NeuralEngine
from neurobridge.signals import SineWaveSignalProvider

@pytest.mark.asyncio
async def test_engine_lifecycle():
    provider = SineWaveSignalProvider(num_channels=128)
    engine = NeuralEngine(provider)
    
    # Run engine in background
    task = asyncio.create_task(engine.start())
    await asyncio.sleep(0.2) # Let it run
    
    # Check stats
    stats = engine.get_stats()
    assert stats['frames_processed'] > 0
    assert stats['running'] == True
    
    await engine.stop()
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except asyncio.TimeoutError:
        task.cancel()

@pytest.mark.asyncio
async def test_engine_subscription():
    provider = SineWaveSignalProvider(num_channels=128)
    engine = NeuralEngine(provider)
    
    # Create subscriber
    queue = asyncio.Queue()
    engine.add_subscriber(queue)
    
    # Start engine
    task = asyncio.create_task(engine.start())
    
    # Wait for a frame
    frame = await asyncio.wait_for(queue.get(), timeout=2.0)
    
    assert "timestamp" in frame
    assert "signal_snapshot" in frame
    assert "latency_ms" in frame
    
    await engine.stop()
    await task