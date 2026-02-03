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
    await asyncio.sleep(0.5) # Let it run
    
    # Check stats
    stats = engine.get_stats()
    assert stats['frames_processed'] > 0
    assert stats['running'] == True
    
    await engine.stop()
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except asyncio.TimeoutError:
        task.cancel()
