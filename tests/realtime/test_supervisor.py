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

def test_apply_mask():
    sup = SignalSupervisor(channels=4)
    sup.channel_status = np.array([True, True, True, False])
    
    frame = np.array([1.0, 2.0, 3.0, 4.0])
    masked = sup.apply_mask(frame)
    
    assert np.array_equal(masked, np.array([1.0, 2.0, 3.0, 0.0]))