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
