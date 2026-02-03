import numpy as np
from typing import List, Dict

class SignalSupervisor:
    def __init__(self, channels: int, variance_threshold: float = 1e-6):
        self.channels = channels
        self.var_threshold = variance_threshold
        self.channel_status = np.ones(channels, dtype=bool) # True = Good
        self.snr_history = []

    def process_window(self, data: np.ndarray):
        """
        Analyze a window of data (Time, Channels) for health.
        Updates internal status map.
        """
        # Calculate variance per channel
        variances = np.var(data, axis=0)
        
        # Simple health check: Variance too low (dead) or too high (artifact)
        # For now, just dead check
        self.channel_status = variances > self.var_threshold
        
    def get_channel_status(self) -> np.ndarray:
        return self.channel_status

    def apply_mask(self, frame: np.ndarray) -> np.ndarray:
        """Zeros out bad channels in a single frame"""
        return frame * self.channel_status
