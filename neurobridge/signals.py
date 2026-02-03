from abc import ABC, abstractmethod
import numpy as np
import math
import random
from pathlib import Path
from typing import Optional, List

class SignalProvider(ABC):
    @abstractmethod
    def get_next_sample(self) -> List[float]:
        pass

class SineWaveSignalProvider(SignalProvider):
    """Fallback provider that generates sine waves with noise."""
    def __init__(self, num_channels: int = 128):
        self.num_channels = num_channels
        self.t = 0

    def get_next_sample(self) -> List[float]:
        data = []
        for i in range(self.num_channels):
            val = math.sin(self.t * 0.1 + i * 0.5) * 0.5 + \
                  math.sin(self.t * 0.03 + i * 0.1) * 0.3 + \
                  random.gauss(0, 0.1)
            data.append(val)
        self.t += 1
        return data

class ReplaySignalProvider(SignalProvider):
    """Replays recorded ECoG data from a .npz or .mat file."""
    def __init__(self, file_path: Path, num_channels: int = 128):
        self.file_path = file_path
        self.num_channels = num_channels
        self.data = self._load_data()
        self.cursor = 0

    def _load_data(self) -> np.ndarray:
        if not self.file_path.exists():
            # If file doesn't exist, return an empty array or raise error
            # For now, let's return a dummy array to avoid crashing if called
            return np.zeros((1000, self.num_channels))
        
        # Logic to load based on extension
        if self.file_path.suffix == ".npz":
            arr = np.load(self.file_path)
            keys = [k for k in arr.files if not k.startswith("__")]
            return arr[keys[0]]
        elif self.file_path.suffix == ".mat":
            from scipy.io import loadmat
            mat = loadmat(self.file_path)
            candidates = [k for k in mat.keys() if not k.startswith("__")]
            return mat[candidates[0]]
        
        return np.zeros((1000, self.num_channels))

    def get_next_sample(self) -> List[float]:
        if len(self.data) == 0:
            return [0.0] * self.num_channels
        
        sample = self.data[self.cursor].tolist()
        self.cursor = (self.cursor + 1) % len(self.data)
        return sample
