
# Configuration for NeuroBridge

# Based on typical BCI phoneme decoding tasks
NUM_TIMESTEPS = 100    # Sequence length (e.g., 100ms of data)
NUM_FEATURES = 128     # Number of ECoG electrode channels
NUM_PHONEMES = 41      # e.g., 40 phonemes + 1 silence token

# Training parameters
BATCH_SIZE = 32
EPOCHS = 5
TOTAL_MOCK_SAMPLES = 1000
