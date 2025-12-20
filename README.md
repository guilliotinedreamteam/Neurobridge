
# NeuroBridge: Speech Neuroprosthesis System

## Scientific Basis

This project implements a state-of-the-art Deep Learning architecture for **Brain-Computer Interface (BCI)** applications, specifically designed to decode intracranial electrocorticography (ECoG) signals into intelligible speech (phonemes).

The core hypothesis is that high-density neural activity in the sensorimotor cortex contains sufficient temporal information to reconstruct the kinematic and acoustic properties of speech. NeuroBridge utilizes **Recurrent Neural Networks (RNNs)**, specifically Bidirectional Long Short-Term Memory (LSTM) units, to model the complex, non-linear, and temporal dynamics of these neural signals.

This approach aligns with groundbreaking research from leading institutions (e.g., UCSF, Stanford) in the field of speech neuroprosthetics, aiming to restore communication for individuals with severe paralysis or speech deficits (e.g., ALS, brainstem stroke).

## Architecture

The system is composed of two primary model variants:

1.  **Offline Decoder (`build_neurobridge_decoder`)**:
    *   **Input**: Time-series ECoG data (Batch, Timesteps, Channels).
    *   **Core**: Stacked **Bidirectional LSTM** layers. This architecture captures context from both past and future neural states within a sentence or utterance, maximizing decoding accuracy for offline analysis.
    *   **Regularization**: Batch Normalization is applied after each recurrent layer to stabilize training and accelerate convergence.
    *   **Output**: A TimeDistributed Dense layer with Softmax activation predicting the probability distribution over the phoneme set (including silence) for each timestep.

2.  **Real-Time Decoder (`build_realtime_decoder`)**:
    *   **Constraint**: Causal processing (no look-ahead).
    *   **Core**: Stacked **Unidirectional LSTM** layers. This ensures that predictions at time *t* depend only on neural activity up to time *t*, enabling low-latency, real-time synthesis.
    *   **Inference Strategy**: Uses a sliding window buffer to maintain sufficient historical context for stable predictions.
    *   **Note**: The current prototype demonstrates inference using the trained **Bidirectional** model within a sliding window to leverage the weights learned during offline training. In a production deployment, a Unidirectional model would be trained specifically for low-latency causal inference.

## Project Structure

*   `src/`: Source code package.
    *   `config.py`: Hyperparameters and system configuration.
    *   `data.py`: Utilities for loading and simulating high-dimensional ECoG data.
    *   `model.py`: Keras/TensorFlow definitions of the neural architectures.
    *   `train.py`: Training pipeline with ModelCheckpointing and artifact management.
    *   `inference.py`: Real-time inference engine with sliding window buffering.
*   `tests/`: Unit test suite ensuring architectural integrity.

## Usage

### Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

To train the offline decoder on the dataset (currently configured for mock data simulation):

```bash
python3 -m src.train
```

This will generate `neurobridge_decoder_model.h5`.

### Real-Time Inference

To simulate the real-time decoding pipeline:

```bash
python3 -m src.inference
```

This script instantiates the decoder, loads the trained weights, and processes a simulated stream of neural data frame-by-frame.

## Future Directions

*   **Transformer Integration**: Evaluating Conformer or Wav2Vec 2.0 based encoders for potentially superior temporal modeling.
*   **Language Model Decoding**: Integrating a beam-search decoder with a language model to convert phoneme probabilities into coherent text.
*   **Hardware Integration**: Optimizing the inference loop for deployment on low-power edge devices or dedicated DSP hardware.
