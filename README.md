
# NeuroBridge: Speech Neuroprosthesis System

## Scientific Basis

This project implements a state-of-the-art Deep Learning architecture for **Brain-Computer Interface (BCI)** applications, specifically designed to decode intracranial electrocorticography (ECoG) signals into intelligible speech (phonemes).

The core hypothesis is that high-density neural activity in the sensorimotor cortex contains sufficient temporal information to reconstruct the kinematic and acoustic properties of speech. NeuroBridge utilizes a **Conformer (Convolution-augmented Transformer)** architecture to model the complex, non-linear, and temporal dynamics of these neural signals.

This approach aligns with groundbreaking research from leading institutions (e.g., UCSF, Stanford) in the field of speech neuroprosthetics, aiming to restore communication for individuals with severe paralysis or speech deficits (e.g., ALS, brainstem stroke). By combining the local feature extraction of CNNs with the global context of Transformers, the Conformer outperforms traditional RNNs in capturing long-range dependencies.

## Architecture

The system is designed around a unified Conformer-based architecture:

1.  **Offline Decoder (`build_neurobridge_decoder`)**:
    *   **Input**: Time-series ECoG data (Batch, Timesteps, Channels).
    *   **Core**: Stacked **Conformer Blocks**. Each block consists of:
        *   **Feed Forward Module**: Two linear projections with SiLU (Swish) activation.
        *   **Multi-Head Self-Attention (MHSA)**: Captures global temporal context.
        *   **Convolution Module**: Captures local temporal patterns using 1D depthwise convolutions and GLU activation.
    *   **Output**: A TimeDistributed Dense layer with Softmax activation predicting the probability distribution over the phoneme set (including silence) for each timestep.

2.  **Real-Time Decoder (`build_realtime_decoder`)**:
    *   **Constraint**: Low-latency inference.
    *   **Core**: Reuses the **Conformer** architecture. While standard self-attention is non-causal (looks at the whole sequence), this prototype achieves pseudo-real-time operation by processing a **sliding window** of neural data.
    *   **Inference Strategy**: A sliding window buffer maintains the necessary historical context (e.g., 100 timesteps) to allow the Conformer to generate stable predictions for the most recent frame.
    *   **Future Optimization**: A strictly causal Conformer (with masked attention) could be trained to further reduce latency requirements.

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
