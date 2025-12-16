import numpy as np
import tensorflow as tf
import naplib as nl
from sklearn.cluster import KMeans

# --- Constants ---
NUM_TIMESTEPS = 100
NUM_FEATURES = 10  # Updated to match naplib sample data (10 channels)
NUM_PHONEMES = 41

# --- Data Loading Function (The New Implementation) ---
def load_real_data_and_generate_labels(timesteps=NUM_TIMESTEPS, num_classes=NUM_PHONEMES):
    """
    Loads real speech task data using naplib, generates surrogate phoneme labels
    via K-Means clustering on the auditory spectrograms, and segments the data
    into windows for training.
    """
    print("Loading real speech task data from naplib...")
    data = nl.io.load_speech_task_data()

    # Extract continuous streams
    aud_list = [d['aud'] for d in data]
    resp_list = [d['resp'] for d in data]

    # 1. Generate Surrogate Phoneme Labels using K-Means on Auditory Spectrograms
    print(f"Generating {num_classes} surrogate phoneme labels using K-Means clustering on auditory spectrograms...")
    # Concatenate all audio to fit the clusterer
    aud_concat = np.concatenate(aud_list, axis=0)

    # Fit K-Means
    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=3)
    labels_concat = kmeans.fit_predict(aud_concat)

    # Split labels back into trials
    labels_list = []
    current_idx = 0
    for aud_trial in aud_list:
        length = aud_trial.shape[0]
        labels_list.append(labels_concat[current_idx : current_idx + length])
        current_idx += length

    # 2. Segment Data into Windows
    print(f"Segmenting data into windows of {timesteps} timesteps...")

    X_segments = []
    y_segments = []

    # We use a sliding window with overlap (stride = timesteps // 2) to generate more samples
    stride = timesteps // 2

    for resp, labels in zip(resp_list, labels_list):
        n_points = resp.shape[0]
        # Ensure we have enough points for at least one window
        if n_points < timesteps:
            continue

        for i in range(0, n_points - timesteps + 1, stride):
            # Extract window
            X_window = resp[i : i + timesteps, :]
            y_window_sparse = labels[i : i + timesteps]

            X_segments.append(X_window)
            y_segments.append(y_window_sparse)

    X = np.array(X_segments).astype(np.float32)
    y_sparse = np.array(y_segments)

    # One-hot encode targets
    y_onehot = tf.keras.utils.to_categorical(y_sparse, num_classes=num_classes)

    print(f"Generated dataset shapes: X={X.shape}, y={y_onehot.shape}")
    return X, y_onehot

# --- Model Definition (Simplified from neurobridge.py) ---
def build_model(timesteps, features, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(timesteps, features)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation='softmax'))
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Main Verification Logic ---
if __name__ == "__main__":
    # 1. Load Data
    print("Step 1: Loading Real Data...")
    X_train, y_train = load_real_data_and_generate_labels()

    # 2. Build Model
    print("\nStep 2: Building Model...")
    model = build_model(NUM_TIMESTEPS, NUM_FEATURES, NUM_PHONEMES)
    model.summary()

    # 3. Train Model
    print("\nStep 3: Training Model (1 Epoch)...")
    try:
        model.fit(X_train, y_train, batch_size=32, epochs=1, validation_split=0.1)
        print("\nSUCCESS: Training completed without errors using real data.")
    except Exception as e:
        print(f"\nFAILURE: Training failed. Error: {e}")
        exit(1)
