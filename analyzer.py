import torchaudio
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import librosa
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForAudioClassification, ASTFeatureExtractor
from typing import List, Tuple, Any
import logging # Import logging

# Import utilities from the utils module
from utils import seconds_to_hhmmss

# Get logger for this module
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def get_common_sampling_rate(sampling_rates: List[int]) -> int:
    """Checks if all sampling rates in a list are the same and returns it."""
    unique_rates = set(sampling_rates)
    if len(unique_rates) == 0:
        logger.error("Cannot determine sampling rate from an empty list.")
        raise ValueError("Cannot determine sampling rate from an empty list.")
    if len(unique_rates) == 1:
        rate = int(list(unique_rates)[0])
        logger.debug(f"Common sampling rate: {rate}")
        return rate
    else:
        logger.error(f"Inconsistent sampling rates found: {unique_rates}")
        raise ValueError(f"All sampling rates should be the same, but found: {unique_rates}")

def load_and_slice_audio(
    file_paths: List[str], slice_duration: int = 10
) -> Tuple[List[np.ndarray], List[int]]:
    """Loads audio files, slices them into segments, and returns numpy arrays."""
    all_samples = []
    all_sampling_rates = []

    logger.info(f"Loading and slicing {len(file_paths)} audio files into {slice_duration}s segments...")
    for path in tqdm(file_paths, desc="Loading/Slicing Audio"):
        logger.debug(f"Processing file: {path}")
        try:
            waveform, sampling_rate = torchaudio.load(path)
            logger.debug(f"Loaded {path}: shape={waveform.shape}, rate={sampling_rate}")
            # Ensure waveform is 2D [channels, samples] even if mono
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
                logger.debug(f"Unsqueezed mono waveform to shape: {waveform.shape}")

            # Calculate number of full slices
            num_samples = waveform.shape[1]
            slice_samples = sampling_rate * slice_duration
            num_slices = num_samples // slice_samples
            logger.debug(f"Total samples: {num_samples}, Slice samples: {slice_samples}, Num slices: {num_slices}")

            if num_slices == 0:
                 logger.warning(f"Audio file {path} is shorter ({num_samples/sampling_rate:.2f}s) than slice duration ({slice_duration}s). Skipping slicing, using whole file.")
                 # Handle short files: append the whole waveform as one "slice"
                 # Convert to numpy here for consistency
                 all_samples.append(waveform.numpy())
                 all_sampling_rates.append(sampling_rate)
                 continue # Skip the slicing loop for this file

            # Slice the waveform
            for i in range(num_slices):
                start_sample = i * slice_samples
                end_sample = start_sample + slice_samples
                segment = waveform[:, start_sample:end_sample]
                # Convert to numpy array immediately after slicing
                all_samples.append(segment.numpy())
                all_sampling_rates.append(sampling_rate)
            logger.debug(f"Sliced {path} into {num_slices} segments.")

        except FileNotFoundError:
            logger.error(f"Audio file not found: {path}. Skipping.")
            continue # Skip to the next file
        except Exception as e:
            logger.exception(f"Error loading or slicing audio file {path}. Skipping.")
            continue # Skip to the next file

    if not all_samples:
         logger.error("No audio samples were successfully loaded or sliced.")
         # Decide whether to raise an error or return empty lists
         # raise RuntimeError("Failed to load any audio data.")
         return [], []

    logger.info(f"Successfully loaded and sliced into {len(all_samples)} total segments.")
    return all_samples, all_sampling_rates

def predict_samples(
    model: AutoModelForAudioClassification,
    feature_extractor: ASTFeatureExtractor,
    audio_samples: List[np.ndarray], # Expect list of 1D numpy arrays
    sampling_rate: int,
    device: torch.device,
    batch_size: int = 8 # Add batching for efficiency
) -> Tuple[List[Dict[str, Any]], List[torch.Tensor]]:
    """Predicts classes for audio samples using the AST model with batching."""
    all_predictions = []
    all_raw_outputs = []
    num_samples = len(audio_samples)
    logger.info(f"Starting prediction for {num_samples} audio segments...")

    # Process in batches
    for i in tqdm(range(0, num_samples, batch_size), desc="Predicting Audio Segments"):
        batch_samples = audio_samples[i:i+batch_size]
        logger.debug(f"Processing batch {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size}")

        try:
            # Extract features - expects list of raw audio arrays, sampling rate
            inputs = feature_extractor(
                batch_samples, sampling_rate=sampling_rate, return_tensors="pt"
            )
            inputs = inputs.to(device)
            logger.debug(f"Feature extraction complete for batch, input shape: {inputs['input_values'].shape}")

            # Perform inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.sigmoid(logits) # Use sigmoid for multi-label

            # Move results back to CPU for processing
            probabilities_cpu = probabilities.cpu()
            logits_cpu = logits.cpu()
            all_raw_outputs.extend(list(logits_cpu)) # Store raw logits if needed

            # Process predictions for each item in the batch
            for j in range(probabilities_cpu.shape[0]):
                segment_index = i + j # Original index of the segment
                segment_probs = probabilities_cpu[j]
                # Get top prediction(s) or all classes above a threshold
                # For simplicity, let's take the single highest probability class
                top_prob, top_class_idx = torch.max(segment_probs, dim=0)
                top_class_name = model.config.id2label[top_class_idx.item()]

                prediction_info = {
                    'index': segment_index,
                    'class': top_class_name,
                    'confidence': top_prob.item(),
                    # Optionally include all probabilities:
                    # 'all_probs': {model.config.id2label[k]: prob.item() for k, prob in enumerate(segment_probs)}
                }
                all_predictions.append(prediction_info)
                logger.debug(f"Segment {segment_index}: Predicted '{top_class_name}' (Confidence: {top_prob.item():.4f})")

        except Exception as e:
            logger.exception(f"Error predicting batch starting at index {i}")
            # Optionally add placeholders for failed predictions
            # for k in range(len(batch_samples)):
            #     all_predictions.append({'index': i+k, 'class': 'ERROR', 'confidence': 0.0})
            #     all_raw_outputs.append(torch.zeros_like(logits_cpu[0]) if 'logits_cpu' in locals() else None)


    logger.info(f"Prediction finished. Got {len(all_predictions)} predictions.")
    return all_predictions, all_raw_outputs

def create_dataframe(
    predictions: List[Dict[str, Any]], time_window_length: int
) -> pd.DataFrame:
    """Creates a DataFrame with start/end times from prediction results."""
    if not predictions:
        logger.warning("No predictions provided to create DataFrame.")
        return pd.DataFrame(columns=['index', 'class', 'confidence', 'start', 'end'])

    logger.info(f"Creating DataFrame for {len(predictions)} predictions with time window {time_window_length}s.")
    df = pd.DataFrame(predictions)

    # Calculate start and end times based on index and window length
    df['start_seconds'] = df['index'] * time_window_length
    df['end_seconds'] = (df['index'] + 1) * time_window_length

    # Convert seconds to HH:MM:SS format
    df['start'] = df['start_seconds'].apply(seconds_to_hhmmss)
    df['end'] = df['end_seconds'].apply(seconds_to_hhmmss)

    # Select and order columns
    result_df = df[['index', 'class', 'confidence', 'start', 'end']].copy()
    logger.debug("DataFrame created successfully.")
    return result_df

def audio_analyzer(
    file_paths: List[str],
    model: AutoModelForAudioClassification,
    device: torch.device,
    slice_duration: int = 5,
    normalize: bool = True
) -> Tuple[pd.DataFrame, List[torch.Tensor]]:
    """Loads, slices, analyzes audio files using an AST model."""
    logger.info(f"Starting audio analysis for {len(file_paths)} files...")
    # Use the passed slice_duration (from config)
    sliced_samples_np, sampling_rates = load_and_slice_audio(file_paths, slice_duration)

    if not sliced_samples_np:
         logger.error("Audio loading and slicing resulted in no samples. Cannot proceed with analysis.")
         return pd.DataFrame(), [] # Return empty results

    # Convert stereo to mono by averaging channels
    mono_array_list = [np.mean(sample, axis=0) for sample in sliced_samples_np]
    if not mono_array_list:
         logger.error("Failed to create mono samples, possibly due to loading/slicing errors.")
         raise RuntimeError("Failed to create mono samples.")

    sampling_rate = get_common_sampling_rate(sampling_rates)

    # Initialize feature extractor
    logger.debug(f"Initializing ASTFeatureExtractor with sampling rate {sampling_rate}, normalize={normalize}")
    feature_extractor = ASTFeatureExtractor(
        sampling_rate=sampling_rate, do_normalize=normalize
    )

    # Perform prediction on the mono samples
    predictions, raw_outputs = predict_samples(
        model, feature_extractor, mono_array_list, sampling_rate, device
    )

    # Create DataFrame from predictions
    result_df = create_dataframe(predictions, time_window_length=slice_duration)

    logger.info("Audio analysis complete.")
    return result_df, raw_outputs

# --- Accompaniment Analysis ---

def mfcc_from_accompanies(
    accompanies_paths: List[str],
    slice_duration: int = 5,
    n_mfcc: int = 13
) -> List[np.ndarray]:
    """Calculates MFCCs for accompaniment tracks."""
    logger.info(f"Starting MFCC calculation for {len(accompanies_paths)} accompaniment files...")
    # Use passed slice_duration (from config)
    sliced_samples_np, sampling_rates = load_and_slice_audio(accompanies_paths, slice_duration)

    if not sliced_samples_np:
         logger.error("Loading/slicing accompaniments resulted in no samples. Cannot calculate MFCCs.")
         return []

    # Convert to mono
    mono_array_list = [np.mean(sample, axis=0) for sample in sliced_samples_np]
    if not mono_array_list:
         logger.error("Failed to create mono samples for MFCC calculation.")
         raise RuntimeError("Failed to create mono samples for MFCC calculation.")

    sampling_rate = get_common_sampling_rate(sampling_rates)

    logger.info(f"Calculating MFCCs ({n_mfcc} coeffs) for {len(mono_array_list)} accompaniment segments...")
    mfccs = []
    # Use tqdm for progress, logging happens inside loop if errors occur
    for i, y in enumerate(tqdm(mono_array_list, desc="Calculating MFCCs")):
         try:
              y_float32 = y.astype(np.float32)
              # Use passed n_mfcc (from config)
              mfcc = librosa.feature.mfcc(y=y_float32, sr=sampling_rate, n_mfcc=n_mfcc)
              mfccs.append(mfcc)
         except Exception as e:
              logger.exception(f"Error calculating MFCC for segment {i}. Skipping.")
              mfccs.append(None) # Mark error

    # Filter out None values if any errors occurred
    mfccs_filtered = [mfcc for mfcc in mfccs if mfcc is not None]
    num_errors = len(mfccs) - len(mfccs_filtered)
    if num_errors > 0:
         logger.warning(f"Skipped {num_errors} segments due to MFCC calculation errors.")

    if not mfccs_filtered:
         logger.error("Failed to calculate MFCCs for any segments.")
         # Decide: return empty list or raise error? Let's return empty.
         return []

    logger.info(f"Successfully calculated MFCCs for {len(mfccs_filtered)} segments.")
    return mfccs_filtered


def calculate_rms_energy(audio_segment: np.ndarray) -> float:
    """Calculates the Root Mean Square (RMS) energy of an audio segment."""
    # Ensure input is numpy array
    audio_segment = np.asarray(audio_segment)
    if audio_segment.size == 0:
        return 0.0
    rms = np.sqrt(np.mean(audio_segment**2))
    return float(rms)


def visualize_energy_over_time(mfccs: List[np.ndarray], cluster_labels: np.ndarray):
    """Visualizes the energy (approximated from MFCCs) over time, colored by cluster."""
    if not mfccs or cluster_labels is None or len(mfccs) != len(cluster_labels):
        logger.warning("Cannot visualize energy: Invalid input MFCCs or cluster labels.")
        return

    logger.info("Visualizing approximate energy over time based on MFCCs...")
    energies = [calculate_rms_energy(np.mean(mfcc, axis=1)) for mfcc in mfccs]
    times = np.arange(len(energies)) # Simple time index

    plt.figure(figsize=(15, 5))
    scatter = plt.scatter(times, energies, c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.xlabel("Segment Index (Time)")
    plt.ylabel("Approximate Energy (Mean MFCC RMS)")
    plt.title("Approximate Energy Over Time Colored by Cluster")
    plt.colorbar(scatter, label='Cluster Label')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def cluster_mfccs_with_pca(
    mfccs: List[np.ndarray],
    visualize: bool = False,
    time_window_length: int = 5,
    n_components: int = 2,
    n_clusters: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clusters MFCC features using PCA and KMeans."""
    logger.info("Starting MFCC clustering with PCA...")
    empty_df = pd.DataFrame(columns=['index', 'class', 'start', 'end', 'class_value']) # Define empty structure

    if not mfccs:
        logger.warning("No MFCCs provided for clustering.")
        return empty_df, empty_df

    # Aggregate MFCCs (e.g., mean over time axis)
    logger.debug("Aggregating MFCCs (mean over time axis)...")
    mfcc_features = np.array([np.mean(mfcc, axis=1) for mfcc in mfccs])
    logger.debug(f"Aggregated MFCC feature shape: {mfcc_features.shape}")

    if mfcc_features.shape[0] < n_clusters:
         logger.warning(f"Number of MFCC segments ({mfcc_features.shape[0]}) is less than n_clusters ({n_clusters}). Cannot perform clustering.")
         return empty_df, empty_df

    # Apply PCA
    actual_n_components = min(n_components, mfcc_features.shape[1], mfcc_features.shape[0])
    if actual_n_components < 1:
         logger.warning("Cannot perform PCA with less than 1 component.")
         return empty_df, empty_df

    logger.info(f"Performing PCA ({actual_n_components} components) on MFCC features...")
    pca = PCA(n_components=actual_n_components)
    try:
        reduced_data = pca.fit_transform(mfcc_features)
        logger.debug(f"PCA completed. Reduced data shape: {reduced_data.shape}")
        logger.debug(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    except Exception as e:
        logger.exception("Error during PCA transformation.")
        return empty_df, empty_df

    # Perform KMeans clustering
    logger.info(f"Performing KMeans clustering (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    try:
        kmeans.fit(reduced_data)
        cluster_labels = kmeans.labels_
        logger.debug(f"KMeans fitting complete. Cluster labels shape: {cluster_labels.shape}")
    except Exception as e:
        logger.exception("Error during KMeans clustering.")
        return empty_df, empty_df

    if visualize:
        logger.info("Visualizing clusters...")
        try:
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1] if actual_n_components > 1 else np.zeros_like(reduced_data[:, 0]),
                                  c=cluster_labels, cmap='viridis', alpha=0.6)
            plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Var)')
            if actual_n_components > 1:
                 plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Var)')
            else:
                 plt.ylabel('Value')
            plt.title(f'KMeans Clustering (k={n_clusters}) of MFCC Features after PCA')
            plt.colorbar(scatter, label='Cluster Label')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.show()
            # Optional energy visualization
            # visualize_energy_over_time(mfccs, cluster_labels)
        except Exception as viz_e:
            logger.warning(f"Failed to generate cluster visualization: {viz_e}", exc_info=True)


    # --- Determine 'Singing' cluster based on energy ---
    logger.info("Determining 'Singing' cluster based on mean MFCC energy...")
    cluster_energy = {label: [] for label in range(n_clusters)}
    for i, label in enumerate(cluster_labels):
        # Use energy of the aggregated MFCC feature vector for the segment
        segment_energy = calculate_rms_energy(mfcc_features[i])
        cluster_energy[label].append(segment_energy)

    # Calculate mean energy per cluster
    mean_cluster_energy = {
        label: np.mean(energies) if energies else 0
        for label, energies in cluster_energy.items()
    }
    logger.info(f"Mean energy per cluster: {mean_cluster_energy}")

    # Determine which cluster has higher mean RMS energy
    if not mean_cluster_energy or all(v == 0 for v in mean_cluster_energy.values()):
         logger.warning("Could not calculate mean cluster energy or all energies are zero. Defaulting singing label to 0.")
         singing_label = 0
    else:
         singing_label = max(mean_cluster_energy, key=mean_cluster_energy.get)

    logger.info(f"Identified cluster {singing_label} as likely 'Singing' based on higher energy.")

    # Map the labels to 'Singing' and 'Non-Singing'
    label_mapping = {label: ('Singing' if label == singing_label else 'Non-Singing')
                     for label in range(n_clusters)}
    logger.debug(f"Cluster label mapping: {label_mapping}")

    # Create a temporary DataFrame with cluster results and timestamps
    indices = np.arange(len(cluster_labels))
    # Use the cluster label directly in the 'class' column for create_dataframe
    temp_df_data = [{'index': idx, 'class': label, 'confidence': mean_cluster_energy.get(label, 0)}
                    for idx, label in enumerate(cluster_labels)]

    temp_df = create_dataframe(temp_df_data, time_window_length=time_window_length)
    # Add the mapped string label
    temp_df['class_value'] = temp_df['class'].map(label_mapping)

    # Separate the dataframes based on the mapped 'class_value'
    singing_df = temp_df[temp_df['class_value'] == 'Singing'].reset_index(drop=True)
    non_singing_df = temp_df[temp_df['class_value'] != 'Singing'].reset_index(drop=True)

    logger.info(f"Created {len(singing_df)} 'Singing' segments and {len(non_singing_df)} 'Non-Singing' segments based on MFCC clustering.")
    return singing_df, non_singing_df 