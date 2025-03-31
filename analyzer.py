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
from typing import List, Tuple, Any, Dict
import logging # Import logging
from scipy import signal
from pathlib import Path
import os

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

class AudioAnalyzer:
    """Class for analyzing audio to detect singing segments"""
    
    def __init__(self, config=None):
        """Initialize the audio analyzer with configuration"""
        self.config = config or {}
        self.analyzer_config = self.config.get('analyzer', {})
        
        # Load configuration parameters with defaults
        self.sr = self.analyzer_config.get('sample_rate', 16000)
        self.frame_length = self.analyzer_config.get('frame_length', 0.025)  # in seconds
        self.frame_shift = self.analyzer_config.get('frame_shift', 0.01)     # in seconds
        self.n_mfcc = self.analyzer_config.get('n_mfcc', 13)
        self.n_fft = self.analyzer_config.get('n_fft', 2048)
        self.hop_length = int(self.sr * self.frame_shift)
        self.win_length = int(self.sr * self.frame_length)
        
        # Clustering parameters
        self.n_clusters = self.analyzer_config.get('n_clusters', 2)
        self.min_segment_length = self.analyzer_config.get('min_segment_length', 1.0)  # in seconds
        
        logger.info(f"Initialized AudioAnalyzer with sample rate {self.sr} Hz")
        
    def analyze_file(self, audio_file, visualize=False, use_demucs=True):
        """
        Analyze an audio file to detect singing segments.
        
        Args:
            audio_file: Path to the audio file
            visualize: Whether to visualize the results
            use_demucs: Whether to use Demucs for vocal separation (slower but more accurate)
                       If False, uses a faster frequency-based approach (quicker but less accurate)
        
        Returns:
            Two DataFrames:
            1. Segments detected as singing with instrumental accompaniment
            2. Segments more likely to be a cappella singing
        """
        logger.info(f"Analyzing audio file: {audio_file}, use_demucs={use_demucs}")
        
        if use_demucs:
            # Original implementation using Demucs for source separation
            return self._analyze_with_demucs(audio_file, visualize)
        else:
            # Simplified implementation without Demucs
            return self._analyze_fast(audio_file, visualize)
            
    def _analyze_fast(self, audio_file, visualize=False):
        """
        Fast analysis without using Demucs for source separation.
        Uses frequency filtering to approximate vocal detection.
        
        Args:
            audio_file: Path to the audio file
            visualize: Whether to visualize the results
            
        Returns:
            Two DataFrames for singing segments
        """
        logger.info(f"Using fast analysis mode (no Demucs) for {audio_file}")
        
        # Load audio file
        y, sr = librosa.load(audio_file, sr=self.sr)
        duration = len(y) / sr
        logger.info(f"Loaded audio: {duration:.2f} seconds")
        
        # Apply bandpass filter to emphasize vocal frequencies (roughly 200-3500 Hz)
        vocal_emphasized = self._bandpass_filter(y, sr, low_cutoff=200, high_cutoff=3500)
        
        # Extract features
        mfccs = self._extract_features(vocal_emphasized)
        
        # Additional voice-focused features
        zcr = librosa.feature.zero_crossing_rate(vocal_emphasized, 
                                               frame_length=self.win_length, 
                                               hop_length=self.hop_length)
        
        spectral_centroid = librosa.feature.spectral_centroid(y=vocal_emphasized, 
                                                            sr=sr, 
                                                            n_fft=self.n_fft, 
                                                            hop_length=self.hop_length)
        
        # Normalize and combine features
        zcr_norm = (zcr - np.mean(zcr)) / np.std(zcr)
        spectral_centroid_norm = (spectral_centroid - np.mean(spectral_centroid)) / np.std(spectral_centroid)
        
        # Combine with MFCCs
        features = np.vstack([mfccs, zcr_norm, spectral_centroid_norm])
        
        # Perform clustering
        segments = self._cluster_frames(features.T, duration)
        
        # Since we don't have vocal/instrumental separation in fast mode,
        # we'll do a simpler classification to approximate a cappella vs. accompanied
        # Using energy ratio in mid vs. low/high frequencies as a rough proxy
        
        # Calculate energy in different frequency bands
        stft = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        
        # Define frequency bands
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        low_band = (freq_bins < 300)
        mid_band = (freq_bins >= 300) & (freq_bins <= 3500)  # Voice range
        high_band = (freq_bins > 3500)
        
        # Calculate energy in bands over time
        low_energy = np.sum(stft[low_band, :], axis=0)
        mid_energy = np.sum(stft[mid_band, :], axis=0)
        high_energy = np.sum(stft[high_band, :], axis=0)
        
        # Ratio of mid (voice) to others (likely instruments)
        accompaniment_ratio = mid_energy / (low_energy + high_energy + 1e-10)
        
        # Determine which segments are more likely a cappella vs. accompanied
        # Higher ratio means more voice-dominant (potential a cappella)
        accompanied_segments = []
        acapella_segments = []
        
        for segment in segments:
            start_frame = int(segment['start_time'] / self.frame_shift)
            end_frame = int(segment['end_time'] / self.frame_shift)
            end_frame = min(end_frame, accompaniment_ratio.shape[0])
            
            if start_frame < end_frame:
                mean_ratio = np.mean(accompaniment_ratio[start_frame:end_frame])
                
                # Choose threshold based on experimentation
                # You may need to adjust this threshold
                ratio_threshold = 5.0
                
                if mean_ratio > ratio_threshold:
                    acapella_segments.append(segment)
                else:
                    accompanied_segments.append(segment)
        
        # Convert to DataFrames
        df_accompanied = pd.DataFrame(accompanied_segments)
        df_acapella = pd.DataFrame(acapella_segments)
        
        if visualize:
            self._visualize_results(y, df_accompanied, df_acapella, vocal_emphasized=vocal_emphasized)
        
        return df_accompanied, df_acapella
    
    def _bandpass_filter(self, audio, sr, low_cutoff, high_cutoff):
        """Apply bandpass filter to audio to emphasize vocals"""
        nyquist = 0.5 * sr
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        
        # Create a bandpass filter
        b, a = signal.butter(5, [low, high], btype='band')
        
        # Apply filter
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def _extract_features(self, audio):
        """Extract MFCC features from audio"""
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc,
                                   n_fft=self.n_fft, hop_length=self.hop_length)
        return mfccs
    
    def _cluster_frames(self, features, duration):
        """Cluster frames based on features and identify singing segments"""
        logger.info(f"Clustering frames with {self.n_clusters} clusters")
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(features)
        
        # Determine which cluster corresponds to singing
        # Here we use a simple heuristic: singing clusters generally have higher energy in MFCCs
        cluster_mfcc_energy = []
        for i in range(self.n_clusters):
            cluster_frames = features[cluster_labels == i]
            if len(cluster_frames) > 0:
                cluster_mfcc_energy.append(np.mean(np.abs(cluster_frames[:, :self.n_mfcc])))
            else:
                cluster_mfcc_energy.append(0)
        
        singing_cluster = np.argmax(cluster_mfcc_energy)
        logger.info(f"Identified singing cluster: {singing_cluster}")
        
        # Create segments from consecutive frames
        segments = []
        in_segment = False
        start_frame = 0
        
        for i, label in enumerate(cluster_labels):
            if label == singing_cluster and not in_segment:
                # Start of a new segment
                in_segment = True
                start_frame = i
            elif label != singing_cluster and in_segment:
                # End of a segment
                end_frame = i
                segment_start_time = start_frame * self.frame_shift
                segment_end_time = end_frame * self.frame_shift
                
                # Only keep segments longer than the minimum length
                if segment_end_time - segment_start_time >= self.min_segment_length:
                    segments.append({
                        'start_time': segment_start_time,
                        'end_time': segment_end_time,
                        'duration': segment_end_time - segment_start_time
                    })
                
                in_segment = False
        
        # Don't forget the last segment if we're still in one
        if in_segment:
            end_frame = len(cluster_labels)
            segment_start_time = start_frame * self.frame_shift
            segment_end_time = end_frame * self.frame_shift
            
            if segment_end_time - segment_start_time >= self.min_segment_length:
                segments.append({
                    'start_time': segment_start_time,
                    'end_time': segment_end_time,
                    'duration': segment_end_time - segment_start_time
                })
        
        logger.info(f"Detected {len(segments)} singing segments")
        return segments
    
    # ... existing methods below ...
    def _analyze_with_demucs(self, audio_file, visualize=False):
        """
        Original implementation using Demucs for source separation
        """
        # Your existing Demucs-based implementation here
        # ...
        
    def _visualize_results(self, audio, df_accompanied, df_acapella, vocals=None, vocal_emphasized=None):
        """Visualize the audio waveform and detected segments"""
        plt.figure(figsize=(12, 8))
        
        # Plot audio waveform
        plt.subplot(311)
        plt.plot(np.linspace(0, len(audio) / self.sr, len(audio)), audio)
        plt.title('Original Audio Waveform')
        plt.xlabel('Time (s)')
        
        # Mark singing with accompaniment segments
        for _, row in df_accompanied.iterrows():
            plt.axvspan(row['start_time'], row['end_time'], alpha=0.3, color='red')
        
        # Mark a cappella segments
        for _, row in df_acapella.iterrows():
            plt.axvspan(row['start_time'], row['end_time'], alpha=0.3, color='blue')
        
        # Plot vocals if available
        if vocals is not None:
            plt.subplot(312)
            plt.plot(np.linspace(0, len(vocals) / self.sr, len(vocals)), vocals)
            plt.title('Extracted Vocals')
            plt.xlabel('Time (s)')
        elif vocal_emphasized is not None:
            plt.subplot(312)
            plt.plot(np.linspace(0, len(vocal_emphasized) / self.sr, len(vocal_emphasized)), vocal_emphasized)
            plt.title('Voice-Emphasized Audio (Bandpass Filtered)')
            plt.xlabel('Time (s)')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.3, label='Singing with Accompaniment'),
            Patch(facecolor='blue', alpha=0.3, label='Potential A Cappella')
        ]
        plt.figlegend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)
        
        plt.tight_layout()
        plt.show() 