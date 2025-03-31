from pathlib import Path
import shutil
import torch
import pandas as pd
from transformers import AutoModelForAudioClassification, ASTFeatureExtractor
from typing import Tuple, Optional, Dict, Any
import logging # Import logging

# Import components from other modules
from utils import overlap # Use the utility overlap function
from downloader import YoutubeDownloader
from slicer import AudioSlicer
from separator import SourceSeparator
from analyzer import (
    audio_analyzer,
    mfcc_from_accompanies,
    cluster_mfccs_with_pca
)
from postprocessing import (
    class_manipulator,
    group_same_songs,
    find_and_filter_overlapping_timestamps
)

# Get logger for this module
logger = logging.getLogger(__name__)

class VideoSingingDetector:
    """
    Encapsulates the workflow for detecting singing segments (with accompaniment
    and potentially a cappella) in a YouTube video.

    Loads ML model on initialization and manages temporary files based on config.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the detector, loading the ML model and settings from config.

        Args:
            config: The configuration dictionary loaded from config.json.
        """
        logger.info("Initializing VideoSingingDetector...")
        self.config = config # Store the config

        # Get settings from config, providing defaults or raising errors
        detector_cfg = self.config.get('detector')
        if not detector_cfg:
            logger.critical("Missing 'detector' section in config.")
            raise KeyError("Missing 'detector' section in config.")

        self.base_temp_dir = Path(detector_cfg.get('base_temp_dir', 'temp_processing')) # Default if missing
        model_name = detector_cfg.get('model_name')
        if not model_name:
            logger.critical("Missing 'model_name' in detector config.")
            raise KeyError("Missing 'model_name' in detector config.")
        singing_like_indices = detector_cfg.get('singing_like_indices')
        if singing_like_indices is None:
            logger.critical("Missing 'singing_like_indices' in detector config.")
            raise KeyError("Missing 'singing_like_indices' in detector config.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        logger.info(f"Loading audio classification model ({model_name})...")
        try:
            self.model = AutoModelForAudioClassification.from_pretrained(model_name).to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load audio classification model '{model_name}'", exc_info=True)
            raise RuntimeError(f"Could not load necessary model: {model_name}") from e

        self.classes = self.model.config.id2label
        # Ensure indices are valid for the loaded model
        valid_singing_indices = [idx for idx in singing_like_indices if idx in self.classes]
        if len(valid_singing_indices) != len(singing_like_indices):
             logger.warning("Warning: Some singing_like_indices from config are invalid for the loaded model.")
        self.singing_like_classes = [self.classes[idx] for idx in valid_singing_indices]
        logger.info(f"Identified {len(self.singing_like_classes)} singing-like classes from config.")
        logger.info("VideoSingingDetector initialized successfully.")

    def _cleanup_temp_dir(self, temp_dir_path: Path):
        """Safely removes the temporary directory."""
        if temp_dir_path and temp_dir_path.is_dir(): # Check if it exists and is a directory
            logger.info(f"Cleaning up temporary directory: {temp_dir_path}")
            try:
                shutil.rmtree(temp_dir_path)
                logger.info("Cleanup successful.")
            except OSError as e:
                logger.warning(f"Warning: Error removing temporary directory {temp_dir_path}: {e}")
        else:
             logger.debug(f"Temporary directory {temp_dir_path} not found or not a directory, skipping cleanup.")

    def process_video(
        self, youtube_url: str, visualize_clusters: bool = False, identify_songs: bool = False, use_fast_mode: bool = True
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Processes a single YouTube video URL through all steps.

        Args:
            youtube_url: The URL of the YouTube video.
            visualize_clusters: Whether to show PCA/Energy plots during MFCC clustering.
            identify_songs: Whether to attempt song identification.
            use_fast_mode: Whether to use the fast mode (no Demucs).

        Returns:
            A tuple containing two pandas DataFrames:
            (higher_probability_timestamps, possible_a_capella)
            Returns (None, None) if processing fails at any critical step.
        """
        logger.info(f"\n--- Processing video: {youtube_url} ---")
        video_temp_dir = None # Initialize to None
        downloaded_audio_path = None

        # Get configs for different steps
        slicer_cfg = self.config.get('slicer', {})
        separator_cfg = self.config.get('separator', {})
        analyzer_cfg = self.config.get('analyzer', {})
        postproc_cfg = self.config.get('postprocessing', {})

        try:
            # --- Step 0: Get Video Info and Create Temp Dir ---
            logger.info("Fetching video info to create temporary directory...")
            # Use a dummy path initially, we only need the ID
            # Create the downloader instance here, it handles dir creation
            temp_downloader_path = self.base_temp_dir / "___temp_info_fetch___"
            info_fetcher = YoutubeDownloader(youtube_url, temp_downloader_path)
            video_id, video_title = info_fetcher.get_video_info()
            # Immediately clean up the dummy dir created by info_fetcher
            self._cleanup_temp_dir(temp_downloader_path)

            if not video_id:
                logger.error("ERROR: Could not get video ID. Aborting processing for this URL.")
                return None, None # Indicate failure

            # Create the actual temporary directory for this video using the ID
            video_temp_dir = self.base_temp_dir / video_id
            video_temp_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using temporary directory: {video_temp_dir}")

            # --- Step 1: Download ---
            logger.info("\n[Step 1/5] Downloading Audio...")
            downloader = YoutubeDownloader(youtube_url, output_dir=video_temp_dir)
            downloaded_audio_path = downloader.download_audio() # Path is inside video_temp_dir
            if not downloaded_audio_path:
                 logger.error("Audio download failed.") # Error logged by downloader
                 raise RuntimeError("Audio download failed.")
            logger.info(f"Audio downloaded to: {downloaded_audio_path}")

            # --- Step 2: Slice ---
            logger.info("\n[Step 2/5] Slicing Audio...")
            slice_base_name = f"{video_id}_slice" # Base name for slices
            slicer = AudioSlicer(
                input_path=downloaded_audio_path,
                output_dir=video_temp_dir / "slices",
                output_base_name=slice_base_name,
                segment_length=slicer_cfg.get('segment_length_seconds', 600) # Use config value
            )
            wav_slice_paths = slicer.slice_audio()
            if wav_slice_paths is None: # Check for None explicitly
                logger.error("Audio slicing failed.") # Error logged by slicer
                raise RuntimeError("Audio slicing failed.")
            if not wav_slice_paths: # Check for empty list
                 logger.error("Slicing completed but no slice paths found.")
                 raise RuntimeError("Slicing completed but no slice paths found.")
            logger.info(f"{len(wav_slice_paths)} slices created in {video_temp_dir}")

            # --- Step 3: Separate Sources ---
            logger.info("\n[Step 3/5] Separating Sources (Vocals/Accompaniment)...")
            # Demucs output goes into a subdirectory within the video's temp dir
            separation_output_dir = video_temp_dir / "separated"
            separator = SourceSeparator(
                wav_slice_paths,
                output_directory=separation_output_dir,
                config=separator_cfg # Pass config dict
            )
            vocal_paths, accompanies_paths = separator.separate_all()

            # Critical check: Ensure separation actually produced files
            if not vocal_paths:
                logger.error(f"No vocal files found after separation in '{separation_output_dir}'. Check Demucs logs/output.")
                raise FileNotFoundError(f"No vocal files found after separation in '{separation_output_dir}'. Check Demucs logs/output.")
            if not accompanies_paths:
                logger.error(f"No accompaniment files ('{SourceSeparator.ACCOMPANIMENT_FILENAME}') found after separation in '{separation_output_dir}'. Check Demucs logs/output.")
                raise FileNotFoundError(f"No accompaniment files ('{SourceSeparator.ACCOMPANIMENT_FILENAME}') found after separation in '{separation_output_dir}'. Check Demucs logs/output.")
            logger.info(f"Source separation complete. Found {len(vocal_paths)} vocals, {len(accompanies_paths)} accompaniments.")

            # --- Step 4: Analyze Vocals and Accompaniment ---
            logger.info("\n[Step 4/5] Analyzing Separated Tracks...")
            # Vocal Analysis (using AST model)
            logger.info("Analyzing vocal tracks...")
            result_vocal_raw, _ = audio_analyzer(
                vocal_paths,
                self.model,
                self.device,
                slice_duration=analyzer_cfg.get('ast_slice_duration_seconds', 5) # Use config
            )
            logger.info("Grouping vocal segments...")
            vocal_singing_segments = class_manipulator(result_vocal_raw, self.singing_like_classes)
            vocal_result_grouped = group_same_songs(
                vocal_singing_segments,
                interval_threshold=postproc_cfg.get('grouping_interval_threshold_seconds', 5), # Use config
                duration_threshold=postproc_cfg.get('grouping_duration_threshold_seconds', 10) # Use config
            )
            logger.info(f"Found {len(vocal_result_grouped)} potential singing groups based on vocals.")

            # Accompaniment Analysis (using MFCC clustering)
            logger.info("Analyzing accompaniment tracks...")
            mfccs = mfcc_from_accompanies(
                accompanies_paths,
                slice_duration=analyzer_cfg.get('mfcc_slice_duration_seconds', 5), # Use config
                n_mfcc=analyzer_cfg.get('mfcc_n_mfcc', 13) # Use config
            )
            logger.info("Clustering accompaniment segments...")
            accomp_singing_segments, _ = cluster_mfccs_with_pca(
                mfccs,
                visualize=visualize_clusters,
                time_window_length=analyzer_cfg.get('mfcc_slice_duration_seconds', 5), # Use config
                n_components=analyzer_cfg.get('pca_n_components', 2), # Use config
                n_clusters=analyzer_cfg.get('kmeans_n_clusters', 2) # Use config
            )
            accompanies_result_grouped = group_same_songs(
                accomp_singing_segments,
                interval_threshold=postproc_cfg.get('grouping_interval_threshold_seconds', 5), # Use config
                duration_threshold=postproc_cfg.get('grouping_duration_threshold_seconds', 10) # Use config
            )
            logger.info(f"Found {len(accompanies_result_grouped)} potential singing groups based on accompaniment.")

            # --- Step 5: Combine Results and Identify A Cappella ---
            logger.info("\n[Step 5/5] Combining Results and Identifying A Cappella...")
            logger.info("Finding overlapping vocal/accompaniment segments...")
            higher_probability_timestamps = find_and_filter_overlapping_timestamps(
                accompanies_result_grouped,
                vocal_df=vocal_result_grouped, # Pass vocal_df explicitly
                threshold=postproc_cfg.get('overlap_merge_threshold_seconds', 30), # Use config
                too_long_threshold=postproc_cfg.get('too_long_threshold_seconds', 420) # Use config
            )
            logger.info(f"Found {len(higher_probability_timestamps)} segments with high probability of singing + accompaniment.")

            # A Cappella Identification: Find vocal segments that *don't* overlap with the high-probability ones
            logger.info("Identifying potential a cappella segments...")
            non_overlapping_rows = []
            # Ensure vocal_result_grouped is not empty
            if not vocal_result_grouped.empty:
                for _, vocal_row in vocal_result_grouped.iterrows():
                    vocal_interval = {'start': vocal_row['start'], 'end': vocal_row['end']}
                    overlaps_found = False
                    # Ensure higher_probability_timestamps is not empty
                    if not higher_probability_timestamps.empty:
                        for _, hp_row in higher_probability_timestamps.iterrows():
                            hp_interval = {'start': hp_row['start'], 'end': hp_row['end']}
                            # Use the utility overlap function
                            if overlap(vocal_interval, hp_interval):
                                overlaps_found = True
                                break
                    if not overlaps_found:
                        non_overlapping_rows.append(vocal_row)

            possible_a_capella = pd.DataFrame(non_overlapping_rows).reset_index(drop=True)
            # Ensure columns are present even if empty
            if possible_a_capella.empty and not vocal_result_grouped.empty:
                 possible_a_capella = pd.DataFrame(columns=vocal_result_grouped.columns)

            logger.info(f"Found {len(possible_a_capella)} potential a cappella segments.")

            logger.info("\n--- Processing Complete ---")
            # Return the final DataFrames
            return higher_probability_timestamps, possible_a_capella

        except (RuntimeError, FileNotFoundError, Exception) as e:
            # Log the error encountered during processing
            logger.exception(f"\n--- ERROR during processing for {youtube_url} ---")
            logger.exception(f"Error Type: {type(e).__name__}")
            logger.exception(f"Error Details: {e}")
            # Indicate failure by returning None
            return None, None

        finally:
            # --- Cleanup ---
            # Ensure the temporary directory for this specific video is removed
            if video_temp_dir:
                 self._cleanup_temp_dir(video_temp_dir) 