from pathlib import Path
import subprocess
import logging
# import contextlib # No longer directly used
import os
from typing import List

# Import utilities from the utils module
from utils import suppress_output, get_filepaths_with_string_and_extension

# Get logger for this module
logger = logging.getLogger(__name__)

class AudioSlicer:
    """Slices an audio file into segments using ffmpeg, skipping if already done."""

    def __init__(self, input_path: Path, output_dir: Path, output_base_name: str, segment_length: int = 600):
        """
        Initializes the audio slicer.

        Args:
            input_path: Path to the input audio file.
            output_dir: Directory to save the sliced audio segments.
            output_base_name: Base name for the output files (e.g., 'videoID_slice').
                               The slicer will append '%03d.wav'.
            segment_length: Desired length of each segment in seconds.
        """
        input_path = Path(input_path)
        logger.info(f"Initializing AudioSlicer for input: {input_path}")
        if not input_path.exists():
             logger.error(f"Input audio file not found: {input_path}")
             raise FileNotFoundError(f"Input audio file not found: {input_path}")
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.output_base_name = output_base_name
        self.segment_length = segment_length
        logger.info(f"Output directory: {self.output_dir}, Base name: {output_base_name}, Segment length: {segment_length}s")

        # Ensure output directory exists
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {self.output_dir}: {e}")
            raise

        # Define the full pattern for output files within the output_dir
        self.output_pattern_full = self.output_dir / f"{self.output_base_name}%03d.wav"
        # Format the first expected file path to check for existence
        self.first_expected_output = Path(str(self.output_pattern_full) % 0)
        logger.debug(f"Output pattern: {self.output_pattern_full}")
        logger.debug(f"First expected output file: {self.first_expected_output}")

    def _check_exists(self) -> bool:
        """Checks if the first expected slice exists."""
        exists = self.first_expected_output.exists()
        logger.debug(f"Checking existence of first slice '{self.first_expected_output}': {exists}")
        return exists

    # Use suppress_output carefully, ffmpeg can print useful errors to stderr
    # Consider removing suppress_output if ffmpeg errors need to be logged directly
    # @suppress_output
    def _perform_slicing(self) -> bool:
        """Executes the ffmpeg command to slice the audio."""
        logger.info(f"Starting audio slicing for {self.input_path.name}...")
        command = [
            'ffmpeg',
            '-i', str(self.input_path), # Input file
            '-f', 'segment',            # Segment muxer
            '-segment_time', str(self.segment_length), # Duration of each segment
            '-c', 'copy',               # Copy codec (faster if possible, no re-encoding)
            '-reset_timestamps', '1',   # Reset timestamps for each segment
            '-map', '0:a',              # Map only audio streams
            str(self.output_pattern_full) # Output pattern
        ]
        logger.debug(f"Executing ffmpeg command: {' '.join(command)}")
        try:
            # Run ffmpeg. Capture output to check stderr on failure.
            process = subprocess.run(
                command,
                check=True,          # Raise error if ffmpeg fails
                capture_output=True, # Capture stdout/stderr
                text=True            # Decode output as text
            )
            # ffmpeg usually prints info to stderr, even on success
            logger.debug(f"ffmpeg stderr:\n{process.stderr}")
            logger.info(f"ffmpeg slicing completed successfully for {self.input_path.name}.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg slicing failed for {self.input_path.name}.")
            logger.error(f"Command: {' '.join(command)}")
            logger.error(f"Return code: {e.returncode}")
            logger.error(f"ffmpeg stderr:\n{e.stderr}")
            logger.error(f"ffmpeg stdout:\n{e.stdout}")
            return False
        except FileNotFoundError:
            logger.critical("ffmpeg command not found. Is ffmpeg installed and in the system PATH?")
            raise # Critical error
        except Exception as e:
            logger.exception(f"An unexpected error occurred during slicing for {self.input_path.name}")
            return False

    def get_slice_paths(self) -> List[str]:
        """Retrieves the absolute paths of all generated slices, sorted."""
        logger.debug(f"Searching for slice files in '{self.output_dir}' with base name '{self.output_base_name}'")
        paths = get_filepaths_with_string_and_extension(
            root_directory=str(self.output_dir),
            target_string=self.output_base_name,
            extension='.wav'
        )
        logger.debug(f"Found {len(paths)} slice files.")
        return paths

    def slice_audio(self) -> List[str] | None:
        """
        Orchestrates the slicing process: checks existence, slices if needed.

        Returns:
            A list of absolute paths to the sliced WAV files, or None if slicing failed.
        """
        logger.info(f"Processing audio slicing for: {self.input_path.name}")
        if self._check_exists():
            logger.info(f"Slices already exist for {self.input_path.name}. Retrieving paths.")
            slice_paths = self.get_slice_paths()
            if not slice_paths:
                 # This case might indicate an issue (first slice exists, but others don't match pattern)
                 logger.warning(f"First slice exists, but failed to find slice files matching pattern '{self.output_base_name}*.wav' in {self.output_dir}")
                 return [] # Return empty list, or None? Let's return empty.
            logger.info(f"Found {len(slice_paths)} existing audio slices.")
            return slice_paths

        # Perform slicing if first slice doesn't exist
        logger.info(f"Slices not found for {self.input_path.name}. Performing slicing.")
        if not self._perform_slicing():
            # Error already logged in _perform_slicing
            return None # Indicate failure

        # Verify and get paths after slicing
        slice_paths = self.get_slice_paths()
        if not slice_paths:
             # This could happen if slicing seemed to succeed but no files were found
             logger.error(f"Slicing reported success, but no files found matching pattern '{self.output_base_name}*.wav' in {self.output_dir}")
             return None # Indicate failure

        logger.info(f"Successfully created {len(slice_paths)} audio slices.")
        return slice_paths 