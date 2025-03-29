from pathlib import Path
import subprocess
from tqdm.auto import tqdm
import os
from typing import List, Tuple, Dict, Any
import logging # Import logging

# Import utilities from the utils module
from utils import get_filepaths_with_string_and_extension

# Get logger for this module
logger = logging.getLogger(__name__)

class SourceSeparator:
    """
    Separates audio files into vocals and accompaniment using Demucs,
    renames accompaniment to 'accompaniment.wav', and skips if already done.
    """

    def __init__(self, input_paths: List[Path | str], output_directory: Path | str, config: Dict[str, Any]):
        """
        Initializes the source separator using settings from config.

        Args:
            input_paths: A list of paths to the audio files to be separated.
            output_directory: The base directory where separated files will be stored.
            config: The separator configuration dictionary (e.g., config['separator']).
        """
        self.input_paths = [Path(p) for p in input_paths]
        self.output_directory = Path(output_directory)
        self.config = config
        logger.info(f"Initializing SourceSeparator for {len(self.input_paths)} files.")
        logger.info(f"Output directory: {self.output_directory}")

        # Get settings from config
        self.demucs_model_subdir = self.config.get('demucs_model_subdir', 'htdemucs')
        self.accompaniment_filename = self.config.get('accompaniment_filename', 'accompaniment.wav')
        self.demucs_timeout = self.config.get('demucs_timeout_seconds', 600)
        logger.info(f"Demucs model subdir: {self.demucs_model_subdir}, Accomp filename: {self.accompaniment_filename}, Timeout: {self.demucs_timeout}s")

        self.vocal_paths = []
        self.accomp_paths = []

    def _get_demucs_output_paths(self, input_path: Path) -> Tuple[Path, Path]:
        """Determines the paths where demucs initially creates the files."""
        input_name = input_path.stem
        base_out = self.output_directory / self.demucs_model_subdir / input_name
        paths = (base_out / "vocals.wav", base_out / "no_vocals.wav")
        logger.debug(f"Demucs initial output paths for {input_path.name}: {paths}")
        return paths

    def _get_final_expected_paths(self, input_path: Path) -> Tuple[Path, Path]:
        """Determines the final expected paths after potential renaming."""
        input_name = input_path.stem
        base_out = self.output_directory / self.demucs_model_subdir / input_name
        paths = (base_out / "vocals.wav", base_out / self.accompaniment_filename)
        logger.debug(f"Final expected paths for {input_path.name}: {paths}")
        return paths

    def _check_exists(self, input_path: Path) -> bool:
        """Checks if the *final* expected files (vocals.wav, accompaniment.wav) exist."""
        vocal_path, accomp_path = self._get_final_expected_paths(input_path)
        logger.debug(f"Checking existence for {input_path.name}: Vocal='{vocal_path}', Accomp='{accomp_path}'")
        if vocal_path.exists() and accomp_path.exists():
            logger.debug(f"Both final files exist for {input_path.name}.")
            return True

        # Optional: Check if original demucs output exists but rename hasn't happened
        _, demucs_accomp_path = self._get_demucs_output_paths(input_path)
        if vocal_path.exists() and demucs_accomp_path.exists():
            logger.warning(f"Found '{demucs_accomp_path.name}' but not '{accomp_path.name}' for {input_path.name}. Attempting rename.")
            try:
                # Ensure parent directory exists before renaming
                accomp_path.parent.mkdir(parents=True, exist_ok=True)
                demucs_accomp_path.rename(accomp_path)
                logger.info(f"Rename successful: {demucs_accomp_path} -> {accomp_path}")
                return True # Now it exists
            except OSError as e:
                logger.error(f"Failed to rename {demucs_accomp_path} to {accomp_path}: {e}")
                # Treat as not existing if rename fails
                return False
        logger.debug(f"Final files do not exist for {input_path.name}.")
        return False

    def _perform_separation(self, input_path: Path) -> bool:
        """Runs demucs and renames 'no_vocals.wav'."""
        logger.info(f"Running Demucs separation for '{input_path.name}'...")
        command = [
            'python', '-m', 'demucs',
            '--two-stems', 'vocals',
            '-o', str(self.output_directory),
            # Optionally add model name if needed: '-n', self.demucs_model_subdir,
            str(input_path)
        ]
        logger.debug(f"Executing Demucs command: {' '.join(command)}")
        try:
            process = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.demucs_timeout
            )
            logger.debug(f"Demucs process completed for {input_path.name}.")
            # Log stderr which often contains progress/info from demucs
            if process.stderr:
                 logger.debug(f"Demucs stderr:\n{process.stderr}")

            # --- Rename 'no_vocals.wav' ---
            demucs_vocal_path, demucs_accomp_path = self._get_demucs_output_paths(input_path)
            final_vocal_path, final_accomp_path = self._get_final_expected_paths(input_path)

            # Check if rename is needed (i.e., target name is different and original exists)
            if demucs_accomp_path != final_accomp_path and demucs_accomp_path.exists():
                logger.info(f"Renaming '{demucs_accomp_path.name}' to '{final_accomp_path.name}'...")
                try:
                    # Ensure parent directory exists
                    final_accomp_path.parent.mkdir(parents=True, exist_ok=True)
                    demucs_accomp_path.rename(final_accomp_path)
                    logger.info("Rename successful.")
                except OSError as e:
                    logger.error(f"Failed to rename {demucs_accomp_path} to {final_accomp_path}: {e}")
                    return False # Consider separation failed if rename fails
            elif not demucs_accomp_path.exists():
                 logger.warning(f"Demucs completed but expected output '{demucs_accomp_path}' not found. Cannot rename.")
                 # Decide if this is a failure. Let's assume it is for now.
                 return False
            else:
                 logger.debug(f"No rename needed for accompaniment file: {final_accomp_path.name}")
            # --- End Renaming Step ---

            # Verify final files exist
            if final_vocal_path.exists() and final_accomp_path.exists():
                 logger.info(f"Separation and renaming successful for {input_path.name}.")
                 return True
            else:
                 logger.error(f"Separation seemed successful, but final expected files not found: Vocal='{final_vocal_path}', Accomp='{final_accomp_path}'")
                 return False

        except subprocess.CalledProcessError as e:
            logger.error(f"Demucs failed for {input_path.name} with return code {e.returncode}.")
            logger.error(f"Command: {' '.join(command)}")
            # Log stderr from demucs for debugging
            if e.stderr:
                logger.error(f"Demucs stderr:\n{e.stderr}")
            if e.stdout: # Log stdout too if it exists
                logger.error(f"Demucs stdout:\n{e.stdout}")
            return False
        except subprocess.TimeoutExpired:
             logger.error(f"Demucs timed out processing {input_path.name} after {self.demucs_timeout} seconds.")
             return False
        except FileNotFoundError:
            logger.critical("'python -m demucs' command failed. Is Python/Demucs installed and in PATH?")
            raise # Critical error
        except Exception as e:
             logger.exception(f"An unexpected error occurred during separation for {input_path.name}")
             return False

    def separate_all(self) -> Tuple[List[str], List[str]]:
        """
        Orchestrates separation for all input files, skipping existing ones.

        Returns:
            A tuple containing two lists of absolute paths:
            (list_of_vocal_paths, list_of_accompaniment_paths)
        """
        logger.info(f"Starting source separation process for {len(self.input_paths)} files...")
        processed_count = 0
        skipped_count = 0
        failed_files = []

        # Use tqdm for progress bar, logging happens inside the loop methods
        for file_path in tqdm(self.input_paths, desc="Separating sources"):
            logger.debug(f"Processing file: {file_path.name}")
            if self._check_exists(file_path):
                logger.info(f"Skipping already separated file: {file_path.name}")
                skipped_count += 1
                continue

            if self._perform_separation(file_path):
                processed_count += 1
            else:
                # Error message already logged in _perform_separation
                logger.warning(f"Separation failed for: {file_path.name}")
                failed_files.append(file_path.name)

        logger.info(f"Separation summary: Processed={processed_count}, Skipped={skipped_count}, Failed={len(failed_files)}")
        if failed_files:
             logger.warning(f"Failed files: {', '.join(failed_files)}")
             # Optionally raise an error if any files failed
             # raise RuntimeError(f"{len(failed_files)} file(s) failed during source separation.")

        # Collect final paths after processing all files
        logger.info(f"Collecting final separated file paths from: {self.output_directory}")
        self.vocal_paths = get_filepaths_with_string_and_extension(
            root_directory=str(self.output_directory),
            target_string='vocals',
            extension='wav'
        )
        self.accomp_paths = get_filepaths_with_string_and_extension(
            root_directory=str(self.output_directory),
            target_string=self.accompaniment_filename.replace('.wav',''),
            extension='wav'
        )

        logger.info(f"Found {len(self.vocal_paths)} vocal files and {len(self.accomp_paths)} accompaniment files.")

        # Add a check if the number of vocals/accompaniments matches expected count
        expected_count = processed_count + skipped_count
        if len(self.vocal_paths) != expected_count or len(self.accomp_paths) != expected_count:
             logger.warning(f"Number of found vocal ({len(self.vocal_paths)}) or accompaniment ({len(self.accomp_paths)}) files does not match the expected count ({expected_count}). There might be issues with separation or file collection.")

        return self.vocal_paths, self.accomp_paths 