from pathlib import Path
import shutil
import argparse
from typing import Optional
import logging # Import logging

# Import the main detector class and config loader
from detector import VideoSingingDetector
from utils import load_config # Import the loader

# Default config path
DEFAULT_CONFIG_PATH = "config.json"

# --- Setup Logging ---
# We configure logging level based on CLI argument later in main()
# Basic config can be set here, but level/handlers might be adjusted
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
# Get a logger for this module
logger = logging.getLogger(__name__)

def main(youtube_url: str, config_path: str, base_temp_dir_override: Optional[str] = None, visualize: bool = False, cleanup_base: bool = False, log_level: str = 'INFO'):
    """
    Main function to run the singing detection process.

    Args:
        youtube_url: The URL of the YouTube video to process.
        config_path: Path to the configuration JSON file.
        base_temp_dir_override: Optional override for the base temporary directory.
        visualize: Whether to show clustering visualization plots.
        cleanup_base: Whether to remove the entire base_temp_dir before starting.
        log_level: The logging level (e.g., 'DEBUG', 'INFO', 'WARNING').
    """
    # --- Configure Logging Level ---
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        logger.warning(f"Invalid log level: {log_level}. Defaulting to INFO.")
        numeric_level = logging.INFO
    # Set the root logger level
    logging.getLogger().setLevel(numeric_level)
    # Example: Configure a specific handler's level if needed
    # for handler in logging.getLogger().handlers:
    #     handler.setLevel(numeric_level)
    logger.info(f"Logging level set to {log_level.upper()}")

    # Load configuration
    try:
        config = load_config(config_path)
        logger.debug(f"Configuration loaded: {config}") # Log config at DEBUG level
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Error loading configuration: {e}")
        return # Exit if config fails to load

    # Determine base temp directory: override > config > default (handled in detector)
    base_temp_dir = base_temp_dir_override if base_temp_dir_override else config.get('detector', {}).get('base_temp_dir', 'temp_processing') # Use config value
    logger.info(f"Using base temporary directory: {base_temp_dir}")

    # Optional: Clear previous base temp dir content if requested
    base_dir_path = Path(base_temp_dir)
    if cleanup_base and base_dir_path.exists():
        logger.info(f"Cleaning up base temporary directory: {base_dir_path}...")
        try:
            shutil.rmtree(base_dir_path)
            logger.info("Base temporary directory cleaned.")
        except OSError as e:
            logger.warning(f"Could not remove base temporary directory {base_dir_path}: {e}")

    # Initialize the detector (loads the model, uses config)
    try:
        # Pass the loaded config dictionary
        detector = VideoSingingDetector(config=config)
    except (RuntimeError, KeyError) as e: # Catch KeyError if config is malformed
         logger.critical(f"Failed to initialize detector: {e}", exc_info=True) # Use critical for init failure
         return # Exit if detector fails to initialize

    # Process the video
    logger.info(f"Starting processing for video: {youtube_url}")
    results_with_accomp, results_a_capella = detector.process_video(
        youtube_url,
        visualize_clusters=visualize
    )

    # Display results if successful
    if results_with_accomp is not None and results_a_capella is not None:
        logger.info("\n\n--- FINAL RESULTS ---") # Use logger.info

        logger.info("\nSegments with High Probability of Singing with Accompaniment:") # Use logger.info
        if results_with_accomp.empty:
            logger.info("None found.") # Use logger.info
        else:
            # Log the dataframe content (might be long for DEBUG/INFO)
            # Consider logging only summary info or saving to file
            logger.info(f"\n{results_with_accomp.to_string()}") # Log full df string
            # Optional: Save to CSV or JSON
            # results_with_accomp.to_csv("results_singing_with_accompaniment.csv", index=False)
            # logger.info("\nJSON format:")
            # logger.info(results_with_accomp.to_json(orient='records', indent=2))

        logger.info("\nSegments with Potential A Cappella Singing:") # Use logger.info
        if results_a_capella.empty:
             logger.info("None found.") # Use logger.info
        else:
            logger.info(f"\n{results_a_capella.to_string()}") # Log full df string
            # Optional: Save to CSV or JSON
            # results_a_capella.to_csv("results_a_capella.csv", index=False)
            # logger.info("\nJSON format:")
            # logger.info(results_a_capella.to_json(orient='records', indent=2))
    else:
        logger.error(f"\nVideo processing failed for {youtube_url}. No results generated.") # Use logger.error


if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Detect singing segments in a YouTube video.")
    parser.add_argument("youtube_url", help="The URL of the YouTube video to process.")
    parser.add_argument(
        "-c", "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the configuration JSON file (default: {DEFAULT_CONFIG_PATH})"
    )
    parser.add_argument(
        "-t", "--temp-dir",
        default=None,
        help="Override the base directory for temporary files specified in config."
    )
    parser.add_argument(
        "-v", "--visualize",
        action="store_true",
        help="Show MFCC clustering visualization plots."
    )
    parser.add_argument(
        "--cleanup-all",
        action="store_true",
        help="Remove the entire base temporary directory before starting."
    )
    parser.add_argument(
        "--log-level",
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Set the logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(
        youtube_url=args.youtube_url,
        config_path=args.config,
        base_temp_dir_override=args.temp_dir,
        visualize=args.visualize,
        cleanup_base=args.cleanup_all,
        log_level=args.log_level # Pass log level
    )

    # Example of how to run from command line:
    # python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --log-level DEBUG
    # python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --visualize --cleanup-all -t my_temp_files
