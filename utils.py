import os
import contextlib
import logging
from functools import wraps
from pathlib import Path
import pandas as pd # Keep for potential future type hints if needed
import json # Import json

# Get logger for this module
logger = logging.getLogger(__name__)

def suppress_output(func):
    """
    Decorator to suppress stdout/stderr during function execution.
    WARNING: May also suppress console logging depending on handler configuration.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Suppressing output for function: {func.__name__}")
        # Redirect stdout and stderr to os.devnull
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                # No changes needed for logging handlers here if basicConfig was used,
                # as it configures the root logger. If specific handlers were added,
                # they might need temporary removal/replacement here.
                try:
                    result = func(*args, **kwargs)
                finally:
                    logger.debug(f"Restoring output after function: {func.__name__}")
        return result
    return wrapper

def get_filepaths_with_string_and_extension(
    root_directory='.', target_string='', extension=''):
    """Finds files in a directory matching a target string and extension."""
    # Ensure extension starts with a dot if provided and not already present
    if extension and not extension.startswith('.'):
        extension = '.' + extension

    return sorted([
        os.path.abspath(os.path.join(root, file))
        for root, _, files in os.walk(root_directory)
        for file in files
        if target_string in file and (not extension or file.endswith(extension))
    ])

def seconds_to_hhmmss(total_seconds):
    """Converts total seconds to HH:MM:SS format."""
    total_seconds = int(total_seconds) # Ensure integer seconds
    m, s = divmod(total_seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def hhmmss_to_seconds(hms: str) -> int:
    """Converts HH:MM:SS string to total seconds."""
    h, m, s = map(int, hms.split(':'))
    return h * 3600 + m * 60 + s

def check_overlap(interval1_start, interval1_end, interval2_start, interval2_end):
    """Checks if two numeric intervals overlap."""
    return (interval1_start <= interval2_end and interval1_end >= interval2_start) or \
           (interval2_start <= interval1_end and interval2_end >= interval1_start)

def overlap(interval1: dict, interval2: dict) -> bool:
    """Checks if two time intervals (given as dicts with 'start' and 'end' HH:MM:SS strings) overlap."""
    # Assuming consistent HH:MM:SS format allows direct string comparison here
    # For robustness, converting to seconds might be better, but let's stick to the original logic
    return not (interval1['end'] < interval2['start'] or interval1['start'] > interval2['end'])

def load_config(config_path: str = "config.json") -> dict:
    """Loads configuration from a JSON file."""
    path = Path(config_path)
    logger.info(f"Attempting to load configuration from: {config_path}")
    if not path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with open(path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {config_path}: {e}")
        raise ValueError(f"Error decoding JSON from {config_path}: {e}") from e
    except Exception as e:
        logger.exception(f"Failed to load config file {config_path}") # Use logger.exception
        raise RuntimeError(f"Failed to load config file {config_path}: {e}") from e 

def add_api_key_to_config(config_path, api_name, api_key, save=True):
    """
    Add or update an API key in the configuration file.
    
    Args:
        config_path: Path to the config.json file
        api_name: Name of the API (e.g., 'gemini', 'openai')
        api_key: The API key to add
        save: Whether to save changes to the file (default True)
        
    Returns:
        Updated config dictionary
    """
    import json
    from pathlib import Path
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load existing config
        config_path = Path(config_path)
        if not config_path.exists():
            logger.error(f"Config file not found at {config_path}")
            return None
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Ensure api section exists
        if 'api' not in config:
            config['api'] = {}
            
        # Add or update the API key
        config['api'][api_name] = api_key
        logger.info(f"Added/updated {api_name} API key in config")
        
        # Save the updated config if requested
        if save:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved updated config to {config_path}")
            
        return config
        
    except Exception as e:
        logger.exception(f"Error adding API key to config: {e}")
        return None