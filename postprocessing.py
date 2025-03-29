import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging # Import logging

# Import utilities from the utils module
from utils import seconds_to_hhmmss, hhmmss_to_seconds, check_overlap, overlap

# Get logger for this module
logger = logging.getLogger(__name__)

def class_manipulator(input_df: pd.DataFrame, singing_like_classes: List[str]) -> pd.DataFrame:
    """
    Groups consecutive segments based on binary classification (singing vs. non-singing)
    and assigns 'Singing' if the group contains any singing-like class.
    """
    logger.info("Starting class manipulation to group consecutive segments...")
    if input_df.empty:
        logger.warning("Input DataFrame for class_manipulator is empty.")
        return pd.DataFrame(columns=['start', 'end', 'class_value'])

    def _get_most_frequent_with_fallback(x: pd.Series) -> str | None:
        """Helper to determine the class of a group."""
        unique_classes = x.unique()
        logger.debug(f"Grouping classes: {unique_classes}")
        if any(cls in singing_like_classes for cls in unique_classes):
            logger.debug("Group contains singing-like class -> Singing")
            return 'Singing'
        logger.debug("Group does not contain singing-like class -> None")
        return None

    # Create a copy to avoid modifying the original DataFrame
    df = input_df.copy()

    # Ensure 'class' column exists
    if 'class' not in df.columns:
        logger.error("Input DataFrame for class_manipulator must contain a 'class' column.")
        raise ValueError("Input DataFrame must contain a 'class' column.")

    logger.debug(f"Singing-like classes considered: {singing_like_classes}")
    # Binarize the 'class' column: 1 if singing-like, 0 otherwise
    df['binarized_class'] = df['class'].apply(
        lambda x: 1 if (x in singing_like_classes) else 0
    )
    logger.debug("Binarized class column created.")

    # Create groups based on consecutive identical binarized classes
    df['group'] = (df['binarized_class'] != df['binarized_class'].shift()).cumsum()
    logger.debug("Consecutive groups identified.")

    # Group by the new 'group' column and aggregate
    logger.debug("Aggregating groups...")
    grouped_df = df.groupby('group').agg(
        start=('start', 'first'),
        end=('end', 'last'),
        # Apply the helper function to the original 'class' values within the group
        class_value=('class', _get_most_frequent_with_fallback)
    ).reset_index(drop=True)
    logger.debug(f"Aggregation complete. {len(grouped_df)} groups found initially.")

    # Drop rows where class_value is None (i.e., groups that were not 'Singing')
    final_df = grouped_df.dropna(subset=['class_value'])
    logger.info(f"Class manipulation finished. Found {len(final_df)} 'Singing' groups.")

    return final_df.reset_index(drop=True)


def group_same_songs(
    input_df: pd.DataFrame,
    interval_threshold: int,
    duration_threshold: int
) -> pd.DataFrame:
    """
    Merges consecutive 'Singing' segments if the gap between them is small
    and filters out groups shorter than a duration threshold.
    """
    logger.info("Starting grouping of singing segments into songs...")
    if input_df.empty:
        logger.warning("Input DataFrame for group_same_songs is empty.")
        return pd.DataFrame(columns=['start', 'end'])

    df = input_df.copy()

    # Ensure required columns exist
    if not all(col in df.columns for col in ['start', 'end']):
         logger.error("Input DataFrame for group_same_songs must contain 'start' and 'end' columns.")
         raise ValueError("Input DataFrame must contain 'start' and 'end' columns.")

    # Convert HH:MM:SS strings to datetime objects for calculations
    try:
        logger.debug("Converting time strings to datetime objects...")
        # Specify format explicitly for robustness
        df['start_dt'] = pd.to_datetime(df['start'], format='%H:%M:%S', errors='raise')
        df['end_dt'] = pd.to_datetime(df['end'], format='%H:%M:%S', errors='raise')
        logger.debug("Time string conversion successful.")
    except ValueError as e:
        logger.error(f"Error parsing time strings in group_same_songs: {e}. Ensure 'start' and 'end' are in HH:MM:SS format.", exc_info=True)
        # Return empty or raise error? Let's return empty.
        return pd.DataFrame(columns=['start', 'end'])

    # Sort by start time
    df = df.sort_values(by='start_dt').reset_index(drop=True)
    logger.debug("DataFrame sorted by start time.")

    # Group consecutive rows if the time gap is within the threshold
    df['group'] = 0
    logger.debug(f"Grouping segments with interval threshold <= {interval_threshold}s...")
    for i in range(1, len(df)):
        # Calculate gap in seconds
        time_gap = (df.loc[i, 'start_dt'] - df.loc[i-1, 'end_dt']).total_seconds()
        if time_gap <= interval_threshold:
            df.loc[i, 'group'] = df.loc[i-1, 'group'] # Assign same group number
        else:
            df.loc[i, 'group'] = df.loc[i-1, 'group'] + 1 # Start a new group
    logger.debug("Initial grouping based on interval threshold complete.")

    # Aggregate groups
    logger.debug("Aggregating grouped segments...")
    grouped_songs = df.groupby('group').agg(
        start=('start', 'first'), # Keep original HH:MM:SS string
        end=('end', 'last'),     # Keep original HH:MM:SS string
        start_dt=('start_dt', 'first'), # Keep datetime for duration calc
        end_dt=('end_dt', 'last')       # Keep datetime for duration calc
    ).reset_index(drop=True)
    logger.debug(f"Aggregation complete. {len(grouped_songs)} potential song groups found.")

    # Calculate duration and filter
    logger.debug(f"Filtering groups shorter than {duration_threshold}s...")
    grouped_songs['duration_seconds'] = (grouped_songs['end_dt'] - grouped_songs['start_dt']).dt.total_seconds()
    filtered_songs = grouped_songs[grouped_songs['duration_seconds'] >= duration_threshold].copy()
    logger.debug(f"Filtering complete. {len(filtered_songs)} groups remaining.")

    # Select final columns
    final_result = filtered_songs[['start', 'end']].reset_index(drop=True)
    logger.info(f"Grouping into songs finished. Found {len(final_result)} final segments.")
    return final_result


def filter_redundant_overlaps(timestamps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filters a list of time intervals (dicts with 'start', 'end' in seconds)
       to remove intervals fully contained within others."""
    if not timestamps:
        return []
    logger.debug(f"Filtering {len(timestamps)} timestamps for redundant overlaps...")
    # Sort by start time, then by end time descending to prioritize larger intervals
    sorted_timestamps = sorted(timestamps, key=lambda x: (x['start'], -x['end']))

    filtered = []
    if not sorted_timestamps: return filtered

    # Add the first interval (which is the earliest starting, and longest if starts are equal)
    filtered.append(sorted_timestamps[0])

    for current in sorted_timestamps[1:]:
        last = filtered[-1]
        # If the current interval is fully contained within the last added interval, skip it
        if current['start'] >= last['start'] and current['end'] <= last['end']:
            logger.debug(f"Removing redundant interval: {current} (contained in {last})")
            continue
        filtered.append(current)

    logger.debug(f"Filtering complete. {len(filtered)} non-redundant intervals remain.")
    return filtered


def merge_overlapping_times_df(df: pd.DataFrame) -> pd.DataFrame:
    """Merges overlapping or adjacent time intervals in a DataFrame (start/end in seconds)."""
    if df.empty:
        return df
    logger.debug(f"Merging {len(df)} overlapping/adjacent intervals...")
    # Sort by start time
    df = df.sort_values(by='start').reset_index(drop=True)

    merged = []
    current_start = df.loc[0, 'start']
    current_end = df.loc[0, 'end']

    for i in range(1, len(df)):
        next_start = df.loc[i, 'start']
        next_end = df.loc[i, 'end']

        # Check for overlap or adjacency (next_start <= current_end)
        if next_start <= current_end:
            # Merge: extend the current end time if the next interval ends later
            current_end = max(current_end, next_end)
            logger.debug(f"Merging interval {i} into previous. New end: {current_end}")
        else:
            # No overlap: finalize the previous interval and start a new one
            merged.append({'start': current_start, 'end': current_end})
            logger.debug(f"Finalized merged interval: start={current_start}, end={current_end}")
            current_start = next_start
            current_end = next_end

    # Add the last merged interval
    merged.append({'start': current_start, 'end': current_end})
    logger.debug(f"Finalized last merged interval: start={current_start}, end={current_end}")

    merged_df = pd.DataFrame(merged)
    logger.debug(f"Merging complete. Resulted in {len(merged_df)} intervals.")
    return merged_df


def find_and_filter_overlapping_timestamps(
    accompanies_df: pd.DataFrame,
    vocal_df: pd.DataFrame,
    threshold: int = 0,
    too_long_threshold: int = 420
) -> pd.DataFrame:
    """
    Finds intervals where singing is detected by both vocal and accompaniment analysis,
    merges them, and filters based on proximity and duration.
    """
    logger.info("Finding and filtering overlapping vocal/accompaniment timestamps...")
    final_cols = ['start', 'end', 'SameGroupPossibility', 'TooLong'] # Define expected output columns
    empty_result = pd.DataFrame(columns=final_cols)

    if accompanies_df.empty or vocal_df.empty:
        logger.warning("One or both input DataFrames are empty. Cannot find overlaps.")
        return empty_result

    unique_overlapping_timestamps = []

    # Convert HH:MM:SS to seconds for comparison
    try:
        logger.debug("Converting time strings to seconds for overlap check...")
        acc_starts = accompanies_df['start'].apply(hhmmss_to_seconds)
        acc_ends = accompanies_df['end'].apply(hhmmss_to_seconds)
        voc_starts = vocal_df['start'].apply(hhmmss_to_seconds)
        voc_ends = vocal_df['end'].apply(hhmmss_to_seconds)
        logger.debug("Time string conversion successful.")
    except Exception as e:
         logger.exception("Error converting time strings to seconds in find_and_filter_overlapping_timestamps.")
         return empty_result

    # Iterate through all pairs of intervals
    logger.debug("Iterating through interval pairs to find overlaps...")
    overlap_count = 0
    for i in range(len(accompanies_df)):
        for j in range(len(vocal_df)):
            start1_s, end1_s = acc_starts.iloc[i], acc_ends.iloc[i]
            start2_s, end2_s = voc_starts.iloc[j], voc_ends.iloc[j]

            # Check for overlap using the numeric intervals
            if check_overlap(start1_s, end1_s, start2_s, end2_s):
                overlap_count += 1
                # If overlap, create a new interval spanning the union
                overlap_start = min(start1_s, start2_s)
                overlap_end = max(end1_s, end2_s)
                overlapping_timestamp = {'start': overlap_start, 'end': overlap_end}
                # Add if not already present (simple check)
                if overlapping_timestamp not in unique_overlapping_timestamps:
                    unique_overlapping_timestamps.append(overlapping_timestamp)
    logger.debug(f"Found {overlap_count} raw overlaps, resulting in {len(unique_overlapping_timestamps)} unique union intervals.")

    # Filter out intervals fully contained within others
    filtered_timestamps = filter_redundant_overlaps(unique_overlapping_timestamps)

    # Check if filtered_timestamps is empty before creating DataFrame
    if not filtered_timestamps:
        logger.info("No overlapping timestamps found after filtering.")
        return empty_result

    # Create DataFrame from the filtered list (start/end are in seconds)
    merged_result_df = pd.DataFrame(filtered_timestamps)
    logger.debug(f"Created DataFrame from {len(filtered_timestamps)} filtered overlapping intervals.")

    # Merge overlapping/adjacent intervals again after the initial filtering
    result_seconds = merge_overlapping_times_df(merged_result_df)

    # Check if result is empty after merging
    if result_seconds.empty:
         logger.info("Resulting DataFrame is empty after merging overlaps.")
         return empty_result

    # --- Calculate additional flags ---
    logger.debug("Calculating 'SameGroupPossibility' and 'TooLong' flags...")
    if len(result_seconds) > 1:
        time_diff_to_next = result_seconds['start'].shift(-1) - result_seconds['end']
        # Use passed threshold (from config)
        result_seconds['SameGroupPossibility'] = time_diff_to_next < threshold
        logger.debug(f"Calculated 'SameGroupPossibility' using threshold {threshold}s.")
    else:
        result_seconds['SameGroupPossibility'] = False
        logger.debug("Only one interval, 'SameGroupPossibility' set to False.")

    result_seconds['SameGroupPossibility'] = result_seconds['SameGroupPossibility'].fillna(False)

    # Use passed too_long_threshold (from config)
    result_seconds['TooLong'] = (result_seconds['end'] - result_seconds['start']) > too_long_threshold
    logger.debug(f"Calculated 'TooLong' flag using threshold {too_long_threshold}s.")

    # Convert final start/end times from seconds back to HH:MM:SS format
    logger.debug("Converting final timestamps back to HH:MM:SS format...")
    try:
        final_result = pd.DataFrame({
            'start': result_seconds['start'].apply(seconds_to_hhmmss),
            'end': result_seconds['end'].apply(seconds_to_hhmmss),
            'SameGroupPossibility': result_seconds['SameGroupPossibility'],
            'TooLong': result_seconds['TooLong']
        })
        logger.info(f"Overlap finding and filtering complete. Found {len(final_result)} final segments.")
        return final_result
    except Exception as e:
        logger.exception("Error converting final timestamps back to HH:MM:SS.")
        return empty_result # Return empty df on conversion error 