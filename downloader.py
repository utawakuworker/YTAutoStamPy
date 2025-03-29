import yt_dlp
from pathlib import Path
import logging
# import contextlib # No longer directly used
import os
import time # Import time for potential retries/sleeps

# Get logger for this module
logger = logging.getLogger(__name__)

# Note: suppress_output decorator is not directly used here,
# yt-dlp is controlled via 'quiet' option.

class YoutubeDownloader:
    """Handles downloading YouTube audio, checking for existing files."""

    def __init__(self, url: str, output_dir: Path):
        """
        Initializes the downloader.

        Args:
            url: The YouTube video URL.
            output_dir: The directory to save the downloaded audio file.
        """
        self.url = url
        self.output_dir = Path(output_dir)
        self.video_id = None
        self.video_title = None
        self.video_info = None
        self.expected_path = None
        # Configure yt-dlp logging through its options if needed, or let it use root logger
        self.base_ydl_opts = {
            'quiet': True, # Suppress yt-dlp's own console output
            'no_warnings': True,
            # 'verbose': False, # Set to True for yt-dlp debug info if needed
            # 'logger': logging.getLogger('yt_dlp') # Optionally direct yt-dlp logs
        }
        # Ensure output directory exists
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {self.output_dir}: {e}")
            raise # Re-raise the error as it's critical

    def _get_info(self) -> bool:
        """Fetches video info and sets instance attributes."""
        try:
            # yt-dlp can be noisy, use its quiet option primarily
            with yt_dlp.YoutubeDL(self.base_ydl_opts) as ydl:
                logger.info(f"Fetching video info for {self.url}...")
                self.video_info = ydl.extract_info(self.url, download=False)
                if not self.video_info:
                     logger.error(f"Failed to extract video info (returned None) from {self.url}.")
                     return False
                self.video_id = self.video_info.get('id')
                if not self.video_id:
                     logger.warning(f"Could not extract video ID from info for {self.url}.")
                     # Decide if this is fatal or not. Let's allow proceeding without ID for now.
                self.video_title = self.video_info.get('title', f'unknown_title_{self.video_id or "no_id"}')
                # Sanitize title for use in filename
                safe_title = "".join([c for c in self.video_title if c.isalnum() or c in (' ', '-')]).rstrip()
                safe_title = safe_title.replace(' ', '_') # Replace spaces
                # Construct expected filename (using ID if available, else title)
                filename_base = self.video_id if self.video_id else safe_title
                self.expected_path = self.output_dir / f"{filename_base}.wav"
                logger.info(f"Video Info: ID='{self.video_id}', Title='{self.video_title}'")
                logger.info(f"Expected output path: {self.expected_path}")
                return True
        except yt_dlp.utils.DownloadError as e:
            logger.error(f"yt-dlp error fetching info for {self.url}: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error fetching video info for {self.url}")
            return False

    def _check_exists(self) -> bool:
        """Checks if the expected output file already exists."""
        if not self.expected_path:
            logger.warning("Cannot check existence: expected path not set (info fetch failed?).")
            return False
        if self.expected_path.exists():
            logger.info(f"Output file already exists: {self.expected_path}")
            return True
        return False

    def _perform_download(self) -> bool:
        """Performs the actual download using yt-dlp."""
        if not self.expected_path:
            logger.error("Cannot download: expected path not set.")
            return False

        # Define download options, including format and postprocessor for WAV conversion
        ydl_opts = {
            **self.base_ydl_opts, # Include base options (quiet, etc.)
            'format': 'bestaudio/best',
            'outtmpl': str(self.output_dir / '%(id)s.%(ext)s'), # Template before conversion
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                # 'preferredquality': '192', # Optional quality setting
            }],
            # Ensure the final output path matches self.expected_path
            # yt-dlp uses the 'outtmpl' before postprocessing. We rely on the ID being in the name.
            # We will verify the final expected path after download.
        }

        logger.info(f"Starting audio download for {self.url}...")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Download the audio
                ydl.download([self.url])
                # Note: yt-dlp handles the renaming to .wav via the postprocessor.
                # We just need to verify the final expected file exists.
                logger.info(f"yt-dlp download process completed for {self.url}.")
                # Add a small delay to ensure filesystem sync, especially on network drives
                time.sleep(1)
                return True # Assume success, verification happens next
        except yt_dlp.utils.DownloadError as e:
            logger.error(f"yt-dlp download failed for {self.url}: {e}")
            return False
        except Exception as e:
            logger.exception(f"An unexpected error occurred during download for {self.url}")
            return False

    def _verify_output(self) -> bool:
        """Verifies that the expected output file exists after download."""
        if not self.expected_path:
            logger.error("Cannot verify output: expected path not set.")
            return False

        if self.expected_path.exists() and self.expected_path.stat().st_size > 0:
            logger.info(f"Successfully downloaded and converted to '{self.expected_path}'.")
            return True
        else:
            # Check if maybe the filename used title instead of ID if ID was missing
            if not self.video_id and self.video_title:
                 safe_title = "".join([c for c in self.video_title if c.isalnum() or c in (' ', '-')]).rstrip()
                 safe_title = safe_title.replace(' ', '_')
                 alternate_path = self.output_dir / f"{safe_title}.wav"
                 if alternate_path.exists() and alternate_path.stat().st_size > 0:
                      logger.warning(f"Output file found at alternate path based on title: {alternate_path}. Renaming to expected: {self.expected_path}")
                      try:
                           alternate_path.rename(self.expected_path)
                           return True
                      except OSError as rename_e:
                           logger.error(f"Failed to rename alternate path {alternate_path} to {self.expected_path}: {rename_e}")
                           # Fall through to error
                 else:
                      logger.debug(f"Alternate path {alternate_path} based on title also not found or empty.")


            logger.error(f"Download process finished, but expected output file '{self.expected_path}' was not found or is empty for {self.url}.")
            # Log directory contents for debugging
            try:
                dir_contents = list(self.output_dir.glob('*'))
                logger.debug(f"Contents of output directory '{self.output_dir}': {dir_contents}")
            except Exception as list_e:
                logger.warning(f"Could not list contents of output directory '{self.output_dir}': {list_e}")
            return False

    def get_video_info(self) -> tuple[str | None, str | None]:
        """Fetches and returns video info (ID, Title) without downloading."""
        if not self.video_info:
            if not self._get_info():
                return None, None # Return None if info fetch fails
        # Ensure video_id and video_title were set
        return getattr(self, 'video_id', None), getattr(self, 'video_title', None)


    def download_audio(self) -> Path | None:
        """
        Main method to orchestrate the download process.

        Returns:
            The path to the downloaded .wav file, or None if failed.
        """
        logger.info(f"Starting download process for URL: {self.url}")
        if not self.video_info:
             if not self._get_info():
                  # Error already logged in _get_info
                  return None

        if self._check_exists():
            return self.expected_path # Return existing path

        if not self._perform_download():
            # Error already logged in _perform_download
            return None

        if not self._verify_output():
             # Error already logged in _verify_output
             # Optional: Attempt cleanup of partial files? yt-dlp usually handles this.
             return None

        logger.info(f"Download successful: {self.expected_path}")
        return self.expected_path 