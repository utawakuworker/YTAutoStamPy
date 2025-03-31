import logging
from pathlib import Path
import pandas as pd
import subprocess
import json
from google import genai
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def identify_songs(youtube_url: str, singing_segments: pd.DataFrame, config: Dict[str, Any], 
                  audio_file_path: str) -> Dict[str, Dict[str, str]]:
    """
    Identify songs from detected singing segments.
    
    Args:
        youtube_url: The YouTube video URL.
        singing_segments: DataFrame with start_time and end_time of singing segments.
        config: Configuration dictionary.
        audio_file_path: Path to the already downloaded audio file.
        
    Returns:
        Dictionary mapping segment indices to song information (title and artist).
    """
    # Extract API keys and configuration
    whisper_model = config.get('song_identifier', {}).get('whisper_model', 'base')
    gemini_api_key = config.get('song_identifier', {}).get('gemini_api_key')
    gemini_model = config.get('song_identifier', {}).get('gemini_model', 'gemini-2.0-flash')
    
    if not gemini_api_key:
        logger.error("Gemini API key not found in config.")
        return {}
    
    # Verify the audio file exists
    if not os.path.exists(audio_file_path):
        logger.error(f"Audio file not found at: {audio_file_path}")
        return {}
        
    # Configure Gemini API
    client = genai.Client(api_key=gemini_api_key)
    
    results = {}
    
    # Create a temporary directory for audio segments
    temp_dir = Path("temp_song_segments")
    temp_dir.mkdir(exist_ok=True)
    
    # Process each singing segment
    for idx, segment in singing_segments.iterrows():
        start_time = segment['start_time']
        end_time = segment['end_time']
        
        logger.info(f"Processing segment {idx} ({start_time:.2f}s - {end_time:.2f}s)")
        
        # Extract audio segment using ffmpeg
        temp_audio_path = temp_dir / f"segment_{idx}.wav"
        try:
            extract_audio_segment(audio_file_path, start_time, end_time, str(temp_audio_path))
            
            # Transcribe with Whisper
            lyrics = transcribe_with_whisper(str(temp_audio_path), whisper_model)
            if not lyrics or lyrics.strip() == "":
                logger.warning(f"No lyrics transcribed for segment {idx}")
                continue
                
            # Identify song with Gemini
            song_info = identify_song_with_gemini(lyrics, client, gemini_model)
            if song_info:
                results[idx] = song_info
                
        except Exception as e:
            logger.error(f"Error processing segment {idx}: {e}", exc_info=True)
        finally:
            # Clean up temporary files
            if temp_audio_path.exists():
                temp_audio_path.unlink()
    
    # Clean up temp directory if empty
    try:
        temp_dir.rmdir()
    except:
        pass
        
    return results

def extract_audio_segment(audio_file_path: str, start_time: float, end_time: float, output_path: str) -> None:
    """Extract audio segment from an existing audio file using ffmpeg."""
    duration = end_time - start_time
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_time), "-t", str(duration),
        "-i", audio_file_path, "-ac", "1", "-ar", "16000", output_path
    ]
    
    logger.debug(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True)
    
def transcribe_with_whisper(audio_path: str, model: str = "base") -> str:
    """Transcribe audio using OpenAI's Whisper model."""
    import whisper
    
    logger.info(f"Transcribing with Whisper model: {model}")
    whisper_model = whisper.load_model(model)
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def identify_song_with_gemini(lyrics: str, client, model_name: str) -> Optional[Dict[str, str]]:
    """Identify song title and artist from lyrics using Gemini API."""
    logger.info("Sending lyrics to Gemini API for song identification")
    
    prompt = f"""
    The following are lyrics transcribed from audio using automatic speech recognition. 
    Since these are machine-transcribed lyrics, they may contain errors or inaccuracies.
    Please identify the song title and artist, even if the lyrics aren't perfectly transcribed.
    
    If you're not confident in the identification, please indicate that.
    Only respond with the song title and artist in JSON format: {{"title": "Song Title", "artist": "Artist Name"}}
    If you cannot identify the song, respond with: {{"title": "Unknown", "artist": "Unknown"}}
    
    Transcribed lyrics:
    {lyrics}
    """
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        
        response_text = response.text
        
        # Extract JSON from response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)
            
            # Validate result format
            if "title" in result and "artist" in result:
                if result["title"] == "Unknown" and result["artist"] == "Unknown":
                    logger.info("Gemini could not identify the song")
                    return None
                return result
        
        logger.warning(f"Unexpected response format from Gemini: {response_text}")
        return None
        
    except Exception as e:
        logger.error(f"Error identifying song with Gemini: {e}", exc_info=True)
        return None 