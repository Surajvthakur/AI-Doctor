# Enhanced version with debugging and improvements
from dotenv import load_dotenv
load_dotenv()

import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import os
from groq import Groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path, timeout=20, phrase_time_limit=10, energy_threshold=300):
    """
    Enhanced audio recording with better error handling and audio quality checks.
    """
    recognizer = sr.Recognizer()
    
    # Adjust recognizer settings for better performance
    recognizer.energy_threshold = energy_threshold  # Minimum audio energy to consider for recording
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1  # Seconds of non-speaking audio before phrase is complete
    
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise (this may take a moment)...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            logging.info(f"Energy threshold set to: {recognizer.energy_threshold}")
            logging.info("Start speaking clearly now...")
            
            # Record the audio
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")
            
            # Check audio data length
            wav_data = audio_data.get_wav_data()
            logging.info(f"Audio data size: {len(wav_data)} bytes")
            
            if len(wav_data) < 1000:  # Very short audio
                logging.warning("Warning: Audio recording seems very short. This might cause transcription issues.")
            
            # Convert to AudioSegment for analysis
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            
            # Check audio properties
            duration_ms = len(audio_segment)
            logging.info(f"Audio duration: {duration_ms/1000:.2f} seconds")
            logging.info(f"Audio channels: {audio_segment.channels}")
            logging.info(f"Sample rate: {audio_segment.frame_rate} Hz")
            
            # Check if audio is too quiet
            if audio_segment.dBFS < -40:
                logging.warning(f"Warning: Audio seems very quiet (dBFS: {audio_segment.dBFS:.2f}). Consider speaking louder or adjusting microphone.")
            
            # Export with better quality settings
            audio_segment.export(file_path, format="mp3", bitrate="192k")
            logging.info(f"Audio saved to {file_path}")
            
            return True

    except sr.WaitTimeoutError:
        logging.error("No speech detected within the timeout period")
        return False
    except Exception as e:
        logging.error(f"An error occurred during recording: {e}")
        return False

def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY):
    """
    Enhanced transcription with better error handling and debugging.
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        # Check if file exists and get file size
        if not os.path.exists(audio_filepath):
            logging.error(f"Audio file not found: {audio_filepath}")
            return None
            
        file_size = os.path.getsize(audio_filepath)
        logging.info(f"Transcribing file: {audio_filepath} (Size: {file_size} bytes)")
        
        if file_size < 1000:
            logging.warning("Audio file seems very small, transcription might be unreliable")
        
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en",
                response_format="verbose_json",  # Get more detailed response
                temperature=0.0,  # More deterministic output
            )
        
        # Log detailed response
        if hasattr(transcription, 'duration'):
            logging.info(f"Transcription duration: {transcription.duration}s")
        
        text = transcription.text.strip()
        logging.info(f"Transcribed text: '{text}'")
        logging.info(f"Text length: {len(text)} characters")
        
        # Check for suspicious results
        if text.lower() in ["thank you", "thanks", "thank you.", ""]:
            logging.warning("⚠️  Suspicious transcription result detected!")
            logging.warning("This might indicate:")
            logging.warning("- Audio quality issues")
            logging.warning("- Very short or silent recording")
            logging.warning("- Microphone problems")
        
        return text
        
    except Exception as e:
        logging.error(f"Transcription error: {e}")
        return None

def test_microphone():
    """
    Test microphone functionality
    """
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logging.info("Testing microphone...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Microphone test: Say something brief...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            logging.info("Microphone test successful!")
            return True
    except Exception as e:
        logging.error(f"Microphone test failed: {e}")
        return False

# Usage example with debugging
if __name__ == "__main__":
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    stt_model = "whisper-large-v3"
    audio_filepath = "patient_voice_test_debug.mp3"
    
    if not GROQ_API_KEY:
        logging.error("GROQ_API_KEY not found in environment variables")
        exit(1)
    
    # Test microphone first
    if not test_microphone():
        logging.error("Microphone test failed. Please check your microphone setup.")
        exit(1)
    
    # Record audio with enhanced settings
    logging.info("=== Starting Enhanced Recording ===")
    if record_audio_enhanced(audio_filepath, timeout=30, phrase_time_limit=15):
        # Transcribe
        logging.info("=== Starting Transcription ===")
        result = transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY)
        
        if result:
            print(f"\n🎯 FINAL RESULT: '{result}'")
        else:
            print("❌ Transcription failed")
    else:
        logging.error("Recording failed")

# Additional debugging function
def analyze_audio_file(filepath):
    """
    Analyze an existing audio file for debugging
    """
    try:
        audio = AudioSegment.from_mp3(filepath)
        print(f"\n📊 Audio Analysis for {filepath}:")
        print(f"Duration: {len(audio)/1000:.2f} seconds")
        print(f"Channels: {audio.channels}")
        print(f"Sample Rate: {audio.frame_rate} Hz")
        print(f"Loudness: {audio.dBFS:.2f} dBFS")
        print(f"Max possible amplitude: {audio.max}")
        
        if audio.dBFS < -40:
            print("⚠️  Audio is very quiet")
        if len(audio) < 1000:
            print("⚠️  Audio is very short")
            
    except Exception as e:
        print(f"Error analyzing audio: {e}")

# Uncomment to analyze existing file:
# analyze_audio_file("patient_voice_test_for_patient.mp3")