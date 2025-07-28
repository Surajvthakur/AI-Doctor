# voice_of_the_doctor.py - Free TTS Services Only
from dotenv import load_dotenv
load_dotenv()

import os
import logging
import subprocess
import platform
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def text_to_speech_with_gtts(input_text, output_filepath="output_audio.mp3", lang='en', slow=False, auto_play=False):
    """
    Enhanced gTTS with better error handling and options
    """
    try:
        from gtts import gTTS
        
        logging.info("Generating speech with Google TTS (gTTS)...")
        
        # Create TTS object
        tts = gTTS(text=input_text, lang=lang, slow=slow)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_filepath) if os.path.dirname(output_filepath) else '.', exist_ok=True)
        
        # Save audio file
        tts.save(output_filepath)
        logging.info(f"Audio saved to {output_filepath}")
        
        # Auto-play if requested
        if auto_play:
            play_audio(output_filepath)
        
        return output_filepath
        
    except Exception as e:
        logging.error(f"gTTS failed: {e}")
        raise e

def text_to_speech_with_pyttsx3(input_text, output_filepath="output_audio.wav", rate=200, volume=0.9, voice_gender='female', auto_play=False):
    """
    Enhanced pyttsx3 TTS - completely offline
    """
    try:
        import pyttsx3
        
        logging.info("Generating speech with pyttsx3 (offline)...")
        
        # Initialize the TTS engine
        engine = pyttsx3.init()
        
        # Set rate and volume
        engine.setProperty('rate', rate)
        engine.setProperty('volume', volume)
        
        # Set voice
        voices = engine.getProperty('voices')
        if voices:
            selected_voice = None
            
            # Try to find preferred voice type
            for voice in voices:
                voice_name = voice.name.lower()
                if voice_gender == 'female':
                    if any(keyword in voice_name for keyword in ['female', 'zira', 'hazel', 'aria', 'susan']):
                        selected_voice = voice.id
                        break
                else:  # male
                    if any(keyword in voice_name for keyword in ['male', 'david', 'mark', 'james']):
                        selected_voice = voice.id
                        break
            
            # Fallback to first available voice
            if not selected_voice and voices:
                selected_voice = voices[0].id
            
            if selected_voice:
                engine.setProperty('voice', selected_voice)
                logging.info(f"Using voice: {[v.name for v in voices if v.id == selected_voice][0]}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_filepath) if os.path.dirname(output_filepath) else '.', exist_ok=True)
        
        # Save to file
        engine.save_to_file(input_text, output_filepath)
        engine.runAndWait()
        
        logging.info(f"Audio saved to {output_filepath}")
        
        # Auto-play if requested
        if auto_play:
            play_audio(output_filepath)
        
        return output_filepath
        
    except Exception as e:
        logging.error(f"pyttsx3 failed: {e}")
        raise e

def text_to_speech_with_espeak(input_text, output_filepath="output_audio.wav", voice='en', speed=175, auto_play=False):
    """
    eSpeak TTS - Free, lightweight, cross-platform
    Requires espeak to be installed on system
    """
    try:
        logging.info("Generating speech with eSpeak...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_filepath) if os.path.dirname(output_filepath) else '.', exist_ok=True)
        
        # Build espeak command
        cmd = [
            'espeak',
            '-v', voice,  # Voice (en, en+f3 for female, en+m3 for male)
            '-s', str(speed),  # Speed in words per minute
            '-w', output_filepath,  # Write to file
            input_text
        ]
        
        # Run espeak
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if result.returncode == 0:
            logging.info(f"Audio saved to {output_filepath}")
            
            # Auto-play if requested
            if auto_play:
                play_audio(output_filepath)
            
            return output_filepath
        else:
            raise Exception(f"eSpeak failed with return code {result.returncode}")
            
    except FileNotFoundError:
        raise Exception("eSpeak not found. Please install espeak: sudo apt-get install espeak (Linux) or brew install espeak (macOS)")
    except Exception as e:
        logging.error(f"eSpeak failed: {e}")
        raise e

def text_to_speech_with_festival(input_text, output_filepath="output_audio.wav", voice="voice_kal_diphone", auto_play=False):
    """
    Festival TTS - Free, high-quality synthesis
    Requires festival to be installed on system
    """
    try:
        logging.info("Generating speech with Festival...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_filepath) if os.path.dirname(output_filepath) else '.', exist_ok=True)
        
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(input_text)
            temp_text_path = temp_file.name
        
        try:
            # Build festival command
            cmd = [
                'festival',
                '--batch',
                f'(voice_{voice})',
                f'(tts_file "{temp_text_path}" nil)',
                f'(utt.save.wave (tts_file "{temp_text_path}" nil) "{output_filepath}")'
            ]
            
            # Run festival
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.returncode == 0 and os.path.exists(output_filepath):
                logging.info(f"Audio saved to {output_filepath}")
                
                # Auto-play if requested
                if auto_play:
                    play_audio(output_filepath)
                
                return output_filepath
            else:
                raise Exception(f"Festival failed with return code {result.returncode}")
                
        finally:
            # Clean up temp file
            if os.path.exists(temp_text_path):
                os.unlink(temp_text_path)
            
    except FileNotFoundError:
        raise Exception("Festival not found. Please install festival: sudo apt-get install festival (Linux) or brew install festival (macOS)")
    except Exception as e:
        logging.error(f"Festival failed: {e}")
        raise e

def play_audio(audio_filepath):
    """
    Cross-platform audio player
    """
    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(['afplay', audio_filepath], check=True)
        elif os_name == "Windows":  # Windows
            subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{audio_filepath}").PlaySync();'], check=True)
        elif os_name == "Linux":  # Linux
            # Try multiple players in order of preference
            players = ['paplay', 'aplay', 'mpg123', 'ffplay', 'cvlc']
            for player in players:
                try:
                    subprocess.run([player, audio_filepath], check=True, capture_output=True)
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            else:
                raise Exception("No audio player found")
        else:
            logging.warning("Unsupported OS for auto-play")
    except Exception as e:
        logging.error(f"Error playing audio: {e}")

def text_to_speech_smart_free(input_text, output_filepath="output_audio.mp3", preferred_service="auto", auto_play=False):
    """
    Smart TTS function using only free services
    """
    # Convert mp3 to wav for offline services if needed
    base_path = os.path.splitext(output_filepath)[0]
    
    services = []
    
    if preferred_service == "gtts":
        services = ["gtts", "pyttsx3", "espeak", "festival"]
    elif preferred_service == "pyttsx3":
        services = ["pyttsx3", "gtts", "espeak", "festival"]
    elif preferred_service == "espeak":
        services = ["espeak", "pyttsx3", "gtts", "festival"]
    elif preferred_service == "festival":
        services = ["festival", "espeak", "pyttsx3", "gtts"]
    else:  # auto
        services = ["gtts", "pyttsx3", "espeak", "festival"]
    
    last_error = None
    
    for service in services:
        try:
            logging.info(f"Trying TTS service: {service}")
            
            if service == "gtts":
                return text_to_speech_with_gtts(input_text, output_filepath, auto_play=auto_play)
                
            elif service == "pyttsx3":
                # pyttsx3 works better with .wav
                if output_filepath.endswith('.mp3'):
                    wav_path = base_path + '.wav'
                    result = text_to_speech_with_pyttsx3(input_text, wav_path, auto_play=auto_play)
                    
                    # Try to convert to mp3
                    try:
                        from pydub import AudioSegment
                        audio = AudioSegment.from_wav(wav_path)
                        audio.export(output_filepath, format="mp3")
                        os.remove(wav_path)  # Clean up wav file
                        return output_filepath
                    except ImportError:
                        logging.warning("pydub not available, returning wav file")
                        return wav_path
                    except Exception as e:
                        logging.warning(f"MP3 conversion failed: {e}, returning wav file")
                        return wav_path
                else:
                    return text_to_speech_with_pyttsx3(input_text, output_filepath, auto_play=auto_play)
                    
            elif service == "espeak":
                wav_path = base_path + '.wav'
                result = text_to_speech_with_espeak(input_text, wav_path, auto_play=auto_play)
                
                if output_filepath.endswith('.mp3'):
                    # Try to convert to mp3
                    try:
                        from pydub import AudioSegment
                        audio = AudioSegment.from_wav(wav_path)
                        audio.export(output_filepath, format="mp3")
                        os.remove(wav_path)
                        return output_filepath
                    except:
                        return wav_path
                else:
                    return result
                    
            elif service == "festival":
                wav_path = base_path + '.wav'
                result = text_to_speech_with_festival(input_text, wav_path, auto_play=auto_play)
                
                if output_filepath.endswith('.mp3'):
                    # Try to convert to mp3
                    try:
                        from pydub import AudioSegment
                        audio = AudioSegment.from_wav(wav_path)
                        audio.export(output_filepath, format="mp3")
                        os.remove(wav_path)
                        return output_filepath
                    except:
                        return wav_path
                else:
                    return result
                    
        except Exception as e:
            last_error = e
            logging.warning(f"TTS service {service} failed: {e}")
            continue
    
    # If all services failed
    logging.error("All free TTS services failed!")
    if last_error:
        raise last_error
    else:
        raise Exception("No TTS services available")

def get_available_free_tts_services():
    """
    Check which free TTS services are available
    """
    available = []
    
    # Check gTTS
    try:
        from gtts import gTTS
        available.append("gtts")
    except ImportError:
        pass
    
    # Check pyttsx3
    try:
        import pyttsx3
        available.append("pyttsx3")
    except ImportError:
        pass
    
    # Check eSpeak
    try:
        subprocess.run(['espeak', '--version'], capture_output=True, check=True)
        available.append("espeak")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Check Festival
    try:
        subprocess.run(['festival', '--version'], capture_output=True, check=True)
        available.append("festival")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return available

# Backward compatibility functions
def text_to_speech_with_elevenlabs(input_text, output_filepath="output_audio.mp3"):
    """
    Backward compatibility - redirects to free TTS
    """
    logging.warning("ElevenLabs function called but redirecting to free TTS services")
    return text_to_speech_smart_free(input_text, output_filepath)

def text_to_speech(input_text, output_filepath="output_audio.mp3"):
    """
    Main TTS function - uses free services
    """
    return text_to_speech_smart_free(input_text, output_filepath)

if __name__ == "__main__":
    # Test the free TTS services
    test_text = "Hello! This is a test of the free text to speech system. How does this sound to you?"
    
    print("Available free TTS services:", get_available_free_tts_services())
    
    try:
        result = text_to_speech_smart_free(test_text, "test_free_audio.mp3")
        print(f"Free TTS successful! Audio saved to: {result}")
        
        # Test auto-play
        print("Testing auto-play...")
        text_to_speech_smart_free("Auto play test!", "autoplay_test.mp3", auto_play=True)
        
    except Exception as e:
        print(f"Free TTS failed: {e}")
        print("\nTo install required packages:")
        print("pip install gtts pyttsx3 pydub")
        print("\nFor Linux users, also install:")
        print("sudo apt-get install espeak festival")