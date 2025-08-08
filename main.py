import os
import streamlit as st
from dotenv import load_dotenv
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_smart_free, get_available_free_tts_services
import tempfile
import time
import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
from datetime import datetime
import threading

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="AI Doctor - Conversational", layout="wide")
st.title("🩺 AI Doctor - Conversational Health Assistant")
st.markdown("Have a **continuous conversation** with your AI doctor. Upload images, record voice messages, and get ongoing medical guidance.")

# Initialize session state for conversation
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'conversation_count' not in st.session_state:
    st.session_state.conversation_count = 0
# Initialize quick action state
if 'quick_action_text' not in st.session_state:
    st.session_state.quick_action_text = ""
if 'clear_input' not in st.session_state:
    st.session_state.clear_input = False
# Initialize recording state
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'recording_start_time' not in st.session_state:
    st.session_state.recording_start_time = None
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""
if 'input_type' not in st.session_state:
    st.session_state.input_type = ""

def record_audio_flexible(file_path):
    """Flexible audio recording with user control"""
    recognizer = sr.Recognizer()
    
    # Configure recognizer for better performance
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.0
    
    try:
        with sr.Microphone() as source:
            st.info("🎙️ Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            
            st.success("🔴 Recording... Speak clearly now!")
            
            # Record audio with proper settings
            audio_data = recognizer.listen(
                source, 
                timeout=5,  # Wait up to 5 seconds for speech to start
                phrase_time_limit=30  # Maximum 30 seconds per phrase
            )
            
            # Get audio data
            wav_data = audio_data.get_wav_data()
            
            if len(wav_data) < 2000:  # Increased minimum size
                st.error("❌ Recording too short or no audio detected. Please try again.")
                return False
            
            # Convert to AudioSegment for analysis
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            duration_ms = len(audio_segment)
            loudness = audio_segment.dBFS
            
            if loudness < -35:  # Adjusted threshold
                st.warning("⚠️ Audio seems quiet. Try speaking louder next time.")
            
            # Export as WAV first (better compatibility with Groq)
            wav_path = file_path.replace('.mp3', '.wav')
            audio_segment.export(wav_path, format="wav")
            
            # Also export as MP3 for compatibility
            audio_segment.export(file_path, format="mp3", bitrate="192k")
            return True
            
    except sr.WaitTimeoutError:
        st.error("❌ No speech detected. Please try again and speak clearly.")
        return False
    except Exception as e:
        st.error(f"❌ Recording error: {e}")
        return False

def transcribe_with_groq_enhanced(stt_model, audio_filepath, GROQ_API_KEY):
    """Enhanced transcription with better error handling"""
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        
        if not os.path.exists(audio_filepath):
            return None, "Audio file not found"
            
        file_size = os.path.getsize(audio_filepath)
        if file_size < 2000:  # Increased minimum size
            return None, "Audio file too small"
        
        # Try WAV format first (better compatibility)
        wav_path = audio_filepath.replace('.mp3', '.wav')
        if os.path.exists(wav_path):
            audio_filepath = wav_path
        
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en",
                response_format="verbose_json",
                temperature=0.0,
            )
        
        text = transcription.text.strip()
        
        # Better validation of transcription
        if not text or text.lower() in ["thank you", "thanks", "thank you.", ""]:
            return None, "⚠️ Transcription failed or returned empty result. Please try again."
        
        return text, "✅ Transcription successful"
        
    except Exception as e:
        return None, f"Transcription error: {str(e)}"

def get_text_only_response(prompt):
    """Simple text-only response function - replace with your actual text model"""
    # This is a placeholder - replace with your actual text-only model call
    return f"Based on what you've described: '{prompt}', I understand your concerns. However, without being able to examine you directly or see any images, I can provide general guidance. If these symptoms persist or worsen, please consult with a healthcare provider in person for a proper evaluation."

# Create layout with sidebar and main content
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Clear conversation button
    if st.button("🗑️ Clear All Conversation", type="secondary"):
        st.session_state.conversation_history = []
        st.session_state.uploaded_images = []
        st.session_state.current_image = None
        st.session_state.conversation_count = 0
        st.session_state.quick_action_text = ""
        st.session_state.clear_input = True
        st.success("Conversation cleared!")
        st.rerun()
    
    # Show conversation stats
    st.metric("💬Total Messages", len(st.session_state.conversation_history))
    # st.metric("📷 Images", len(st.session_state.uploaded_images))
    
    # Available TTS services
    available_tts = get_available_free_tts_services()
    if available_tts:
        st.success(f"🔊 TTS: {', '.join(available_tts)}")
    else:
        st.warning("🔇 No TTS available")
        with st.expander("Install TTS"):
            st.code("pip install gtts pyttsx3")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Conversation")
    
    # Display conversation history
    if st.session_state.conversation_history:
        for i, entry in enumerate(st.session_state.conversation_history):
            with st.container():
                st.markdown(f"**#{i+1} - {entry['timestamp']}**")
                
                # Show user input
                if entry['type'] == 'voice':
                    st.markdown(f"🎙️ **You said:** {entry['user_input']}")
                elif entry['type'] == 'text':
                    st.markdown(f"💬 **You wrote:** {entry['user_input']}")
                
                # Show image only if it's a new image (not the same as previous)
                if entry.get('image_path') and entry.get('is_new_image', False):
                    st.image(entry['image_path'], caption="Medical Image", width=200)
                
                # Show doctor response
                st.markdown(f"🩺 **AI Doctor:** {entry['doctor_response']}")
                
                # Audio response if available
                if entry.get('audio_path') and os.path.exists(entry['audio_path']):
                    st.audio(entry['audio_path'], format="audio/mp3")
                
                st.markdown("---")
    else:
        st.info("👋 Start your conversation with the AI Doctor below!")

with col2:
    st.header("📝 Add to Conversation")
    
    # Image upload section
    st.subheader("📷 Upload Image (Optional)")
    uploaded_image = st.file_uploader(
        "Medical image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a new image or keep using the previous one"
    )
    
    if uploaded_image:
        st.image(uploaded_image, caption="New Image", width=200)
        st.session_state.current_image = uploaded_image
    elif st.session_state.current_image:
        st.info("📷 Using previous image")
    
    # Input method selection
    st.subheader("💬 Choose Input Method")
    input_method = st.radio(
        "How would you like to communicate?",
        ["🎙️ Voice Recording", "💬 Text Input"],
        horizontal=True
    )
    
    # Get transcribed text from session state if available
    if st.session_state.transcribed_text:
        user_input = st.session_state.transcribed_text
        input_type = st.session_state.input_type
    else:
        user_input = ""
        input_type = ""
    
    if input_method == "🎙️ Voice Recording":
        st.subheader("🎙️ Record Voice Message")
        
        audio_file_path = f"conversation_audio_{st.session_state.conversation_count}.mp3"
        
        # Simple recording button
        if st.button("🎙️ Record Voice Message", type="primary"):
            with st.spinner("🎙️ Recording in progress..."):
                success = record_audio_flexible(audio_file_path)
                if success:
                    # Transcribe immediately
                    with st.spinner("🎯 Converting speech to text..."):
                        groq_api_key = os.environ.get("GROQ_API_KEY")
                        if not groq_api_key:
                            st.error("❌ GROQ_API_KEY not found in environment variables")
                        else:
                            transcription, trans_status = transcribe_with_groq_enhanced(
                                stt_model="whisper-large-v3",
                                audio_filepath=audio_file_path,
                                GROQ_API_KEY=groq_api_key
                            )
                            
                            if transcription:
                                st.session_state.transcribed_text = transcription
                                st.session_state.input_type = "voice"
                                st.success(f"✅ You said: '{transcription}'")
                            else:
                                st.error(f"❌ Speech-to-text failed: {trans_status}")
                else:
                    st.error("❌ Recording failed. Please try again.")
    
    else:  # Text Input
        st.subheader("💬 Type Your Message")
        
        # Handle quick action text and clear input logic
        default_value = ""
        if st.session_state.quick_action_text:
            default_value = st.session_state.quick_action_text
            st.session_state.quick_action_text = ""  # Clear after using
        elif st.session_state.clear_input:
            default_value = ""
            st.session_state.clear_input = False
        
        # Create unique key for each text area to avoid conflicts
        text_key = f"text_input_{st.session_state.conversation_count}"
        
        user_input = st.text_area(
            "Describe your symptoms or ask a question:",
            value=default_value,
            placeholder="e.g., 'I also have a headache and feel dizzy' or 'What should I do next?' or 'Is this serious?'",
            height=100,
            key=text_key
        )
        input_type = "text"
        
        if user_input.strip():
            st.success(f"✅ Message ready: '{user_input.strip()}'")
    
    # Send message button
    if st.button("🚀 Send to AI Doctor", type="primary", disabled=not user_input.strip()):
        if user_input.strip():
            # Store the input for processing
            current_user_input = user_input.strip()
            current_input_type = input_type
            with st.spinner("🤖 AI Doctor is analyzing..."):
                
                # Build conversation context
                conversation_context = ""
                if st.session_state.conversation_history:
                    conversation_context = "\n\nPrevious conversation context:\n"
                    for entry in st.session_state.conversation_history[-3:]:  # Last 3 exchanges
                        conversation_context += f"Patient: {entry['user_input']}\n"
                        conversation_context += f"Doctor: {entry['doctor_response']}\n"
                
                # Build system prompt based on whether we have an image
                if st.session_state.current_image or uploaded_image:
                    system_prompt = f"""
                    You are a professional doctor having a continuous conversation with a patient. 
                    Look at the medical image and listen to what the patient is saying. 
                    Provide ongoing medical guidance as the conversation develops.
                    
                    Guidelines:
                    - This is a follow-up conversation, so reference previous information when relevant
                    - If patient asks follow-up questions, answer them directly
                    - If patient provides new symptoms, incorporate them into your assessment
                    - Keep responses concise but thorough (2-3 sentences max)
                    - Always sound like a real doctor, not an AI
                    - Start directly with your response, no preambles
                    - If you see concerning symptoms, advise seeing a real doctor
                    
                    Current patient message: {current_user_input}{conversation_context}
                    """
                else:
                    system_prompt = f"""
                    You are a professional doctor having a text-based conversation with a patient.
                    The patient doesn't have an image to share, so focus on their verbal/text description.
                    
                    Guidelines:
                    - This is ongoing conversation, reference previous context when relevant
                    - Answer follow-up questions directly
                    - Incorporate new symptoms into your assessment
                    - Keep responses concise (2-3 sentences max)
                    - Sound like a real doctor, not an AI
                    - If symptoms sound concerning, recommend seeing a real doctor
                    
                    Current patient message: {current_user_input}{conversation_context}
                    """
                
                try:
                    # Get AI response
                    image_path = None
                    is_new_image = False
                    
                    if st.session_state.current_image or uploaded_image:
                        # Save current image
                        current_img = uploaded_image or st.session_state.current_image
                        # Reset file pointer if it's a file-like object
                        if hasattr(current_img, 'seek'):
                            current_img.seek(0)
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
                            temp_img.write(current_img.read())
                            image_path = temp_img.name
                            
                        # Check if this is a new image
                        if uploaded_image:
                            is_new_image = True
                            
                        encoded_img = encode_image(image_path)
                        doctor_response = analyze_image_with_query(
                            query=system_prompt,
                            encoded_image=encoded_img,
                            model="meta-llama/llama-4-scout-17b-16e-instruct"
                        )
                    else:
                        # Text-only response
                        doctor_response = get_text_only_response(current_user_input)
                    
                    # Generate audio response
                    audio_path = None
                    if available_tts:
                        try:
                            audio_filename = f"doctor_response_{st.session_state.conversation_count}.mp3"
                            audio_path = text_to_speech_smart_free(
                                input_text=doctor_response,
                                output_filepath=audio_filename,
                                preferred_service="auto"
                            )
                        except Exception as e:
                            st.warning(f"⚠️ Could not generate audio: {e}")
                    
                    # Add to conversation history
                    conversation_entry = {
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'type': current_input_type,
                        'user_input': current_user_input,
                        'doctor_response': doctor_response,
                        'image_path': image_path if (st.session_state.current_image or uploaded_image) else None,
                        'is_new_image': is_new_image,
                        'audio_path': audio_path
                    }
                    
                    st.session_state.conversation_history.append(conversation_entry)
                    st.session_state.conversation_count += 1
                    
                    # Clear the input and transcribed text by triggering a rerun
                    st.session_state.clear_input = True
                    st.session_state.transcribed_text = ""
                    st.session_state.input_type = ""
                    
                    st.success("✅ Response added to conversation!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error getting AI response: {e}")

# Quick action buttons
st.markdown("---")
st.subheader("⚡ Quick Actions")

quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)

with quick_col1:
    if st.button("❓ Ask Follow-up"):
        st.session_state.quick_action_text = "Can you explain more about this condition?"
        st.rerun()

with quick_col2:
    if st.button("💊 Treatment Options"):
        st.session_state.quick_action_text = "What are my treatment options?"
        st.rerun()

with quick_col3:
    if st.button("⚠️ When to Worry"):
        st.session_state.quick_action_text = "When should I see a doctor urgently?"
        st.rerun()

with quick_col4:
    if st.button("🏠 Home Remedies"):
        st.session_state.quick_action_text = "What can I do at home to help with this?"
        st.rerun()

# Footer
st.markdown("---")
st.markdown("⚠️ **Disclaimer**: This is an AI assistant for educational purposes only. Always consult a real healthcare professional for medical advice.")

# Debug info
if st.checkbox("🔧 Debug Info"):
    st.json({
        "conversation_entries": len(st.session_state.conversation_history),
        "has_current_image": st.session_state.current_image is not None,
        "conversation_count": st.session_state.conversation_count,
        "groq_api_key": bool(os.environ.get("GROQ_API_KEY")),
        "available_tts": available_tts,
        "quick_action_text": st.session_state.quick_action_text,
        "clear_input": st.session_state.clear_input,
        "is_recording": st.session_state.is_recording
    })
    
    # Test microphone button
    if st.button("🎤 Test Microphone"):
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            
            with sr.Microphone() as source:
                st.info("🎙️ Testing microphone... Please speak something.")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                st.info(f"🎙️ Energy threshold: {recognizer.energy_threshold}")
                
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
                wav_data = audio.get_wav_data()
                
                st.success(f"✅ Microphone test successful! Audio size: {len(wav_data)} bytes")
                
                # Try to transcribe the test audio
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(wav_data)
                    temp_path = temp_file.name
                
                groq_api_key = os.environ.get("GROQ_API_KEY")
                if groq_api_key:
                    test_transcription, test_status = transcribe_with_groq_enhanced(
                        stt_model="whisper-large-v3",
                        audio_filepath=temp_path,
                        GROQ_API_KEY=groq_api_key
                    )
                    if test_transcription:
                        st.success(f"✅ Test transcription: '{test_transcription}'")
                    else:
                        st.error(f"❌ Test transcription failed: {test_status}")
                else:
                    st.warning("⚠️ No GROQ API key for test transcription")
                    
                # Clean up
                os.unlink(temp_path)
                
        except Exception as e:
            st.error(f"❌ Microphone test failed: {e}")