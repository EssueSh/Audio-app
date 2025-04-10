import streamlit as st
from transformers import pipeline
from gtts import gTTS
from io import BytesIO
import base64
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import tempfile

# Load the QA model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased")

st.set_page_config(page_title="🎙️ Audio Assistant", layout="centered")
st.title("🎙️ Audio Assistant")
st.write("Ask your question using your voice!")

# Record audio using Streamlit mic_recorder
audio_bytes = mic_recorder(start_prompt="🎤 Start Recording", stop_prompt="⏹ Stop Recording", key="recorder")

# Convert audio bytes to text
def transcribe_audio(audio_bytes):
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio.flush()
        with sr.AudioFile(temp_audio.name) as source:
            audio = recognizer.record(source)
            try:
                return recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                st.error("Sorry, could not understand your audio.")
            except sr.RequestError:
                st.error("Could not connect to Google's servers.")
    return None

# Text-to-speech helper
def speak(text):
    tts = gTTS(text)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    st.audio(mp3_fp.getvalue(), format="audio/mp3")

# Process if audio is recorded
if audio_bytes:
    st.success("Audio recorded!")
    question = transcribe_audio(audio_bytes)
    if question:
        st.write(f"**You asked:** {question}")
        context = "This is a general-purpose assistant. It answers questions based on its knowledge of language and facts."
        result = qa_pipeline(question=question, context=context)
        answer = result["answer"]
        st.success(f"🤖 Answer: {answer}")
        speak(answer)
