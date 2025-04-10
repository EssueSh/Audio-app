import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
from transformers import pipeline

# Load the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased")

st.title("🎙️ Audio Assistant")
st.write("Ask your question through your voice!")

# Speech-to-text
def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        st.success("Audio received!")
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        st.error("Sorry, could not understand your audio.")
        return None
    except sr.RequestError:
        st.error("Could not request results. Check your internet connection.")
        return None

# Text-to-speech
def speak(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
        tts.save(fp.name)
        os.system(f"mpg123 {fp.name}" if os.name != 'nt' else f"start {fp.name}")

# Run voice assistant
if st.button("🎤 Start Listening"):
    question = recognize_speech()
    if question:
        st.write(f"**You asked:** {question}")
        context = "This is a general-purpose assistant. It answers questions based on its knowledge of language and facts."  # Use real context for better answers
        result = qa_pipeline(question=question, context=context)
        answer = result["answer"]
        st.success(f"🤖 Answer: {answer}")
        speak(answer)
