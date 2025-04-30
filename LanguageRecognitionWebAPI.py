# Requirements:
#pip install fastapi uvicorn speechrecognition joblib scikit-learn pycld2 langdetect pygame gtts requests
#pip install python-multipart

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pycld2 as cld2
from langdetect import detect, DetectorFactory
import threading
import pygame
import time
from gtts import gTTS
import os
import tempfile
import subprocess
import requests

DetectorFactory.seed = 0
pygame.mixer.init()

app = FastAPI()

# Load pre-trained models
try:
    clf = joblib.load("custom_lang_model.pkl")
    vectorizer = joblib.load("custom_vectorizer.pkl")
except Exception as e:
    clf, vectorizer = None, None
    print(f"Error loading models: {e}")

# Lang detection

def detect_language(text):
    try:
        if clf and vectorizer:
            x = vectorizer.transform([text])
            lang_code = clf.predict(x)[0]
            if lang_code in ["en", "fr", "es", "hat"]:
                return "Custom Model", lang_code, "100%"
        is_reliable, _, details = cld2.detect(text)
        return "CLD2", details[0][1], f"{details[0][2]}%"
    except:
        try:
            lang_code = detect(text)
            return "Langdetect", lang_code, "Unknown"
        except:
            return "Default", "en", "Unknown"

# Gemma 2b integration via Ollama (local LLM service)
def query_gemma(prompt: str):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma:2b", "prompt": prompt, "stream": False},
            timeout=30
        )
        result = response.json()
        return result.get("response", "I don't know how to answer that.")
    except Exception as e:
        return f"Error communicating with Gemma model: {e}"

# Convert text to speech and return path to audio file
def text_to_speech(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        return str(e)

@app.post("/transcribe/")
def transcribe_audio(file: UploadFile):
    recognizer = sr.Recognizer()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(file.file.read())
            tmp.flush()
            with sr.AudioFile(tmp.name) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                detector, lang_code, confidence = detect_language(text)
                response = query_gemma(text)
                audio_path = text_to_speech(response, lang_code)
                return JSONResponse(content={
                    "transcribed_text": text,
                    "detected_language": lang_code,
                    "model_confidence": confidence,
                    "response": response,
                    "tts_audio_file": audio_path
                })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def read_root():
    return {"message": "Multilingual Voice Assistant API is running."}
