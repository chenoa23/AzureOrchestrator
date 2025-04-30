import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import subprocess
import pycld2 as cld2
from langdetect import detect, DetectorFactory

# Language Detection Seed Consistency
DetectorFactory.seed = 0

# Load your custom model and vectorizer
clf = joblib.load("custom_lang_model.pkl")
vectorizer = joblib.load("custom_vectorizer.pkl")

def ask_gemma(prompt):
    print(f"[DEBUG] Sending to Gemma: {prompt}")
    try:
        result = subprocess.run(
            ["ollama", "run", "gemma3:1b"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=30
        )
        response = result.stdout.decode("utf-8")
        print(f"[DEBUG] Gemma raw output: {response}")
        return response.strip()
    except Exception as e:
        print(f"[Gemma ERROR] {e}")
        return f"[Gemma Error] {str(e)}"

def detect_language(text):
    try:
        x = vectorizer.transform([text])
        lang_code = clf.predict(x)[0]
        if lang_code not in ["en", "fr", "es", "hat"]:
            raise ValueError("Custom model uncertain, using fallback")
        return "Custom Model", lang_code, "100%", text
    except Exception:
        try:
            is_reliable, _, details = cld2.detect(text)
            return "CLD2", details[0][1], f"{details[0][2]}%", text
        except Exception:
            try:
                return "Langdetect", detect(text), "unknown", text
            except Exception as e:
                return f"Error: {str(e)}", "??", 0, text

def detect_language_from_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        status_label.config(text="🎙️ Listening... please speak")
        window.update()
        try:
            audio = recognizer.listen(source, timeout=2, phrase_time_limit=5)
            print("[DEBUG] Finished recording.")
            status_label.config(text="🧠 Transcribing...")
        except Exception as e:
            messagebox.showerror("Microphone Error", f"Error capturing audio: {e}")
            status_label.config(text="Ready")
            return

        try:
            spoken_text = recognizer.recognize_google(audio)
            print(f"[DEBUG] Recognized: {spoken_text}")
        except sr.UnknownValueError:
            messagebox.showerror("Speech Error", "Could not understand audio.")
            status_label.config(text="Ready")
            return
        except Exception as e:
            messagebox.showerror("Speech Error", str(e))
            status_label.config(text="Ready")
            return

        model_used, lang_code, confidence, text = detect_language(spoken_text)

        # Ask Gemma for a response
        if lang_code == "hat":
            gemma_prompt = f"This sentence is in Haitian Creole: {text}"
        else:
            gemma_prompt = f"Respond to this message: {text}"

        gemma_response = ask_gemma(gemma_prompt)

        result_label.config(
            text=f"Detected: {lang_code} via {model_used}\nInput: {text}\nGemma: {gemma_response}"
        )
        status_label.config(text="Ready")

# GUI Setup
window = tk.Tk()
window.title("Voice Language Detector")
window.geometry("500x300")

title_label = tk.Label(window, text="🎤 Speak to Detect Language", font=("Helvetica", 14))
title_label.pack(pady=10)

status_label = tk.Label(window, text="Ready", fg="blue")
status_label.pack()

detect_button = tk.Button(
    window,
    text="Start Listening",
    command=detect_language_from_speech,
    bg="#007acc",
    fg="white",
    font=("Helvetica", 12)
)
detect_button.pack(pady=20)

result_label = tk.Label(window, text="", wraplength=450, justify="left", font=("Helvetica", 12))
result_label.pack(pady=10)

window.mainloop()
