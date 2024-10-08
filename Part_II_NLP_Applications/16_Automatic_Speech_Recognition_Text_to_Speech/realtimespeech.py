from transformers import pipeline
import numpy as np
import scipy.io.wavfile as wav
import os
import wave
import pyaudio

def ses_tanima():
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo")
    
    print("Lütfen konuşun... (Türkçe veya İngilizce)")
    fs = 16000  # Örnekleme frekansı (Hz)
    duration = 5  # Kayıt süresi (saniye)
    
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)
        frames = []
        for _ in range(0, int(fs / 1024 * duration)):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    wav_filename = "gecici_kayit.wav"
    with wave.open(wav_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))

    result = asr(wav_filename)
    print("Tanıma sonucu: " + result['text'])
    
    if os.path.exists(wav_filename):
        os.remove(wav_filename)

if __name__ == "__main__":
    ses_tanima()