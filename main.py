import time

from openai import OpenAI
import speech_recognition as sr

import pyaudio
import wave
import numpy as np

from dotenv import load_dotenv
from playsound import playsound

import whisper

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = whisper.load_model('base').to(device)

load_dotenv()

client = OpenAI()

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10  # Tempo máximo de gravação (em segundos)
THRESHOLD = 500  # Limiar para detecção de pausa (ajuste conforme necessário)

r = sr.Recognizer()
source = sr.Microphone()

with source as m:
    r.adjust_for_ambient_noise(m)

def record_audio():
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    frames = []
    silent_frames = 0

    print("Gravando...")

    while True:
        data = stream.read(CHUNK)
        frames.append(data)

        # Converta os dados para um array numpy
        audio_array = np.frombuffer(data, dtype=np.int16)

        # Verifica se o áudio é silencioso (abaixo do limiar)
        # print(np.max(np.abs(audio_array)))
        if np.max(np.abs(audio_array)) < THRESHOLD:
            silent_frames += 1
        else:
            silent_frames = 0

        # print(silent_frames)
        # Se detectar uma pausa, encerra a gravação
        # print(RATE / CHUNK * RECORD_SECONDS)
        if silent_frames >= 50:
            break

    print("Gravação concluída.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Salva o áudio em um arquivo WAV
    with wave.open("gravacao.wav", "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    data = open("gravacao.wav", "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1", file=data, response_format="text", language="pt"
    )

    print(transcript)

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                # "content": """You are a virtual assistant. Answer only like GLaDOS from the portal games would.
                # When receiving a command, say you will do it, but have some resistante/sarcasm in there as well.
                # Don't be afraid to be arrogant/rude, as long as it's in character. """,
                "content": """You are a virtual assistant. """,
            },
            {"role": "user", "content": transcript},
        ],
    )

    response = completion.choices[0].message.content

    print(response)

    audio_response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=response
    )

    audio_response.write_to_file("response.mp3")

    playsound("response.mp3")


def callback(recognizer, audio):
    print("callbacking")
    with open("recording.wav", "wb") as f:
        f.write(audio.get_wav_data())
    result = model.transcribe("recording.wav", fp16=False)
    print(result["text"])
    # with open("audio.wav", "wb") as f:
    #     f.write(audio.get_wav_data())
    # audioFile = open("audio.wav", "rb")
    # transcript = client.audio.transcriptions.create(
    #     model="whisper-1", file=audioFile, response_format="text", language="pt"
    # )
    #
    # print(transcript)
    print("Done")


def startListening():
    print("listening")
    stop_listening = r.listen_in_background(source, callback)
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    startListening()
