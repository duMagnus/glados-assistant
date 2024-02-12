from os import system
import speech_recognition as sr
import sys
import time
import pyttsx3
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

wake_word = 'jarvis'
r = sr.Recognizer()
listening_for_wake_word = True
source = sr.Microphone()

engine = pyttsx3.init()

client = OpenAI()


def speak(text):
    if sys.platform == 'darwin':
        ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!-_$:+-/ ")
        clean_text = ''.join(c for c in text if c in ALLOWED_CHARS)
        system(f"say '{clean_text}'")
    else:
        engine.say(text)
        engine.runAndWait()


def listen_for_wake_word(audio):
    global listening_for_wake_word
    with open("wake_detect.wav", "wb") as f:
        f.write(audio.get_wav_data())
    data = open("wake_detect.wav", "rb")
    result = client.audio.transcriptions.create(
        model="whisper-1", file=data, response_format="text", language="pt"
    )
    text_input = result
    if wake_word in text_input.lower().strip():
        print("Wake word detected. Please speak your prompt to GPT4All.")
        speak('Listening')
        listening_for_wake_word = False


def prompt_gpt(audio):
    global listening_for_wake_word
    try:
        with open("prompt.wav", "wb") as f:
            f.write(audio.get_wav_data())
        data = open("prompt.wav", "rb")
        result = client.audio.transcriptions.create(
            model="whisper-1", file=data, response_format="text", language="pt"
        )
        prompt_text = result
        if len(prompt_text.strip()) == 0:
            print("Empty prompt. Please speak again.")
            speak("Empty prompt. Please speak again.")
            listening_for_wake_word = True
        else:
            print('User: ' + prompt_text)

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
                    {"role": "user", "content": prompt_text},
                ],
            )

            output = completion.choices[0].message.content
            print('GPT4All: ', output)
            speak(output)
            print('\nSay', wake_word, 'to wake me up. \n')
            listening_for_wake_word = True
    except Exception as e:
        print("Prompt error: ", e)


def callback(recognizer, audio):
    print("I hear something")
    global listening_for_wake_word
    if listening_for_wake_word:
        listen_for_wake_word(audio)
    else:
        prompt_gpt(audio)


def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print('\nSay', wake_word, 'to wake me up. \n')
    r.listen_in_background(source, callback)
    while True:
        time.sleep(0.001)


if __name__ == '__main__':
    start_listening()
