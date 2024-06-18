import os
import struct
import wave
import asyncio
from datetime import datetime

from pvrecorder import PvRecorder
import pvporcupine

from speech_recognizer import SpeechRecognizer
from chatbot import Chatbot
from text_to_speech import TextToSpeech
from dotenv import load_dotenv
load_dotenv()
output_path = 'jarvis.wav'
access_key = os.environ.get("ACCESS_KEY")
keyword_path = 'wake_word/pomoshnik_ru_windows_v3_0_0.ppn'
model_path = './wake_word/porcupine_params_ru.pv'
keyword_stop_path = 'wake_word/xvatit_ru_windows_v3_0_0.ppn'


async def main():
    speech_recognizer = SpeechRecognizer()
    chatbot = Chatbot()
    text_to_speech = TextToSpeech()
    porcupine = pvporcupine.create(
        access_key=access_key,
        keyword_paths=[keyword_path, keyword_stop_path],
        model_path=model_path,
        sensitivities=[0.7, 0.7]
    )
    recorder = PvRecorder(frame_length=512, device_index=0)
    print("pvrecorder version: %s" % recorder.version)

    recorder.start()
    print("Using device: %s" % recorder.selected_device)

    wavfile = None
    silence_timer = None
    try:
        if output_path is not None:
            wavfile = wave.open(output_path, "w")
            wavfile.setparams((1, 2, recorder.sample_rate, recorder.frame_length, "NONE", "NONE"))

        while True:
            frame = recorder.read()
            if wavfile is not None:
                wavfile.writeframes(struct.pack("h" * len(frame), *frame))
            result = porcupine.process(frame)
            if result == 0:
                print('Detected wake word')
                silence_timer = None
                result = speech_recognizer.recognize(output_path)
                user_message = result["text"]
                assistant_message = chatbot.generate_response(user_message)
                print("User: ", user_message)
                print("GPT-3.5 Response: ", assistant_message)
                text_to_speech.speak(assistant_message)
            elif result == 1:
                print('Detected stop word')
                break
            else:
                if silence_timer is None:
                    silence_timer = datetime.now()
                else:
                    elapsed_time = (datetime.now() - silence_timer).total_seconds()
                    if elapsed_time >= 10:
                        print("Generating joke...")
                        assistant_message = chatbot.generate_joke()
                        print("GPT-3.5 Joke: ", assistant_message)
                        text_to_speech.speak(assistant_message)
                        silence_timer = None

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if wavfile is not None:
            wavfile.close()
        recorder.stop()
        porcupine.delete()

if __name__ == '__main__':
    asyncio.run(main())
