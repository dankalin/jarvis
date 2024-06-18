import pyttsx3
import pythoncom

class TextToSpeech:
    def __init__(self):
        pythoncom.CoInitialize()
        self.engine = pyttsx3.init()
    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def stop(self):
        self.engine.endLoop()