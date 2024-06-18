import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class Chatbot:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.openai_api_key)
        self.messages = []

    def generate_response(self, user_message):
        self.messages.append({"role": "user", "content": 'system: Ты чат бот с женским голосом, ты помогаешь и общаешься как человек \nuser:'+user_message})
        response = self.client.chat.completions.create(
            messages=self.messages,
            model="gpt-3.5-turbo",
        )
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message
    def generate_joke(self):
        self.messages.append({"role": "user", "content": 'Шутка, пожалуйста!'})
        response = self.client.chat.completions.create(
            messages=self.messages,
            model="gpt-3.5-turbo",
        )
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message