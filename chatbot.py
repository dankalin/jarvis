import os
from langchain.document_loaders import TextLoader
from langchain.embeddings import BM25Embeddings
from langchain.vectorstores import SimpleVectorStore
from langchain.retrievers import BM25Retriever
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class Chatbot:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.openai_api_key)
        self.messages = []
        self.jokes_retriever = self._setup_jokes_retriever()

    def _setup_jokes_retriever(self):
        loader = TextLoader('jokes.txt')
        documents = loader.load_and_split(chunk_size=500)
        embeddings = BM25Embeddings()
        vector_store = SimpleVectorStore.from_documents(documents, embeddings)
        retriever = BM25Retriever(vector_store=vector_store)
        return retriever

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
        query = 'Шутка, пожалуйста!'
        retrieved_docs = self.jokes_retriever.retrieve(query)
        if retrieved_docs:
            joke = retrieved_docs[0].text
            assistant_message = joke
        else:
            self.messages.append({"role": "user", "content": query})
            response = self.client.chat.completions.create(
                messages=self.messages,
                model="gpt-3.5-turbo",
            )
            assistant_message = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message


