from langchain_groq import ChatGroq
from pydantic import SecretStr

from dotenv import load_dotenv
import os
load_dotenv()

class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llm = ChatGroq(model=model_name, api_key=SecretStr(os.getenv("GROQ_API_KEY","")))

    def getLLM(self):
        return self.llm