from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
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

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2"
)

vector_db = Qdrant.from_existing_collection(
    path='http://localhost:6333',
    embedding=embedding_model,
    collection_name="assignment_rag"
)

# Take user input

user_query = input("Ask Something... ")

## Retrieve

search_results = vector_db.similarity_search(query=user_query)

context = "\n\n\n".join([f"page_content: {result.page_content}\n\nAuthor:{result.metadata["author"]}\n\nPage_number:{result.metadata["page_label"]}" for result in search_results])

## Augment

system_prompt = f"""
    You are a helpful AI Assitant who answers user query based on the available context retrieved from a pdf file along with page_content.

    You should only answer the user based on the following context.

    If the context is missing or irrelevant - You should simply tell the user, "I don't know!"

    Context: {context}

    Query: {user_query}

    Answer:
"""

## Generate

llm = LLM(model_name=str(os.getenv("GROQ_MODEL_NAME",""))).getLLM()

response = llm.invoke(system_prompt)

print(response.content)
