from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

## Loading

pdf_path = Path(__file__).parent / "docs" / "Assignment.pdf"
loader = PyPDFLoader(file_path=pdf_path)

docs=loader.load()

# print(docs[0])

## Chunking - Splitting

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents=docs)

## Create embeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2"
)

## Store embeddings into vector db

vector_store = Qdrant.from_documents(
    documents=chunks,
    embedding=embedding_model,
    path='http://localhost:6333',
    collection_name="assignment_rag"
)

print("Indexing done ...")