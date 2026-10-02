import os
import re

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DATA_DIR = "data/"
DB_DIR = "chroma_store"



def load_books():
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()

    for doc in docs:
        text = re.sub(r"[ \t]+", " ", doc.page_content)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.encode("utf-8", errors="replace").decode("utf-8")
        doc.page_content = text.strip()

    return docs



def load_store():
    embeddings = OllamaEmbeddings(model="snowflake-arctic-embed2",base_url="https://lucrative-unhinge-boozy.ngrok-free.dev/")

    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    docs = load_books()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)

    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)



def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})



if __name__ == "__main__":
    retriever = build_retriever()

    results = retriever.invoke("what is Machine Learning?")
    
    for r in results:
        print(f"{r.page_content[:150]}...\n")