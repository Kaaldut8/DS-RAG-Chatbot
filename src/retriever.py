import os
import re

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "data/"
DB_DIR = "chroma_store"

from dotenv import load_dotenv
load_dotenv()

def load_books():
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()

    for doc in docs:
        text = re.sub(r"[ \t]+", " ", doc.page_content)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.encode("utf-8", errors="replace").decode("utf-8")
        doc.page_content = text.strip()

    return docs



def load_store():
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    if os.path.exists(DB_DIR):
        print("Loading existing Chroma...")

        store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
        )

        print("Documents in Chroma:", store._collection.count())

        if store._collection.count() > 0:
            return store

    print("Creating new Chroma...")

    docs = load_books()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)

    print("PDF documents:", len(docs))
    print("Chunks:", len(chunks))

    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )

    print("Documents in Chroma:", store._collection.count())

    return store



def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})



if __name__ == "__main__":
    retriever = build_retriever()

    results = retriever.invoke("what is Machine Learning?")
    
    for r in results:
        print(f"{r.page_content[:150]}...\n")