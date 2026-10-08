import os
import json
import bs4

from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "data/"
DB_DIR = "chroma_store"

from dotenv import load_dotenv
load_dotenv()

def load_data():
    with open("data/scikit_learn.json", "r", encoding="utf-8") as f:
        SCIKIT_LEARN_URLS = json.load(f)

    docs = []

    for category_dict in SCIKIT_LEARN_URLS:
        for category, urls in category_dict.items():
            loader = WebBaseLoader(
                web_paths=urls,
                requests_per_second=2,
                bs_kwargs={
                    "parse_only": bs4.SoupStrainer(
                        "article",
                        class_="bd-article"
                    )
                },
                bs_get_text_kwargs={
                    "separator": "\n",
                    "strip": True
                }
            )

            cateogry_docs = loader.load()

            for doc in cateogry_docs:
                doc.metadata["category"] = category
                doc.metadata["library"] = "scikit-learn"

            docs.extend(cateogry_docs)

    return docs



def load_store():
    embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")

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

    docs = load_data()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    ).split_documents(docs)

    print("Web Pages:", len(docs))
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

    results = retriever.invoke("what is Supervised Learning?")

    for r in results:
        print(f"{r.page_content[:150]}...\n")