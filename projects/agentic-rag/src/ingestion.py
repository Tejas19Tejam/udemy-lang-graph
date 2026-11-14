from hashlib import sha256
from pathlib import Path

from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


CHROMA_DIR = Path(__file__).resolve().parent / ".chroma"
COLLECTION_NAME = "rag-chroma-minilm-l6"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_web_page(url: str) -> list[Document]:
    return UnstructuredLoader(
        web_url=url, chunking_strategy="basic", max_characters=1000000
    ).load()


urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]


def make_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"local_files_only": True},
    )


def ingest_documents() -> None:
    docs = [document for url in urls for document in load_web_page(url)]
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    chunks = splitter.split_documents(docs)
    ids = [
        sha256(
            f"{chunk.metadata.get('source', '')}\0{chunk.page_content}".encode()
        ).hexdigest()
        for chunk in chunks
    ]

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=make_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    vectorstore.add_documents(chunks, ids=ids)


retriever = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_DIR),
    embedding_function=make_embeddings(),
).as_retriever()


if __name__ == "__main__":
    ingest_documents()
