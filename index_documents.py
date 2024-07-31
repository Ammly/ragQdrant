import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache

# Constants
DATA_DIR = "./data"
MODEL_NAME = "llama3.1"
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "document_collection"

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("pikepdf._core").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Set up caching
set_llm_cache(InMemoryCache())


def load_and_split_documents(directory):
    try:
        loader = DirectoryLoader(directory, glob="**/*")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=lambda x: len(x) - 1
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        logger.error(f"Error loading or splitting documents: {e}")
        raise


def embed_documents_batch(embeddings, docs, batch_size=64):
    all_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        batch_embeddings = embeddings.embed_documents([doc.page_content for doc in batch])
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


def main():
    # Initialize embeddings
    try:
        embeddings = OllamaEmbeddings(model=MODEL_NAME)
    except Exception as e:
        logger.error(f"Error initializing embeddings: {e}")
        return

    # Load and split documents
    try:
        docs = load_and_split_documents(DATA_DIR)
    except Exception as e:
        logger.error(f"Error loading and splitting documents: {e}")
        return

    # Create vector store
    try:
        # Embed documents in batches
        embedded_docs = embed_documents_batch(embeddings, docs)

        vectorstore = Qdrant.from_embeddings(
            embedded_docs,
            embeddings,
            url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION_NAME,
            vector_name="document_vector",
            distance_func="Cosine"
        )
        logger.info(f"Documents successfully embedded and stored in Qdrant.")
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")


if __name__ == "__main__":
    main()
