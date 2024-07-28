import logging

import torch
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import DirectoryLoader


# Constants
DATA_DIR = "./data"
MODEL_NAME = "llama3.1"
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "document_collection"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_split_documents(directory):
    try:
        loader = DirectoryLoader(directory, glob="**/*")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        logger.error(f"Error loading or splitting documents: {e}")
        raise


def main():
    # Initialize embeddings
    try:
        embeddings = OllamaEmbeddings(model=MODEL_NAME)
        # embeddings = OllamaEmbeddings(model=MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")

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
        vectorstore = Qdrant.from_documents(
            docs,
            embeddings,
            url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION_NAME,
        )
        logger.info("Documents successfully embedded and stored in Qdrant.")
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")


def is_cuda_available():
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("CUDA is not available. Using CPU.")
        return False


if __name__ == "__main__":
    is_cuda_available()
    main()
