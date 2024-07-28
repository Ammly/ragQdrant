import logging

import torch
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

# Constants
MODEL_NAME = "llama3.1"
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "document_collection"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Initialize Ollama LLM
    try:
        llm = Ollama(model=MODEL_NAME)
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        return

    # Initialize embeddings
    try:
        embeddings = OllamaEmbeddings(model=MODEL_NAME)
        # embeddings = OllamaEmbeddings(model=MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")

    except Exception as e:
        logger.error(f"Error initializing embeddings: {e}")
        return

    # Connect to Qdrant vector store
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL)
        vectorstore = QdrantVectorStore(
            client=qdrant_client,
            embedding=embeddings,
            collection_name=QDRANT_COLLECTION_NAME,
        )
    except Exception as e:
        logger.error(f"Error connecting to Qdrant: {e}")
        return

    print("Chat with your documents. Type 'quit' to exit.")

    while True:
        query = input("> ")
        if query.lower() == 'quit':
            break

        try:
            # Retrieve relevant documents
            relevant_docs = vectorstore.similarity_search(query, k=4)
            # Generate context
            context = "Based on the following documents:\n\n"
            for doc in relevant_docs:
                context += f"Document: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}\n\n"
            context += f"Answer the following question: {query}"

            # Generate response
            response = llm.invoke(context)
            print(response)
        except Exception as e:
            logger.error(f"Error during querying or response generation: {e}")


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
