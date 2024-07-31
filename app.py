import logging
from functools import lru_cache

import torch
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Constants
MODEL_NAME = "llama3.1"
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "document_collection"

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("pikepdf._core").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Set up caching
set_llm_cache(InMemoryCache())

# Create a prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


@lru_cache(maxsize=100)
def get_cached_response(query):
    # This function will cache responses for repeated queries
    return generate_response(query)


def generate_response(query):
    try:
        docs = vectorstore.similarity_search(query, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        result = qa_chain.invoke({"query": query})
        return result["result"]
    except Exception as e:
        logger.error(f"Error during querying or response generation: {e}")
        return "I'm sorry, I encountered an error while processing your question."


def main():
    global llm, vectorstore

    # Initialize Ollama LLM
    try:
        llm = Ollama(model=MODEL_NAME)
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        return

    # Initialize embeddings
    try:
        embeddings = OllamaEmbeddings(model=MODEL_NAME)
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

        response = get_cached_response(query)
        print(response)


if __name__ == "__main__":
    main()
