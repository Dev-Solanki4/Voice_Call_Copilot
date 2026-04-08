import os
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

class MenuVectorStore:
    _embeddings = None
    _vector_store = None

    def __init__(self, index_name: str = "restaurant-menu"):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.index_name = index_name

        if MenuVectorStore._embeddings is None:
            MenuVectorStore._embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=self.google_api_key
            )
        
        self.pc = Pinecone(api_key=self.api_key)
        
        # Ensure index exists (Serverless)
        if self.index_name not in self.pc.list_indexes().names():
            logging.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=3072, # Dimension for gemini-embedding-001/v4
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=MenuVectorStore._embeddings,
            pinecone_api_key=self.api_key
        )

    def search(self, query: str, k: int = 3):
        import time
        start = time.time()
        results = self.vector_store.similarity_search(query, k=k)
        duration = time.time() - start
        print(f"\033[93m[⌚ VECTOR] Pinecone search took {duration:.2f}s\033[0m", flush=True)
        return results

    def add_menu_items(self, documents: list):
        """Add menu items to the vector store."""
        self.vector_store.add_documents(documents)
