import os
import logging
from dotenv import load_dotenv
from database.postgres_service import PetPoojaDB
from database.vector_store import MenuVectorStore
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
load_dotenv()

def seed():
    db = PetPoojaDB()
    vector_store = MenuVectorStore()
    
    # Fetch all menu items
    items = db.search_menu_items("%") # Get all items
    logging.info(f"Fetched {len(items)} items from Supabase")
    
    documents = []
    for item in items:
        content = f"Name: {item['name']}\nDescription: {item['description']}\nPrice: ₹{item['price']}"
        # You can add category or other metadata here
        doc = Document(
            page_content=content,
            metadata={
                "id": str(item["id"]),
                "name": item["name"],
                "price": float(item["price"]),
                "restaurant_id": str(item["restaurant_id"]) if item.get("restaurant_id") else "unknown"
            }
        )
        documents.append(doc)
    
    if documents:
        vector_store.add_menu_items(documents)
        logging.info("Successfully seeded Pinecone index")
    else:
        logging.warning("No items found to seed")

if __name__ == "__main__":
    seed()
