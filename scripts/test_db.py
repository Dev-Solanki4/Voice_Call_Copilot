import os
import logging
from dotenv import load_dotenv
from database.postgres_service import PetPoojaDB

logging.basicConfig(level=logging.INFO)
load_dotenv()

def test_conn():
    db_url = os.getenv("DATABASE_URL")
    logging.info(f"Connecting to: {db_url[:20]}...")
    db = PetPoojaDB()
    try:
        items = db.search_menu_items("%")
        logging.info(f"SUCCESS: Found {len(items)} items")
    except Exception as e:
        logging.error(f"FAILED: {e}")

if __name__ == "__main__":
    test_conn()
