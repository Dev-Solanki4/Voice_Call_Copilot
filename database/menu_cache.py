"""
In-Memory Menu Cache — Load once at startup, filter in-memory for zero-latency lookups.
Replaces all per-turn Postgres queries in MenuAgent.
"""

import logging
from typing import List, Dict, Optional
from database.postgres_service import PetPoojaDB

logger = logging.getLogger(__name__)


class MenuCache:
    """Singleton in-memory cache for all menu data."""
    
    _instance: Optional["MenuCache"] = None
    
    def __init__(self):
        self.items: List[Dict] = []           # All menu items (full records)
        self.categories: List[str] = []        # Category names in sort order
        self.items_by_category: Dict[str, List[Dict]] = {}  # category_name → items
        self.items_by_id: Dict[str, Dict] = {}              # item_id → item
        self._loaded = False
    
    @classmethod
    def get_instance(cls) -> "MenuCache":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = MenuCache()
        return cls._instance
    
    def load(self, restaurant_id: str = None):
        """Load entire menu from Postgres into memory. Call once at startup."""
        db = PetPoojaDB()
        try:
            # 1. Load all categories
            self.categories = db.get_menu_categories(restaurant_id)
            logger.info(f"[MenuCache] Loaded {len(self.categories)} categories: {self.categories}")
            
            # 2. Load ALL items with category info (single query)
            conn = db._get_conn()
            with conn.cursor() as cur:
                rid = restaurant_id or db.__class__.__dict__.get("DEFAULT_RESTAURANT_ID", None)
                if rid is None:
                    from database.postgres_service import DEFAULT_RESTAURANT_ID
                    rid = DEFAULT_RESTAURANT_ID
                
                cur.execute("""
                    SELECT mi.*, mc.name as category_name
                    FROM menu_items mi
                    LEFT JOIN menu_categories mc ON mi.category_id = mc.id
                    WHERE mi.is_available = true AND mi.restaurant_id = %s::uuid
                    ORDER BY mc.name, mi.name
                """, (rid,))
                rows = cur.fetchall()
            
            # 3. Build in-memory indexes
            self.items = []
            self.items_by_category = {cat: [] for cat in self.categories}  # Pre-fill all categories
            self.items_by_id = {}
            
            for row in rows:
                item = dict(row) # Copy all columns directly
                item["id"] = str(item["id"]) # Ensure UUID is string
                item["price"] = float(item["price"])
                item["description"] = item.get("description") or ""
                item["is_veg"] = item.get("is_veg", False)
                item["category"] = item.get("category_name") or "Uncategorized"
                self.items.append(item)
                self.items_by_id[item["id"]] = item
                
                cat = item["category"]
                if cat not in self.items_by_category:
                    self.items_by_category[cat] = []
                self.items_by_category[cat].append(item)
            
            self._loaded = True
            logger.info(f"[MenuCache] Loaded {len(self.items)} menu items into memory")
            
            # Print the whole data for visibility as requested by the user
            print("\n" + "="*50)
            print("🚀 LOADING MENU DATA FROM DATABASE")
            print("="*50)
            for cat, items in self.items_by_category.items():
                print(f"\n📂 CATEGORY: {cat}")
                for item in items:
                    print(f"  - [{item['id'][:8]}] {item['name']} (₹{item['price']}) {'[VEG]' if item['is_veg'] else ''}")
            print("="*50 + "\n")
            
        except Exception as e:
            logger.error(f"[MenuCache] Failed to load menu: {e}")
            raise
        finally:
            try:
                db.close()
            except:
                pass
    
    # ── In-Memory Filter Methods (replace all DB queries) ──────────────
    
    def get_categories(self) -> List[str]:
        """Return all category names (sorted)."""
        return self.categories
    
    def get_items_by_category(self, category_name: str) -> List[Dict]:
        """Return items matching a category (case-insensitive partial match)."""
        cat_lower = category_name.lower()
        for cat, items in self.items_by_category.items():
            if cat_lower in cat.lower() or cat.lower() in cat_lower:
                return items
        return []
    
    def get_veg_items(self) -> List[Dict]:
        """Return all vegetarian items."""
        return [item for item in self.items if item.get("is_veg")]
    
    def search_items(self, query: str) -> List[Dict]:
        """Search items by name or description (case-insensitive, in-memory ILIKE)."""
        q = query.lower()
        results = []
        for item in self.items:
            if q in item["name"].lower() or q in item["description"].lower():
                results.append(item)
        return results
    
    def get_all_items(self) -> List[Dict]:
        """Return all items."""
        return self.items
    
    def get_item_names(self) -> List[str]:
        """Return all item names (for STT keyterms)."""
        return list(set(item["name"] for item in self.items))
    
    def get_item_by_id(self, item_id: str) -> Optional[Dict]:
        """Get a single item by ID."""
        return self.items_by_id.get(item_id)
