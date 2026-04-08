import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import json

# Default restaurant ID for this session (from manual check)
DEFAULT_RESTAURANT_ID = "73313cb0-dcd4-4f03-94e0-5ec7aaf711ad"

class PetPoojaDB:
    def __init__(self, db_url: str = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.conn = None

    def _get_conn(self):
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)
        return self.conn

    def search_menu_items(self, query: str, restaurant_id: str = None):
        conn = self._get_conn()
        with conn.cursor() as cur:
            rid = restaurant_id or DEFAULT_RESTAURANT_ID
            sql = "SELECT * FROM menu_items WHERE (name ILIKE %s OR description ILIKE %s) AND is_available = true AND restaurant_id = %s::uuid"
            params = (f"%{query}%", f"%{query}%", rid)
            cur.execute(sql, params)
            return cur.fetchall()

    def get_all_menu_items(self, restaurant_id: str = None):
        conn = self._get_conn()
        with conn.cursor() as cur:
            rid = restaurant_id or DEFAULT_RESTAURANT_ID
            sql = "SELECT name, price, description, is_veg FROM menu_items WHERE is_available = true AND restaurant_id = %s::uuid"
            cur.execute(sql, (rid,))
            return cur.fetchall()

    def get_menu_categories(self, restaurant_id: str = None):
        conn = self._get_conn()
        with conn.cursor() as cur:
            rid = restaurant_id or DEFAULT_RESTAURANT_ID
            sql = "SELECT name FROM menu_categories WHERE is_active = true AND restaurant_id = %s::uuid ORDER BY sort_order"
            cur.execute(sql, (rid,))
            return [row['name'] for row in cur.fetchall()]

    def get_items_by_category_name(self, category_name: str, restaurant_id: str = None):
        conn = self._get_conn()
        with conn.cursor() as cur:
            rid = restaurant_id or DEFAULT_RESTAURANT_ID
            sql = """
                SELECT mi.name, mi.price, mi.description, mi.is_veg 
                FROM menu_items mi
                JOIN menu_categories mc ON mi.category_id = mc.id
                WHERE mc.name ILIKE %s AND mi.is_available = true AND mi.restaurant_id = %s::uuid
            """
            cur.execute(sql, (f"%{category_name}%", rid))
            return cur.fetchall()

    def get_veg_items(self, restaurant_id: str = None):
        conn = self._get_conn()
        with conn.cursor() as cur:
            rid = restaurant_id or DEFAULT_RESTAURANT_ID
            sql = "SELECT name, price, description FROM menu_items WHERE is_veg = true AND is_available = true AND restaurant_id = %s::uuid"
            cur.execute(sql, (rid,))
            return cur.fetchall()

    def get_inventory_status(self, menu_item_id: str):
        conn = self._get_conn()
        with conn.cursor() as cur:
            # Cast menu_item_id to uuid explicitly
            sql = """
                SELECT i.name, i.current_stock as stock, ri.quantity_required as required
                FROM menu_items mi
                JOIN recipes r ON mi.id = r.menu_item_id
                JOIN recipe_ingredients ri ON r.id = ri.recipe_id
                JOIN inventory i ON ri.ingredient_id = i.id
                WHERE mi.id = %s::uuid
            """
            cur.execute(sql, (menu_item_id,))
            return cur.fetchall()

    def create_order(self, restaurant_id: str = None, total: float = 0.0, tax: float = 0.0, order_type: str = "Call", payment_mode: str = None, dining_type: str = None, customer_name: str = None):
        import uuid
        conn = self._get_conn()
        with conn.cursor() as cur:
            rid = restaurant_id or DEFAULT_RESTAURANT_ID
            cgst = tax / 2.0
            sgst = tax / 2.0
            subtotal = total - tax
            
            # Generate a simple unique order number e.g., ORD-A1B2C3
            order_number = f"ORD-{uuid.uuid4().hex[:6].upper()}"
            
            sql = """
                INSERT INTO orders (restaurant_id, order_number, subtotal, total, cgst, sgst, status, order_type, payment_mode, dining_type, customer_name, created_at)
                VALUES (%s::uuid, %s, %s, %s, %s, %s, 'pending', %s, %s, %s, %s, NOW())
                RETURNING id
            """
            cur.execute(sql, (rid, order_number, subtotal, total, cgst, sgst, order_type, payment_mode, dining_type, customer_name))
            order_id = cur.fetchone()['id']
            conn.commit()
            return {"order_id": order_id, "order_number": order_number}

    def add_order_items(self, order_id: str, items: list):
        conn = self._get_conn()
        with conn.cursor() as cur:
            for item in items:
                # Cast order_id and menu_item_id to uuid
                sql = """
                    INSERT INTO order_items (order_id, menu_item_id, item_name, quantity, unit_price, modifiers, subtotal)
                    VALUES (%s::uuid, %s::uuid, %s, %s, %s, %s, %s)
                """
                cur.execute(sql, (
                    order_id, 
                    item['menu_item_id'], 
                    item.get('name', 'Unknown Item'), 
                    item['quantity'], 
                    item.get('price', 0.0),
                    json.dumps(item.get('modifiers', [])),
                    item['subtotal']
                ))
            conn.commit()

    def update_inventory_post_order(self, order_id: str):
        conn = self._get_conn()
        with conn.cursor() as cur:
            # Deduct current_stock
            sql = """
                UPDATE inventory i
                SET current_stock = i.current_stock - (oi.quantity * ri.quantity_required)
                FROM order_items oi
                JOIN recipes r ON oi.menu_item_id = r.menu_item_id
                JOIN recipe_ingredients ri ON r.id = ri.recipe_id
                WHERE oi.order_id = %s::uuid AND i.id = ri.ingredient_id
            """
            cur.execute(sql, (order_id,))
            
            # Log transactions
            sql_log = """
                INSERT INTO inventory_transactions (restaurant_id, ingredient_id, quantity_change, transaction_type, reference_id, created_at)
                SELECT o.restaurant_id, ri.ingredient_id, -(oi.quantity * ri.quantity_required), 'sale', oi.order_id::uuid, NOW()
                FROM order_items oi
                JOIN orders o ON oi.order_id = o.id
                JOIN recipes r ON oi.menu_item_id = r.menu_item_id
                JOIN recipe_ingredients ri ON r.id = ri.recipe_id
                WHERE oi.order_id = %s::uuid
            """
            cur.execute(sql_log, (order_id,))
            conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
