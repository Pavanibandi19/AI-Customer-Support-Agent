"""SQLite Database Access Layer for Customer Support Agent.

Provides safe, parameterized query execution, automatic schema creation,
and initial data seeding.
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import config
from mcp_server.seed_data import CUSTOMERS, ORDERS, DELIVERIES, SUPPORT_TICKETS

logger = logging.getLogger(__name__)


def get_db_path(db_path: Optional[str | Path] = None) -> Path:
    """Resolve database path, defaulting to configured path."""
    if db_path is None:
        return config.DATABASE_PATH
    p = Path(db_path)
    return p if p.is_absolute() else config.BASE_DIR / p


def get_connection(db_path: Optional[str | Path] = None) -> sqlite3.Connection:
    """Create a SQLite connection with Row factory enabled."""
    path = get_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[str | Path] = None) -> None:
    """Initialize database tables and seed initial data if empty."""
    path = get_db_path(db_path)
    logger.info("Initializing database at: %s", path)

    with get_connection(path) as conn:
        cursor = conn.cursor()

        # Customers table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT NOT NULL,
                membership_tier TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )

        # Orders table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                order_date TEXT NOT NULL,
                status TEXT NOT NULL,
                total_amount REAL NOT NULL,
                item_name TEXT NOT NULL,
                category TEXT NOT NULL,
                is_final_sale INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            );
            """
        )

        # Deliveries table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS deliveries (
                delivery_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                carrier TEXT NOT NULL,
                tracking_number TEXT NOT NULL,
                status TEXT NOT NULL,
                estimated_delivery_date TEXT,
                actual_delivery_date TEXT,
                delivery_address TEXT NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(order_id)
            );
            """
        )

        # Support Tickets table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS support_tickets (
                ticket_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                order_id TEXT,
                issue TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                priority TEXT NOT NULL DEFAULT 'medium',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
                FOREIGN KEY (order_id) REFERENCES orders(order_id)
            );
            """
        )

        # Check if customers need seeding
        cursor.execute("SELECT COUNT(*) FROM customers;")
        if cursor.fetchone()[0] == 0:
            logger.info("Seeding customers...")
            cursor.executemany(
                """
                INSERT INTO customers (customer_id, name, email, phone, membership_tier, created_at)
                VALUES (:customer_id, :name, :email, :phone, :membership_tier, :created_at);
                """,
                CUSTOMERS,
            )

        # Check if orders need seeding
        cursor.execute("SELECT COUNT(*) FROM orders;")
        if cursor.fetchone()[0] == 0:
            logger.info("Seeding orders...")
            cursor.executemany(
                """
                INSERT INTO orders (order_id, customer_id, order_date, status, total_amount, item_name, category, is_final_sale)
                VALUES (:order_id, :customer_id, :order_date, :status, :total_amount, :item_name, :category, :is_final_sale);
                """,
                ORDERS,
            )

        # Check if deliveries need seeding
        cursor.execute("SELECT COUNT(*) FROM deliveries;")
        if cursor.fetchone()[0] == 0:
            logger.info("Seeding deliveries...")
            cursor.executemany(
                """
                INSERT INTO deliveries (delivery_id, order_id, carrier, tracking_number, status, estimated_delivery_date, actual_delivery_date, delivery_address)
                VALUES (:delivery_id, :order_id, :carrier, :tracking_number, :status, :estimated_delivery_date, :actual_delivery_date, :delivery_address);
                """,
                DELIVERIES,
            )

        # Check if support tickets need seeding
        cursor.execute("SELECT COUNT(*) FROM support_tickets;")
        if cursor.fetchone()[0] == 0:
            logger.info("Seeding support tickets...")
            cursor.executemany(
                """
                INSERT INTO support_tickets (ticket_id, customer_id, order_id, issue, description, status, priority, created_at, updated_at)
                VALUES (:ticket_id, :customer_id, :order_id, :issue, :description, :status, :priority, :created_at, :updated_at);
                """,
                SUPPORT_TICKETS,
            )

        conn.commit()
    logger.info("Database initialized successfully.")


# ---------------------------------------------------------
# Parameterized Query Functions
# ---------------------------------------------------------

def get_order_by_id(order_id: str, db_path: Optional[str | Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve an order by its ID with linked customer and delivery details."""
    clean_id = (order_id or "").strip().upper()
    if not clean_id:
        return None

    query = """
        SELECT o.order_id, o.customer_id, c.name AS customer_name, c.email AS customer_email,
               o.order_date, o.status, o.total_amount, o.item_name, o.category, o.is_final_sale,
               d.carrier, d.tracking_number, d.status AS delivery_status,
               d.estimated_delivery_date, d.actual_delivery_date, d.delivery_address
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        LEFT JOIN deliveries d ON o.order_id = d.order_id
        WHERE o.order_id = ?;
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, (clean_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_orders_by_customer_id(customer_id: str, db_path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    """Retrieve all orders for a given customer ID."""
    clean_id = (customer_id or "").strip().upper()
    if not clean_id:
        return []

    query = """
        SELECT o.order_id, o.customer_id, o.order_date, o.status, o.total_amount,
               o.item_name, o.category, o.is_final_sale,
               d.tracking_number, d.status AS delivery_status, d.estimated_delivery_date
        FROM orders o
        LEFT JOIN deliveries d ON o.order_id = d.order_id
        WHERE o.customer_id = ?
        ORDER BY o.order_date DESC;
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, (clean_id,))
        return [dict(row) for row in cursor.fetchall()]


def get_customer_by_id(customer_id: str, db_path: Optional[str | Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve customer profile by ID."""
    clean_id = (customer_id or "").strip().upper()
    if not clean_id:
        return None

    query = """
        SELECT customer_id, name, email, phone, membership_tier, created_at
        FROM customers
        WHERE customer_id = ?;
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, (clean_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_delivery_by_order_id(order_id: str, db_path: Optional[str | Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve delivery tracking details for an order ID."""
    clean_id = (order_id or "").strip().upper()
    if not clean_id:
        return None

    query = """
        SELECT delivery_id, order_id, carrier, tracking_number, status,
               estimated_delivery_date, actual_delivery_date, delivery_address
        FROM deliveries
        WHERE order_id = ?;
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, (clean_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def create_support_ticket(
    customer_id: str,
    order_id: Optional[str],
    issue: str,
    description: str,
    priority: str = "medium",
    db_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Create and persist a new support ticket in SQLite with parameterized queries."""
    clean_cust_id = (customer_id or "").strip().upper()
    clean_order_id = (order_id or "").strip().upper() if order_id else None
    clean_issue = (issue or "").strip()
    clean_desc = (description or "").strip()
    clean_priority = (priority or "medium").strip().lower()

    if not clean_cust_id:
        raise ValueError("customer_id is required to create a ticket.")
    if not clean_issue:
        raise ValueError("issue title is required to create a ticket.")
    if not clean_desc:
        raise ValueError("description is required to create a ticket.")

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Validate customer exists
        cursor.execute("SELECT customer_id FROM customers WHERE customer_id = ?;", (clean_cust_id,))
        if not cursor.fetchone():
            raise ValueError(f"Customer '{clean_cust_id}' does not exist in the database.")

        # If order_id provided, validate order exists and belongs to this customer
        if clean_order_id:
            cursor.execute(
                "SELECT customer_id FROM orders WHERE order_id = ?;", (clean_order_id,)
            )
            order_row = cursor.fetchone()
            if not order_row:
                raise ValueError(f"Order '{clean_order_id}' does not exist.")
            if order_row["customer_id"] != clean_cust_id:
                raise ValueError(
                    f"Order '{clean_order_id}' belongs to customer '{order_row['customer_id']}', not '{clean_cust_id}'."
                )

        # Generate next ticket ID: TICKET1001, TICKET1002, ...
        cursor.execute(
            """
            SELECT ticket_id FROM support_tickets
            WHERE ticket_id LIKE 'TICKET%'
            ORDER BY LENGTH(ticket_id) DESC, ticket_id DESC
            LIMIT 1;
            """
        )
        last_row = cursor.fetchone()
        if last_row and last_row["ticket_id"].startswith("TICKET"):
            try:
                num = int(last_row["ticket_id"].replace("TICKET", "")) + 1
            except ValueError:
                num = 1001
        else:
            num = 1001
        new_ticket_id = f"TICKET{num}"

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            """
            INSERT INTO support_tickets (ticket_id, customer_id, order_id, issue, description, status, priority, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?);
            """,
            (new_ticket_id, clean_cust_id, clean_order_id, clean_issue, clean_desc, clean_priority, now_str, now_str),
        )
        conn.commit()

        # Return created ticket
        cursor.execute("SELECT * FROM support_tickets WHERE ticket_id = ?;", (new_ticket_id,))
        return dict(cursor.fetchone())


def get_support_ticket(ticket_id: str, db_path: Optional[str | Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve an existing support ticket by ticket ID."""
    clean_id = (ticket_id or "").strip().upper()
    if not clean_id:
        return None

    query = """
        SELECT t.ticket_id, t.customer_id, c.name AS customer_name, t.order_id,
               t.issue, t.description, t.status, t.priority, t.created_at, t.updated_at
        FROM support_tickets t
        JOIN customers c ON t.customer_id = c.customer_id
        WHERE t.ticket_id = ?;
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, (clean_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def list_all_tickets(db_path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    """List all support tickets for dashboard inspection."""
    query = """
        SELECT ticket_id, customer_id, order_id, issue, description, status, priority, created_at, updated_at
        FROM support_tickets
        ORDER BY created_at DESC;
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]


def list_all_orders(db_path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    """List all orders for dashboard inspection."""
    query = """
        SELECT order_id, customer_id, order_date, status, total_amount, item_name, category, is_final_sale
        FROM orders
        ORDER BY order_id ASC;
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]


def list_all_customers(db_path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    """List all customers for dashboard inspection."""
    query = """
        SELECT customer_id, name, email, phone, membership_tier, created_at
        FROM customers
        ORDER BY customer_id ASC;
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]
