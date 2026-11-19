"""Automated tests for SQLite Database access layer."""

import os
import tempfile
import pytest
from pathlib import Path
from mcp_server.database import (
    init_db,
    get_order_by_id,
    get_orders_by_customer_id,
    get_customer_by_id,
    get_delivery_by_order_id,
    create_support_ticket,
    get_support_ticket,
    list_all_tickets,
    list_all_orders,
    list_all_customers,
)


@pytest.fixture
def temp_db():
    """Create an isolated temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    init_db(db_path)
    yield db_path
    if db_path.exists():
        try:
            os.remove(db_path)
        except Exception:
            pass


def test_database_initialization(temp_db):
    """Verify that tables and seed data are populated properly."""
    customers = list_all_customers(temp_db)
    orders = list_all_orders(temp_db)
    tickets = list_all_tickets(temp_db)

    assert len(customers) >= 10, "Expected at least 10 seed customers"
    assert len(orders) >= 20, "Expected at least 20 seed orders"
    assert len(tickets) >= 3, "Expected initial seed support tickets"


def test_get_order_by_id_valid(temp_db):
    """Verify retrieval of an existing order with joined customer and delivery info."""
    order = get_order_by_id("ORD1001", temp_db)
    assert order is not None
    assert order["order_id"] == "ORD1001"
    assert order["customer_id"] == "CUST1001"
    assert order["customer_name"] == "Alice Johnson"
    assert order["status"] == "delivered"
    assert order["delivery_status"] == "Delivered"
    assert order["carrier"] == "FedEx"
    assert order["tracking_number"] == "FDX-99881122"


def test_get_order_by_id_invalid(temp_db):
    """Verify that a non-existent order ID returns None without error."""
    order = get_order_by_id("ORD9999", temp_db)
    assert order is None

    # Empty and whitespace checks
    assert get_order_by_id("", temp_db) is None
    assert get_order_by_id("   ", temp_db) is None


def test_sql_injection_defense(temp_db):
    """Verify that parameterized queries protect against SQL injection attempts."""
    malicious_input = "ORD1001' OR '1'='1"
    order = get_order_by_id(malicious_input, temp_db)
    # The literal string will not match any legitimate order ID
    assert order is None


def test_get_orders_by_customer_id(temp_db):
    """Verify retrieval of customer orders."""
    orders = get_orders_by_customer_id("CUST1001", temp_db)
    assert len(orders) >= 2
    order_ids = [o["order_id"] for o in orders]
    assert "ORD1001" in order_ids
    assert "ORD1002" in order_ids


def test_get_delivery_by_order_id(temp_db):
    """Verify delivery record lookup."""
    delivery = get_delivery_by_order_id("ORD1002", temp_db)
    assert delivery is not None
    assert delivery["carrier"] == "UPS"
    assert delivery["status"] == "In Transit"
    assert delivery["estimated_delivery_date"] == "2026-10-23"


def test_create_support_ticket_success(temp_db):
    """Verify creating and persisting a new support ticket."""
    initial_count = len(list_all_tickets(temp_db))
    ticket = create_support_ticket(
        customer_id="CUST1003",
        order_id="ORD1004",
        issue="Cracked Display",
        description="Display arrived cracked during transit",
        priority="high",
        db_path=temp_db,
    )

    assert ticket is not None
    assert ticket["ticket_id"].startswith("TICKET")
    assert ticket["customer_id"] == "CUST1003"
    assert ticket["order_id"] == "ORD1004"
    assert ticket["issue"] == "Cracked Display"
    assert ticket["status"] == "open"
    assert ticket["priority"] == "high"

    # Verify count incremented
    new_count = len(list_all_tickets(temp_db))
    assert new_count == initial_count + 1

    # Verify retrieval
    retrieved = get_support_ticket(ticket["ticket_id"], temp_db)
    assert retrieved is not None
    assert retrieved["ticket_id"] == ticket["ticket_id"]
    assert retrieved["customer_name"] == "Carol White"


def test_create_support_ticket_validation_failures(temp_db):
    """Verify that ticket creation rejects invalid customers and missing fields."""
    # Invalid customer
    with pytest.raises(ValueError, match="does not exist"):
        create_support_ticket(
            customer_id="CUST9999",
            order_id="ORD1001",
            issue="Test Issue",
            description="Test Description",
            db_path=temp_db,
        )

    # Order belonging to another customer
    with pytest.raises(ValueError, match="belongs to customer"):
        create_support_ticket(
            customer_id="CUST1002",  # Bob Smith
            order_id="ORD1001",      # Belongs to Alice Johnson CUST1001
            issue="Test Issue",
            description="Test Description",
            db_path=temp_db,
        )

    # Missing issue
    with pytest.raises(ValueError, match="issue title is required"):
        create_support_ticket(
            customer_id="CUST1001",
            order_id="ORD1001",
            issue="",
            description="Desc",
            db_path=temp_db,
        )
