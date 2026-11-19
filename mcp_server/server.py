"""Independent MCP Server for AI Customer Support Agent.

Exposes official Model Context Protocol (MCP) tools, resources, and prompts
using the FastMCP API from the official MCP Python SDK.
"""

import sys
import json
import logging
from datetime import datetime, date
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mcp.server.fastmcp import FastMCP
from mcp_server.database import (
    init_db,
    get_order_by_id,
    get_orders_by_customer_id,
    get_customer_by_id,
    get_delivery_by_order_id,
    create_support_ticket as db_create_ticket,
    get_support_ticket as db_get_ticket,
)

# Configure logging to stderr so stdio JSON-RPC protocol on stdout remains clean
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mcp_server")

# Initialize FastMCP Server
mcp = FastMCP(
    "CustomerSupportServer",
    instructions="MCP Server providing customer support database operations, policies, and ticket management.",
)

# Ensure database is initialized upon server startup
init_db()


# =====================================================================
# MCP TOOLS
# =====================================================================

@mcp.tool()
def get_order(order_id: str) -> Dict[str, Any]:
    """Retrieve details and status for a specific order by its order ID (e.g. ORD1001).

    Args:
        order_id: The unique identifier of the order (e.g., 'ORD1001').

    Returns:
        Structured dictionary containing order details, status, item name, and delivery tracking.
    """
    clean_id = (order_id or "").strip().upper()
    if not clean_id:
        return {"success": False, "error": "Order ID cannot be empty."}

    order = get_order_by_id(clean_id)
    if not order:
        return {
            "success": False,
            "error": f"Order '{clean_id}' was not found in the customer support database.",
            "order_id": clean_id,
        }

    return {
        "success": True,
        "order": {
            "order_id": order["order_id"],
            "customer_id": order["customer_id"],
            "customer_name": order["customer_name"],
            "order_date": order["order_date"],
            "status": order["status"],
            "total_amount": order["total_amount"],
            "item_name": order["item_name"],
            "category": order["category"],
            "is_final_sale": bool(order["is_final_sale"]),
            "delivery_status": order["delivery_status"],
            "tracking_number": order["tracking_number"],
            "estimated_delivery_date": order["estimated_delivery_date"],
            "actual_delivery_date": order["actual_delivery_date"],
        },
    }


@mcp.tool()
def get_customer_orders(customer_id: str) -> Dict[str, Any]:
    """Retrieve all historical and current orders belonging to a specific customer ID (e.g. CUST1001).

    Args:
        customer_id: The unique identifier of the customer (e.g., 'CUST1001').

    Returns:
        Structured dictionary containing the customer's orders and profile summary.
    """
    clean_id = (customer_id or "").strip().upper()
    if not clean_id:
        return {"success": False, "error": "Customer ID cannot be empty."}

    customer = get_customer_by_id(clean_id)
    if not customer:
        return {
            "success": False,
            "error": f"Customer '{clean_id}' was not found in the customer support database.",
            "customer_id": clean_id,
        }

    orders = get_orders_by_customer_id(clean_id)
    return {
        "success": True,
        "customer_id": clean_id,
        "customer_name": customer["name"],
        "membership_tier": customer["membership_tier"],
        "order_count": len(orders),
        "orders": orders,
    }


@mcp.tool()
def check_delivery_status(order_id: str) -> Dict[str, Any]:
    """Retrieve carrier tracking information, current delivery status, and expected delivery date for an order.

    Args:
        order_id: The unique order identifier (e.g., 'ORD1001').

    Returns:
        Structured dictionary containing carrier, tracking number, status, and dates.
    """
    clean_id = (order_id or "").strip().upper()
    if not clean_id:
        return {"success": False, "error": "Order ID cannot be empty."}

    delivery = get_delivery_by_order_id(clean_id)
    if not delivery:
        # Check if the order itself exists
        order = get_order_by_id(clean_id)
        if not order:
            return {"success": False, "error": f"Order '{clean_id}' not found."}
        return {
            "success": False,
            "error": f"No delivery record found for order '{clean_id}'. Current order status: {order['status']}.",
            "order_id": clean_id,
        }

    return {
        "success": True,
        "order_id": clean_id,
        "carrier": delivery["carrier"],
        "tracking_number": delivery["tracking_number"],
        "delivery_status": delivery["status"],
        "estimated_delivery_date": delivery["estimated_delivery_date"],
        "actual_delivery_date": delivery["actual_delivery_date"],
        "delivery_address": delivery["delivery_address"],
    }


@mcp.tool()
def check_return_eligibility(order_id: str, eval_date: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate whether an order is eligible for return based on delivery date, return policy window (30 days), and item terms.

    Args:
        order_id: The unique order identifier (e.g., 'ORD1001').
        eval_date: Optional evaluation date string in YYYY-MM-DD format (defaults to current date).

    Returns:
        Structured dictionary indicating return eligibility (is_eligible: True/False) and explicit reason.
    """
    clean_id = (order_id or "").strip().upper()
    if not clean_id:
        return {"success": False, "error": "Order ID cannot be empty."}

    order = get_order_by_id(clean_id)
    if not order:
        return {"success": False, "error": f"Order '{clean_id}' not found in the database."}

    # Condition 1: Check order status
    order_status = (order["status"] or "").lower()
    if order_status == "cancelled":
        return {
            "success": True,
            "order_id": clean_id,
            "is_eligible": False,
            "reason": "Cancelled orders cannot be returned because the order was cancelled prior to fulfillment.",
        }

    if order_status != "delivered":
        return {
            "success": True,
            "order_id": clean_id,
            "is_eligible": False,
            "current_status": order_status,
            "reason": f"Order has not been delivered yet (current status: '{order_status}'). Returns can only be initiated after delivery.",
        }

    # Condition 2: Check Final Sale restriction
    if order["is_final_sale"]:
        return {
            "success": True,
            "order_id": clean_id,
            "item_name": order["item_name"],
            "is_eligible": False,
            "reason": "This item was purchased as Final Sale / Clearance and is strictly non-returnable and non-refundable.",
        }

    # Condition 3: Check 30-day window from actual delivery date
    if not order.get("actual_delivery_date"):
        return {
            "success": True,
            "order_id": clean_id,
            "item_name": order["item_name"],
            "is_eligible": False,
            "reason": "Order has no verified actual delivery date recorded. Returns require confirmed delivery.",
        }

    try:
        delivery_dt = datetime.strptime(str(order["actual_delivery_date"]), "%Y-%m-%d").date()
    except Exception:
        return {
            "success": False,
            "error": f"Invalid delivery date format for order '{clean_id}'.",
        }

    if eval_date:
        try:
            today = datetime.strptime(eval_date.strip(), "%Y-%m-%d").date()
        except Exception:
            today = date.today()
    else:
        today = date.today()

    days_elapsed = (today - delivery_dt).days

    RETURN_WINDOW_DAYS = 30
    if days_elapsed <= RETURN_WINDOW_DAYS:
        return {
            "success": True,
            "order_id": clean_id,
            "item_name": order["item_name"],
            "is_eligible": True,
            "delivery_date": str(delivery_dt),
            "days_since_delivery": days_elapsed,
            "return_window_days": RETURN_WINDOW_DAYS,
            "days_remaining": max(0, RETURN_WINDOW_DAYS - days_elapsed),
            "reason": f"Eligible for return. Delivered {days_elapsed} day(s) ago, within the {RETURN_WINDOW_DAYS}-day return window (Day {RETURN_WINDOW_DAYS} is inclusive).",
        }
    else:
        return {
            "success": True,
            "order_id": clean_id,
            "item_name": order["item_name"],
            "is_eligible": False,
            "delivery_date": str(delivery_dt),
            "days_since_delivery": days_elapsed,
            "return_window_days": RETURN_WINDOW_DAYS,
            "reason": f"Ineligible for return. Delivered {days_elapsed} days ago, which exceeds the {RETURN_WINDOW_DAYS}-day return window.",
        }


@mcp.tool()
def get_return_policy() -> Dict[str, Any]:
    """Retrieve the company return and refund policy guidelines.

    Returns:
        Structured dictionary detailing return windows, conditions, restocking fees, and instructions.
    """
    return {
        "success": True,
        "policy_name": "Standard Customer Return & Refund Policy",
        "return_window_days": 30,
        "eligible_condition": "Items must be in original, unused condition with all tags and packaging intact.",
        "return_shipping": "Free return shipping labels for damaged, defective, or incorrect items. Customer covers return postage for change-of-mind returns.",
        "non_returnable_items": [
            "Items marked as Final Sale or Clearance",
            "Opened personal care products and intimate apparel",
            "Perishable goods and grocery items",
            "Gift cards and digital download codes",
        ],
        "refund_timeline": "Refunds are processed to the original payment method within 5-7 business days after inspection at our warehouse.",
        "support_process": "To initiate a return for damaged or defective items, a support ticket must be created with order details.",
    }


@mcp.tool()
def create_support_ticket(
    customer_id: str,
    order_id: Optional[str],
    issue: str,
    description: str,
) -> Dict[str, Any]:
    """Create and persist a new customer support ticket in the database.

    Args:
        customer_id: The ID of the customer reporting the issue (e.g. 'CUST1003').
        order_id: The relevant order ID, or None if general issue (e.g. 'ORD1004').
        issue: Short title or summary of the issue (e.g. 'Damaged item received').
        description: Detailed explanation of the customer inquiry or problem.

    Returns:
        Structured dictionary confirming successful ticket creation and providing the persistent ticket ID.
    """
    clean_cust_id = (customer_id or "").strip().upper()
    clean_order_id = (order_id or "").strip().upper() if order_id else None
    clean_issue = (issue or "").strip()
    clean_desc = (description or "").strip()

    if not clean_cust_id:
        return {"success": False, "error": "customer_id is required to create a ticket."}
    if not clean_issue:
        return {"success": False, "error": "issue summary is required to create a ticket."}
    if not clean_desc:
        return {"success": False, "error": "description is required to create a ticket."}

    try:
        ticket = db_create_ticket(
            customer_id=clean_cust_id,
            order_id=clean_order_id,
            issue=clean_issue,
            description=clean_desc,
        )
        return {
            "success": True,
            "message": f"Support ticket {ticket['ticket_id']} successfully created and recorded.",
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "order_id": ticket["order_id"],
            "issue": ticket["issue"],
            "status": ticket["status"],
            "priority": ticket["priority"],
            "created_at": ticket["created_at"],
        }
    except ValueError as e:
        logger.warning("Ticket creation validation error: %s", str(e))
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error("Unexpected error creating ticket: %s", str(e))
        return {"success": False, "error": f"Database error creating ticket: {str(e)}"}


@mcp.tool()
def get_support_ticket(ticket_id: str) -> Dict[str, Any]:
    """Retrieve details and status for a previously created support ticket by ticket ID (e.g. TICKET1001).

    Args:
        ticket_id: Unique support ticket ID (e.g. 'TICKET1001').

    Returns:
        Structured dictionary containing ticket details, status, and linked order.
    """
    clean_id = (ticket_id or "").strip().upper()
    if not clean_id:
        return {"success": False, "error": "Ticket ID cannot be empty."}

    ticket = db_get_ticket(clean_id)
    if not ticket:
        return {
            "success": False,
            "error": f"Support ticket '{clean_id}' was not found in the database.",
            "ticket_id": clean_id,
        }

    return {
        "success": True,
        "ticket": {
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "customer_name": ticket["customer_name"],
            "order_id": ticket["order_id"],
            "issue": ticket["issue"],
            "description": ticket["description"],
            "status": ticket["status"],
            "priority": ticket["priority"],
            "created_at": ticket["created_at"],
            "updated_at": ticket["updated_at"],
        },
    }


# =====================================================================
# MCP RESOURCES
# =====================================================================

@mcp.resource("support://policies/returns")
def get_return_policy_resource() -> str:
    """Official Company Return, Replacement, and Refund Policy Document."""
    return """# E-Commerce Customer Support Return Policy
Version: 2026.2
Effective Date: January 1, 2026

## 1. Return Window
Customers may return items within thirty (30) calendar days from the verified date of delivery.

## 2. Eligibility Requirements
- Item must be in its original condition, unworn, unwashed, and in original packaging.
- Valid proof of purchase (Order ID) is required for all return requests.
- Returns cannot be initiated for orders that have not yet been delivered or are currently in transit.

## 3. Exceptions & Non-Returnable Items
The following items are strictly non-returnable:
- Clearance and items marked 'Final Sale' at the time of purchase.
- Opened hygienic and personal care products.
- Digital products, gift certificates, and software keys.

## 4. Damaged or Defective Items
- If an item arrives damaged or broken, the customer must report it within 7 calendar days of delivery.
- A support ticket will be created immediately with priority escalation.
- Return shipping for verified damaged items is 100% complimentary.

## 5. Refund Processing
- Refunds are credited to the original method of payment within 5 to 7 business days following warehouse inspection.
"""


@mcp.resource("support://knowledge/faq")
def get_support_faq_resource() -> str:
    """Frequently Asked Questions (FAQ) for Customer Support."""
    return """# Customer Support Knowledge Base - Frequently Asked Questions (FAQ)

### Q1: How do I track my order delivery?
A: You can track your order at any time by providing your Order ID (e.g., ORD1001). Our tracking system reports carrier name, tracking number, current transit status, and estimated delivery dates.

### Q2: What should I do if my package has not arrived by the expected date?
A: Check the delivery status using your Order ID. If the package shows delayed or stalled in transit for more than 48 hours past the estimated delivery date, our support agent can create an escalation ticket with carrier logistics.

### Q3: What is the return window for my purchase?
A: Standard purchases are eligible for return within 30 days of delivery. Items marked as Final Sale or items delivered over 30 days ago cannot be returned.

### Q4: My item arrived damaged or broken. What should I do?
A: We apologize for the inconvenience! Please provide your Order ID and describe the damage. Our agent will verify the order and immediately create an official support ticket (e.g., TICKET1004) for priority replacement or refund.

### Q5: How do I check the status of my existing support ticket?
A: Provide your Support Ticket ID (e.g., TICKET1001). The agent will look up the current ticket state (open, in progress, resolved, closed) and recent updates.

### Q6: Can I return an item that hasn't arrived yet?
A: No. Return eligibility requires delivery verification. If your order is still processing, you may request order cancellation instead.
"""


# =====================================================================
# MCP PROMPTS
# =====================================================================

@mcp.prompt("customer_support_response")
def customer_support_response_prompt(query: str = "", evidence: str = "") -> str:
    """Prompt template for formulating an evidence-grounded, professional customer response."""
    return f"""You are a professional, empathetic, and accurate AI Customer Support Agent.

CUSTOMER QUERY:
"{query}"

VERIFIED MCP TOOL EVIDENCE:
{evidence}

INSTRUCTIONS:
1. Ground your answer ONLY in the verified tool evidence above.
2. If an order, ticket, or delivery detail is missing or not found in evidence, explicitly state that fact. NEVER invent dates, tracking numbers, or ticket IDs.
3. Be concise, polite, and helpful.
4. If a support ticket was created, clearly state the confirmed Ticket ID and advise next steps.
5. If the customer's request was ambiguous or missing an Order ID, politely ask for the required information.
"""


@mcp.prompt("resolve_customer_issue")
def resolve_customer_issue_prompt(
    issue_description: str = "",
    customer_id: str = "",
    order_id: str = "",
) -> str:
    """Prompt template for diagnosing a customer issue and determining required resolution steps."""
    return f"""You are an expert customer resolution specialist.

ISSUE: {issue_description}
CUSTOMER ID: {customer_id or 'Not provided'}
ORDER ID: {order_id or 'Not provided'}

RESOLUTION WORKFLOW:
1. UNDERSTAND: Identify the customer's core problem (delivery delay, damaged goods, return request, or general inquiry).
2. VERIFY: Determine which MCP tools must be called to verify order state, delivery status, or policy conditions.
3. ACTION: If damage or defect is reported, create a support ticket with accurate customer and order linkage.
4. COMMUNICATE: Formulate a clear, helpful resolution with specific next steps.
"""


# =====================================================================
# MAIN ENTRYPOINT
# =====================================================================

if __name__ == "__main__":
    logger.info("Starting CustomerSupportServer over stdio transport...")
    mcp.run()
