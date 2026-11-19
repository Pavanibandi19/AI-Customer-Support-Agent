"""Automated tests for FastMCP Server and official MCP Client over stdio transport."""

import pytest
import pytest_asyncio
from agent.mcp_client import MCPClientWrapper


@pytest.fixture
def mcp_client():
    """Fixture providing an official MCP client connected to the local server."""
    return MCPClientWrapper()


@pytest.mark.asyncio
async def test_tool_discovery(mcp_client):
    """Verify that all 7 required MCP tools are discovered."""
    tools = await mcp_client.list_tools()
    tool_names = [t["name"] for t in tools]

    expected_tools = [
        "get_order",
        "get_customer_orders",
        "check_delivery_status",
        "check_return_eligibility",
        "get_return_policy",
        "create_support_ticket",
        "get_support_ticket",
    ]
    for exp in expected_tools:
        assert exp in tool_names, f"Tool '{exp}' missing from MCP discovery"


@pytest.mark.asyncio
async def test_resource_discovery_and_reading(mcp_client):
    """Verify that required resources are registered and readable."""
    resources = await mcp_client.list_resources()
    resource_uris = [r["uri"] for r in resources]

    assert "support://policies/returns" in resource_uris
    assert "support://knowledge/faq" in resource_uris

    # Read returns policy resource
    content = await mcp_client.read_resource("support://policies/returns")
    assert "Return Policy" in content
    assert "30" in content

    # Read FAQ resource
    faq_content = await mcp_client.read_resource("support://knowledge/faq")
    assert "Frequently Asked Questions" in faq_content


@pytest.mark.asyncio
async def test_prompt_discovery_and_fetching(mcp_client):
    """Verify prompt templates are discoverable and instantiable."""
    prompts = await mcp_client.list_prompts()
    prompt_names = [p["name"] for p in prompts]

    assert "customer_support_response" in prompt_names
    assert "resolve_customer_issue" in prompt_names

    rendered = await mcp_client.get_prompt("customer_support_response", {"query": "Test query", "evidence": "Facts"})
    assert "Test query" in rendered


@pytest.mark.asyncio
async def test_tool_get_order(mcp_client):
    """Verify get_order returns valid structured details or safe error."""
    # Valid order
    res = await mcp_client.call_tool("get_order", {"order_id": "ORD1001"})
    assert res["success"] is True
    assert res["order"]["order_id"] == "ORD1001"
    assert res["order"]["item_name"] == "Wireless Noise-Canceling Headphones"

    # Non-existent order
    res_invalid = await mcp_client.call_tool("get_order", {"order_id": "ORD9999"})
    assert res_invalid["success"] is False
    assert "not found" in res_invalid["error"].lower()

    # Empty order ID
    res_empty = await mcp_client.call_tool("get_order", {"order_id": ""})
    assert res_empty["success"] is False


@pytest.mark.asyncio
async def test_tool_get_customer_orders(mcp_client):
    """Verify get_customer_orders lists all orders for customer."""
    res = await mcp_client.call_tool("get_customer_orders", {"customer_id": "CUST1001"})
    assert res["success"] is True
    assert res["customer_name"] == "Alice Johnson"
    assert len(res["orders"]) >= 2


@pytest.mark.asyncio
async def test_tool_check_delivery_status(mcp_client):
    """Verify check_delivery_status returns carrier tracking."""
    res = await mcp_client.call_tool("check_delivery_status", {"order_id": "ORD1002"})
    assert res["success"] is True
    assert res["carrier"] == "UPS"
    assert res["delivery_status"] == "In Transit"
    assert res["estimated_delivery_date"] == "2026-10-23"


@pytest.mark.asyncio
async def test_tool_check_return_eligibility_scenarios(mcp_client):
    """Verify return eligibility logic across varied order states."""
    # 1. Delivered within 30 days -> Eligible
    res_eligible = await mcp_client.call_tool("check_return_eligibility", {"order_id": "ORD1004", "eval_date": "2026-10-25"})
    assert res_eligible["success"] is True
    assert res_eligible["is_eligible"] is True
    assert "within the 30-day" in res_eligible["reason"]

    # 2. Delivered > 30 days ago -> Ineligible
    res_expired = await mcp_client.call_tool("check_return_eligibility", {"order_id": "ORD1003"})
    assert res_expired["success"] is True
    assert res_expired["is_eligible"] is False
    assert "exceeds" in res_expired["reason"]

    # 3. Final Sale item -> Ineligible
    res_final_sale = await mcp_client.call_tool("check_return_eligibility", {"order_id": "ORD1007"})
    assert res_final_sale["success"] is True
    assert res_final_sale["is_eligible"] is False
    assert "Final Sale" in res_final_sale["reason"]

    # 4. Cancelled order -> Ineligible
    res_cancelled = await mcp_client.call_tool("check_return_eligibility", {"order_id": "ORD1006"})
    assert res_cancelled["success"] is True
    assert res_cancelled["is_eligible"] is False
    assert "cancelled" in res_cancelled["reason"].lower()

    # 5. Processing order -> Ineligible
    res_processing = await mcp_client.call_tool("check_return_eligibility", {"order_id": "ORD1005"})
    assert res_processing["success"] is True
    assert res_processing["is_eligible"] is False
    assert "not been delivered yet" in res_processing["reason"]


@pytest.mark.asyncio
async def test_tool_get_return_policy(mcp_client):
    """Verify company return policy tool."""
    res = await mcp_client.call_tool("get_return_policy", {})
    assert res["success"] is True
    assert res["return_window_days"] == 30
    assert len(res["non_returnable_items"]) > 0


@pytest.mark.asyncio
async def test_tool_ticket_lifecycle(mcp_client):
    """Verify creating and retrieving support tickets via MCP tools."""
    # Create ticket
    create_res = await mcp_client.call_tool("create_support_ticket", {
        "customer_id": "CUST1003",
        "order_id": "ORD1004",
        "issue": "Damaged Keyboard Spacebar",
        "description": "The spacebar is fractured on arrival.",
    })
    assert create_res["success"] is True
    ticket_id = create_res["ticket_id"]
    assert ticket_id.startswith("TICKET")

    # Retrieve ticket
    get_res = await mcp_client.call_tool("get_support_ticket", {"ticket_id": ticket_id})
    assert get_res["success"] is True
    assert get_res["ticket"]["ticket_id"] == ticket_id
    assert get_res["ticket"]["issue"] == "Damaged Keyboard Spacebar"
    assert get_res["ticket"]["customer_name"] == "Carol White"
