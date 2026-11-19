"""Automated tests for Agent Orchestrator, Planner, and Reflection modules."""

import pytest
from agent.orchestrator import AgentOrchestrator
from agent.planner import AgentPlanner
from agent.reflection import AgentReflection, VerificationResult


@pytest.fixture
def orchestrator():
    """Fixture providing initialized orchestrator."""
    return AgentOrchestrator()


@pytest.fixture
def planner():
    """Fixture providing AgentPlanner."""
    return AgentPlanner()


def test_entity_extraction(planner):
    """Verify regex entity extraction for orders, customers, and tickets."""
    q1 = "Where is order ORD1001?"
    entities1 = planner.extract_entities(q1)
    assert entities1.get("order_id") == "ORD1001"

    q2 = "Customer CUST1002 has an issue with TICKET1001"
    entities2 = planner.extract_entities(q2)
    assert entities2.get("customer_id") == "CUST1002"
    assert entities2.get("ticket_id") == "TICKET1001"

    q3 = "Hello, can you help me?"
    entities3 = planner.extract_entities(q3)
    assert len(entities3) == 0


def test_plan_intent_classification(planner):
    """Verify intent mapping for key question types."""
    # Order tracking
    p1 = planner.plan("Where is order ORD1001?")
    assert p1.intent == "order_tracking"
    assert not p1.needs_clarification

    # Delivery follow-up
    p2 = planner.plan("My order ORD1002 has not arrived. When will it be delivered?")
    assert p2.intent == "delivery_followup"

    # Return eligibility
    p3 = planner.plan("Can I return order ORD1003?")
    assert p3.intent == "return_eligibility"

    # Damaged item
    p4 = planner.plan("My order ORD1004 arrived damaged. Please create a support ticket.")
    assert p4.intent == "damaged_item_support"

    # Ambiguous
    p5 = planner.plan("I need help with my order.")
    assert p5.needs_clarification is True
    assert "ORD1001" in p5.clarification_question

    # Return policy FAQ
    p6 = planner.plan("What is your return policy?")
    assert p6.intent == "faq_policy"


def test_scenario_order_tracking(orchestrator):
    """Scenario 1: Order tracking for delivered order."""
    trace = orchestrator.run_sync("Where is order ORD1001?")
    assert trace.intent == "order_tracking"
    assert trace.verification["passed"] is True
    assert "ORD1001" in trace.final_response
    assert "Wireless Noise-Canceling Headphones" in trace.final_response
    assert "Delivered" in trace.final_response


def test_scenario_delivery_followup(orchestrator):
    """Scenario 2: Delivery follow-up for order in transit."""
    trace = orchestrator.run_sync("My order ORD1002 has not arrived. When will it be delivered?")
    assert trace.intent == "delivery_followup"
    assert "In Transit" in trace.final_response
    assert "2026-10-23" in trace.final_response  # Expected delivery date


def test_scenario_return_eligibility_expired(orchestrator):
    """Scenario 3: Return eligibility for expired order."""
    trace = orchestrator.run_sync("Can I return order ORD1003?")
    assert trace.intent == "return_eligibility"
    assert "not eligible for return" in trace.final_response.lower()
    assert "30-day" in trace.final_response


def test_scenario_damaged_item_ticket_creation(orchestrator):
    """Scenario 4: Damaged item creates persistent support ticket."""
    trace = orchestrator.run_sync("My order ORD1004 arrived damaged. Please create a support ticket.")
    assert trace.intent == "damaged_item_support"
    assert trace.verification["passed"] is True
    assert trace.verification["ticket_created"] is True
    assert trace.verification["created_ticket_id"] is not None
    assert trace.verification["created_ticket_id"].startswith("TICKET")
    assert trace.verification["created_ticket_id"] in trace.final_response


def test_scenario_existing_ticket_status(orchestrator):
    """Scenario 5: Check status of existing support ticket."""
    trace = orchestrator.run_sync("What is the status of ticket TICKET1001?")
    assert trace.intent == "get_ticket"
    assert trace.verification["passed"] is True
    assert "TICKET1001" in trace.final_response
    assert "RESOLVED" in trace.final_response


def test_scenario_unknown_order(orchestrator):
    """Scenario 6: Unknown order ORD9999."""
    trace = orchestrator.run_sync("Where is order ORD9999?")
    assert trace.verification["order_found"] is False
    assert "could not be found" in trace.final_response
    # Grounding check: never invent fake delivery dates or tracking
    assert "2026-" not in trace.final_response


def test_scenario_ambiguous_request(orchestrator):
    """Scenario 7: Ambiguous request triggers clarification question."""
    trace = orchestrator.run_sync("I need help with my order.")
    assert trace.needs_clarification is True
    assert "ORD1001" in trace.final_response
    assert len(trace.tool_calls) == 0


def test_scenario_general_faq(orchestrator):
    """Scenario 8: Return policy FAQ inquiry."""
    trace = orchestrator.run_sync("What is your return policy?")
    assert trace.intent == "faq_policy"
    assert "30 calendar days" in trace.final_response
    assert "Final Sale" in trace.final_response


def test_scenario_multi_step_request(orchestrator):
    """Scenario 9: Multi-step composite request combining 3 tools."""
    trace = orchestrator.run_sync("Check order ORD1001, tell me its delivery status, and check whether it can be returned.")
    assert trace.intent == "multi_step_order_inquiry"
    assert len(trace.tool_calls) == 3
    executed_tools = [t["tool"] for t in trace.tool_calls]
    assert "get_order" in executed_tools
    assert "check_delivery_status" in executed_tools
    assert "check_return_eligibility" in executed_tools
    assert trace.verification["passed"] is True
    assert "Delivery Status" in trace.final_response
    assert "Return Eligibility" in trace.final_response
