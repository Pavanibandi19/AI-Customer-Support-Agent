"""Reflection and Verification module for AI Customer Support Agent.

Validates tool execution outcomes against grounding rules, detects missing
or contradictory evidence, and ensures the agent never reports hallucinated
orders, delivery dates, or unverified ticket creations.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Outcome of the reflection and verification stage."""
    passed: bool
    findings: List[str] = field(default_factory=list)
    grounded_facts: Dict[str, Any] = field(default_factory=dict)
    order_found: Optional[bool] = None
    ticket_created: Optional[bool] = None
    created_ticket_id: Optional[str] = None
    is_return_eligible: Optional[bool] = None
    reason: str = ""


class AgentReflection:
    """Evaluates tool results and enforces grounding rules."""

    def verify_tool_results(
        self,
        intent: str,
        planned_steps: List[Any],
        executed_results: List[Dict[str, Any]],
    ) -> VerificationResult:
        """Inspect all executed tool outputs and verify factual grounding.

        Args:
            intent: The recognized user intent.
            planned_steps: Steps that were planned.
            executed_results: List of dicts with 'tool', 'arguments', 'output', 'success'.

        Returns:
            VerificationResult detailing verification status and grounded facts.
        """
        findings = []
        grounded_facts = {}
        all_passed = True
        order_found = None
        ticket_created = None
        created_ticket_id = None
        is_return_eligible = None

        for exec_item in executed_results:
            tool_name = exec_item.get("tool")
            args = exec_item.get("arguments", {})
            output = exec_item.get("output", {})
            tool_call_ok = exec_item.get("success", True)

            # If tool threw an unhandled exception
            if not tool_call_ok:
                all_passed = False
                findings.append(f"Tool '{tool_name}' communication error: {output.get('error', 'Unknown')}")
                continue

            # 1. Verification of get_order
            if tool_name == "get_order":
                if output.get("success") is True and "order" in output:
                    order = output["order"]
                    order_found = True
                    grounded_facts["order_id"] = order["order_id"]
                    grounded_facts["customer_id"] = order["customer_id"]
                    grounded_facts["customer_name"] = order.get("customer_name")
                    grounded_facts["item_name"] = order["item_name"]
                    grounded_facts["order_status"] = order["status"]
                    grounded_facts["order_date"] = order["order_date"]
                    grounded_facts["total_amount"] = order["total_amount"]
                    findings.append(f"Verified order '{order['order_id']}' exists with status '{order['status']}'.")
                else:
                    order_found = False
                    all_passed = False
                    order_id = args.get("order_id", "Unknown")
                    findings.append(f"Order '{order_id}' was not found in the database.")

            # 2. Verification of check_delivery_status
            elif tool_name == "check_delivery_status":
                if output.get("success") is True:
                    grounded_facts["carrier"] = output.get("carrier")
                    grounded_facts["tracking_number"] = output.get("tracking_number")
                    grounded_facts["delivery_status"] = output.get("delivery_status")
                    grounded_facts["estimated_delivery_date"] = output.get("estimated_delivery_date")
                    grounded_facts["actual_delivery_date"] = output.get("actual_delivery_date")
                    findings.append(
                        f"Verified delivery status: {output.get('delivery_status')} "
                        f"(Carrier: {output.get('carrier')}, Est: {output.get('estimated_delivery_date')})."
                    )
                else:
                    findings.append(f"Delivery status inquiry returned: {output.get('error', 'Not found')}")

            # 3. Verification of check_return_eligibility
            elif tool_name == "check_return_eligibility":
                if output.get("success") is True:
                    is_return_eligible = output.get("is_eligible")
                    grounded_facts["is_return_eligible"] = is_return_eligible
                    grounded_facts["return_reason"] = output.get("reason", "")
                    grounded_facts["return_window_days"] = output.get("return_window_days")
                    findings.append(f"Verified return eligibility: {is_return_eligible}. Reason: {output.get('reason')}")
                else:
                    all_passed = False
                    findings.append(f"Return eligibility check failed: {output.get('error')}")

            # 4. Verification of create_support_ticket
            elif tool_name == "create_support_ticket":
                if output.get("success") is True and output.get("ticket_id"):
                    ticket_created = True
                    created_ticket_id = output["ticket_id"]
                    grounded_facts["created_ticket_id"] = created_ticket_id
                    grounded_facts["ticket_status"] = output.get("status", "open")
                    findings.append(f"Verified support ticket {created_ticket_id} was successfully persisted to SQLite database.")
                else:
                    ticket_created = False
                    all_passed = False
                    findings.append(f"CRITICAL: Support ticket creation failed or was rejected: {output.get('error')}")

            # 5. Verification of get_support_ticket
            elif tool_name == "get_support_ticket":
                if output.get("success") is True and "ticket" in output:
                    t = output["ticket"]
                    grounded_facts["ticket_id"] = t["ticket_id"]
                    grounded_facts["ticket_status"] = t["status"]
                    grounded_facts["ticket_issue"] = t["issue"]
                    findings.append(f"Verified ticket {t['ticket_id']} exists with status '{t['status']}'.")
                else:
                    all_passed = False
                    t_id = args.get("ticket_id", "Unknown")
                    findings.append(f"Ticket '{t_id}' was not found in database.")

            # 6. Verification of get_return_policy
            elif tool_name == "get_return_policy":
                if output.get("success") is True:
                    policy_name = output.get("policy_name", "Standard Customer Return & Refund Policy")
                    policy_window = output.get("return_window_days", 30)
                    grounded_facts["policy_name"] = policy_name
                    grounded_facts["policy_window"] = policy_window
                    if output.get("content"):
                        grounded_facts["policy_content"] = output["content"]
                    findings.append(f"Verified company policy ({policy_window}-day return window) from support://policies/returns.")

            # 7. Verification of get_customer_orders
            elif tool_name == "get_customer_orders":
                if output.get("success") is True:
                    grounded_facts["customer_id"] = output.get("customer_id")
                    grounded_facts["customer_name"] = output.get("customer_name")
                    grounded_facts["membership_tier"] = output.get("membership_tier")
                    grounded_facts["order_count"] = output.get("order_count")
                    grounded_facts["orders"] = output.get("orders", [])
                    findings.append(f"Verified customer '{output.get('customer_id')}' ({output.get('customer_name')}) with {output.get('order_count')} orders.")
                else:
                    all_passed = False
                    cust_id = args.get("customer_id", "Unknown")
                    findings.append(f"Customer '{cust_id}' was not found in database: {output.get('error', 'Not found')}")

        return VerificationResult(
            passed=all_passed,
            findings=findings,
            grounded_facts=grounded_facts,
            order_found=order_found,
            ticket_created=ticket_created,
            created_ticket_id=created_ticket_id,
            is_return_eligible=is_return_eligible,
            reason="; ".join(findings),
        )
