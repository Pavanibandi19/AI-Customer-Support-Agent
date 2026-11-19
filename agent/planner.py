"""Planner module for Agentic Customer Support Agent.

Performs Request Understanding (intent classification, entity extraction)
and Planning (tool selection, sequencing, and parameter validation).
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ToolStep:
    """Represents a planned tool invocation."""
    tool_name: str
    arguments: Dict[str, Any]
    description: str = ""
    is_resource: bool = False
    resource_uri: Optional[str] = None


@dataclass
class Plan:
    """Represents the agent's execution plan."""
    intent: str
    extracted_entities: Dict[str, Any]
    steps: List[ToolStep] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str = ""
    reasoning: str = ""


class AgentPlanner:
    """Understands user intent and constructs validated tool execution plans."""

    def __init__(self, discovered_tools: Optional[List[str]] = None):
        self.discovered_tools = set(discovered_tools or [
            "get_order",
            "get_customer_orders",
            "check_delivery_status",
            "check_return_eligibility",
            "get_return_policy",
            "create_support_ticket",
            "get_support_ticket",
        ])

    def detect_unsupported_action(self, q_lower: str) -> Optional[str]:
        """Detect action requests that are not supported through the customer support chat agent."""
        # 1. Delivery agent phone number or live GPS location
        if any(k in q_lower for k in [
            "phone number of the delivery agent", "delivery agent phone number",
            "phone number of the driver", "driver phone number", "driver number", "contact the driver",
            "phone number of my delivery agent", "delivery driver phone", "driver's phone",
            "exact location of my delivery agent", "exact location of the delivery agent",
            "location of my delivery agent", "location of the delivery agent", "where is the driver",
            "live location", "real-time location", "gps location", "real time location",
        ]):
            return "delivery_agent_contact"

        # 2. Shipping label generation
        if any(k in q_lower for k in [
            "shipping label", "return label", "return shipping label",
            "generate the return label", "generate a return label", "generate return label",
            "send me the return shipping label", "send me a return shipping label",
            "generate a shipping label", "generate shipping label", "send me the shipping label",
            "create a return label", "create return label", "print return label", "print shipping label",
        ]) and any(w in q_lower for w in ["send", "generate", "create", "print", "provide", "email", "mail", "give", "can you"]):
            return "shipping_label_generation"

        # 3. Direct Refund processing
        if any(k in q_lower for k in [
            "process my refund", "process the refund", "process refund", "process a refund",
            "issue my refund", "issue the refund", "issue refund", "issue a refund",
            "refund right now", "refund immediately", "send my refund", "give me my refund right now",
            "execute refund", "credit my refund right now",
        ]):
            return "refund_processing"

        # 4. Delivery address modification
        if any(k in q_lower for k in [
            "change the delivery address", "change delivery address", "change my address", "change address",
            "change the address", "change shipping address",
            "update the delivery address", "update delivery address", "update my address", "update address",
            "update the address", "update shipping address",
            "modify the delivery address", "modify delivery address", "modify my address", "modify address",
            "modify the address", "modify shipping address",
        ]) or bool(re.search(r"\b(change|update|modify)\b.*\baddress\b", q_lower)):
            return "address_modification"

        # 5. Order cancellation
        if (any(k in q_lower for k in [
            "cancel my order", "cancel the order", "cancel order", "cancel ord", "cancel",
        ]) or bool(re.search(r"\bcancel\b.*\b(order|ord\d+|package)\b", q_lower))) and not any(k in q_lower for k in ["cancellation policy", "can i cancel", "how to cancel"]):
            return "order_cancellation"

        return None

    def detect_all_unsupported_actions(
        self, query: str, entities: Dict[str, Any], memory: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect all unsupported action requests with their associated order IDs."""
        q_lower = query.lower()
        # Split into clauses by comma, semicolon, "and ", "also ", newline
        raw_clauses = re.split(r"[,;\n]|\band\b|\balso\b", q_lower)
        clauses = [c.strip() for c in raw_clauses if c.strip()]

        actions = []
        seen_types = set()

        for clause in clauses:
            action_type = None
            if any(k in clause for k in [
                "phone number of the delivery agent", "delivery agent phone number",
                "phone number of the driver", "driver phone number", "driver number", "contact the driver",
                "phone number of my delivery agent", "delivery driver phone", "driver's phone",
                "exact location of my delivery agent", "exact location of the delivery agent",
                "location of my delivery agent", "location of the delivery agent", "where is the driver",
                "live location", "real-time location", "gps location", "real time location",
            ]):
                action_type = "delivery_agent_contact"

            elif any(k in clause for k in [
                "shipping label", "return label", "return shipping label",
                "generate the return label", "generate a return label", "generate return label",
                "send me the return shipping label", "send me a return shipping label",
                "generate a shipping label", "generate shipping label", "send me the shipping label",
                "create a return label", "create return label", "print return label", "print shipping label",
            ]) and any(w in clause for w in ["send", "generate", "create", "print", "provide", "email", "mail", "give", "can you"]):
                action_type = "shipping_label_generation"

            elif any(k in clause for k in [
                "process my refund", "process the refund", "process refund", "process a refund",
                "issue my refund", "issue the refund", "issue refund", "issue a refund",
                "refund right now", "refund immediately", "send my refund", "give me my refund right now",
                "execute refund", "credit my refund right now",
            ]):
                action_type = "refund_processing"

            elif any(k in clause for k in [
                "change the delivery address", "change delivery address", "change my address", "change address",
                "change the address", "change shipping address",
                "update the delivery address", "update delivery address", "update my address", "update address",
                "update the address", "update shipping address",
                "modify the delivery address", "modify delivery address", "modify my address", "modify address",
                "modify the address", "modify shipping address",
            ]) or bool(re.search(r"\b(change|update|modify)\b.*\baddress\b", clause)):
                action_type = "address_modification"

            elif (any(k in clause for k in [
                "cancel my order", "cancel the order", "cancel order", "cancel ord", "cancel",
            ]) or bool(re.search(r"\bcancel\b.*\b(order|ord\d+|package)\b", clause))) and not any(k in clause for k in ["cancellation policy", "can i cancel", "how to cancel"]):
                action_type = "order_cancellation"

            if action_type and action_type not in seen_types:
                m_ord = re.search(r"\bORD[-_]?(\d{4})\b", clause, re.IGNORECASE)
                clause_oid = f"ORD{m_ord.group(1)}" if m_ord else None
                actions.append({
                    "action_type": action_type,
                    "order_id": clause_oid,
                    "clause": clause,
                })
                seen_types.add(action_type)

        # Assign unassigned order_ids from entities if any action is missing order_id
        all_query_oids = list(entities.get("order_ids", []))
        assigned_oids = {a["order_id"] for a in actions if a.get("order_id")}
        unassigned_oids = [o for o in all_query_oids if o not in assigned_oids]
        for a in actions:
            if not a.get("order_id") and unassigned_oids:
                a["order_id"] = unassigned_oids.pop(0)

        # Fallback: if no clause matched but whole query matches detect_unsupported_action
        if not actions:
            single = self.detect_unsupported_action(q_lower)
            if single:
                target_oid = entities.get("order_id") or (memory["recent_orders"][0] if memory.get("recent_orders") else None)
                actions.append({
                    "action_type": single,
                    "order_id": target_oid,
                    "clause": q_lower,
                })

        # Ensure order_ids are populated if missing in single-action scenario
        if len(actions) == 1 and not actions[0]["order_id"]:
            target_oid = entities.get("order_id") or (memory["recent_orders"][0] if memory.get("recent_orders") else None)
            actions[0]["order_id"] = target_oid

        return actions

    def is_privacy_violation_query(self, q_lower: str) -> bool:
        """Detect requests attempting bulk dump or unauthorized disclosure of all customer private information."""
        dump_phrases = [
            "every customer", "all customers", "entire customer database", "all users",
            "every user", "entire database", "dump the database", "dump all",
            "private information of all", "passwords", "credit card", "payment credentials",
            "customer details of every", "details of every customer", "list every customer",
            "all customer records", "all customer details", "all customer information",
            "information about every customer", "details of all customers",
        ]
        return any(k in q_lower for k in dump_phrases)

    def is_ticket_creation_query(self, q_lower: str) -> bool:
        """Check if query expresses intent to create/raise a support ticket or file/open a complaint."""
        ticket_phrases = [
            "create a ticket", "create ticket", "create support ticket",
            "raise a ticket", "raise ticket", "raise a support ticket",
            "need to raise a ticket", "want to raise a ticket", "have to raise a ticket",
            "open a ticket", "file a ticket", "file ticket", "log a ticket", "issue ticket",
            "open a complaint", "open complaint", "file a complaint", "raise a complaint",
            "submit a complaint", "make a complaint", "lodge a complaint", "file complaint",
            "damaged", "broken", "defective",
        ]
        if any(p in q_lower for p in ticket_phrases):
            return True
        if re.search(r"\b(raise|create|open|file|log|submit|make|lodge)\b.*\b(ticket|complaint)\b", q_lower):
            return True
        if re.search(r"\b(ticket|complaint)\s+for\b", q_lower):
            return True
        return False

    def extract_context_from_history(
        self, conversation_history: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Extract recent entities, orders, customer info, and ticket context from conversation history."""
        if not conversation_history:
            return {
                "recent_orders": [],
                "recent_tickets": [],
                "recent_customers": [],
                "turns_orders": [],
                "turns_tickets": [],
                "turns_customers": [],
                "order_details": {},
                "created_tickets_by_order": {},
                "pending_ticket_request": None,
                "last_issue_description": None,
            }

        turns_orders: List[List[str]] = []
        turns_tickets: List[List[str]] = []
        turns_customers: List[List[str]] = []
        order_details: Dict[str, Dict[str, Any]] = {}
        all_orders: List[str] = []
        all_tickets: List[str] = []
        all_customers: List[str] = []
        created_tickets_by_order: Dict[str, str] = {}
        pending_ticket_request: Optional[Dict[str, Any]] = None
        last_issue_description: Optional[str] = None
        damaged_orders: List[str] = []

        # Chronological scan to track created tickets, pending ticket requests, and issue descriptions
        for msg in conversation_history:
            role = msg.get("role")
            content = msg.get("content", "")
            trace = msg.get("trace")

            if trace:
                trace_dict = trace
                if hasattr(trace, "__dataclass_fields__"):
                    from dataclasses import asdict
                    trace_dict = asdict(trace)
                if isinstance(trace_dict, dict):
                    # Check tool calls
                    for t in trace_dict.get("tool_calls", []):
                        if t.get("tool") == "create_support_ticket" and t.get("success"):
                            out = t.get("output", {})
                            args = t.get("arguments", {})
                            t_id = out.get("ticket_id")
                            o_id = out.get("order_id") or args.get("order_id")
                            if t_id and o_id:
                                created_tickets_by_order[str(o_id).strip().upper()] = str(t_id).strip().upper()
                    # Check verification
                    verif = trace_dict.get("verification", {})
                    if verif.get("ticket_created") and verif.get("created_ticket_id"):
                        t_id = verif.get("created_ticket_id")
                        facts = trace_dict.get("grounded_facts", {})
                        ents = trace_dict.get("extracted_entities", {})
                        o_id = facts.get("order_id") or ents.get("order_id")
                        if t_id and o_id:
                            created_tickets_by_order[str(o_id).strip().upper()] = str(t_id).strip().upper()

            if role == "assistant" and isinstance(content, str):
                m_t = re.search(r"\b(TICKET\d{4})\b", content, re.IGNORECASE)
                m_o = re.search(r"\b(ORD\d{4})\b", content, re.IGNORECASE)
                if m_t and m_o:
                    t_str = m_t.group(1).upper()
                    o_str = m_o.group(1).upper()
                    c_lower = content.lower()
                    if "ticket" in c_lower and any(w in c_lower for w in ["created", "support ticket", "official", "already", "file"]):
                        created_tickets_by_order[o_str] = t_str

            if role == "user" and isinstance(content, str):
                u_lower = content.lower()
                is_ticket_req = self.is_ticket_creation_query(u_lower)
                if is_ticket_req:
                    m_o = re.search(r"\bORD[-_]?(\d{4})\b", content, re.IGNORECASE)
                    req_oid = f"ORD{m_o.group(1)}" if m_o else None
                    m_c = re.search(r"\bCUST[-_]?(\d{4})\b", content, re.IGNORECASE)
                    req_cid = f"CUST{m_c.group(1)}" if m_c else None

                    specific_issue_keywords = [
                        "damaged", "broken", "defective", "faulty", "cracked", "shattered", "torn", "scratched",
                        "missing", "wrong item", "wrong product", "incorrect item", "not working", "doesn't work",
                        "does not work", "malfunction", "failed", "late delivery", "not delivered", "haven't received",
                        "have not received", "never arrived", "arrived late", "poor quality", "leaking", "stained",
                    ]
                    if any(k in u_lower for k in specific_issue_keywords):
                        last_issue_description = content.strip()
                        pending_ticket_request = {
                            "order_id": req_oid,
                            "customer_id": req_cid,
                            "issue": "Damaged / Defective item reported" if any(k in u_lower for k in ["damaged", "broken", "defective"]) else "Customer reported issue",
                            "description": last_issue_description,
                        }

                if any(k in u_lower for k in ["not received", "have not received", "haven't received", "never received", "did not receive"]):
                    m_o = re.search(r"\bORD[-_]?(\d{4})\b", content, re.IGNORECASE)
                    req_oid = f"ORD{m_o.group(1)}" if m_o else None
                    last_issue_description = "The package was marked delivered but customer reports not receiving it"
                    pending_ticket_request = {
                        "order_id": req_oid,
                        "customer_id": None,
                        "issue": "Delivery non-receipt / Missing package",
                        "description": last_issue_description,
                    }

                if any(k in u_lower for k in ["damaged", "broken", "defective", "faulty"]):
                    for m in re.finditer(r"\bORD[-_]?(\d{4})\b", content, re.IGNORECASE):
                        d_oid = f"ORD{m.group(1)}"
                        if d_oid not in damaged_orders:
                            damaged_orders.append(d_oid)

            if pending_ticket_request and pending_ticket_request.get("order_id"):
                p_oid = pending_ticket_request["order_id"]
                if p_oid in created_tickets_by_order:
                    pending_ticket_request = None

        # Iterate in reverse chronological order (newest turns first)
        for msg in reversed(conversation_history):
            turn_orders: List[str] = []
            turn_tickets: List[str] = []
            turn_customers: List[str] = []

            # Skip initial greeting or assistant messages without execution trace
            if msg.get("role") == "assistant" and not msg.get("trace"):
                continue

            # 1. Inspect structured execution trace if present
            trace = msg.get("trace")
            if trace:
                trace_dict = trace
                if hasattr(trace, "__dataclass_fields__"):
                    from dataclasses import asdict
                    trace_dict = asdict(trace)

                if isinstance(trace_dict, dict):
                    # Check tool calls
                    for t in trace_dict.get("tool_calls", []):
                        args = t.get("arguments", {})
                        out = t.get("output", {})
                        if args.get("order_id"):
                            oid = str(args["order_id"]).strip().upper()
                            if oid not in turn_orders:
                                turn_orders.append(oid)
                        if args.get("ticket_id"):
                            tid = str(args["ticket_id"]).strip().upper()
                            if tid not in turn_tickets:
                                turn_tickets.append(tid)
                        if args.get("customer_id") and args["customer_id"] != "LOOKUP_FROM_ORDER":
                            cid = str(args["customer_id"]).strip().upper()
                            if cid not in turn_customers:
                                turn_customers.append(cid)

                        if isinstance(out, dict):
                            if "order" in out and isinstance(out["order"], dict):
                                o = out["order"]
                                oid = o.get("order_id")
                                if oid:
                                    oid = str(oid).strip().upper()
                                    if oid not in turn_orders:
                                        turn_orders.append(oid)
                                    order_details[oid] = {
                                        "order_id": oid,
                                        "customer_id": o.get("customer_id"),
                                        "customer_name": o.get("customer_name"),
                                        "item_name": o.get("item_name"),
                                        "total_amount": o.get("total_amount"),
                                        "status": o.get("status"),
                                    }
                                    if o.get("customer_id"):
                                        cid = str(o["customer_id"]).strip().upper()
                                        if cid not in turn_customers:
                                            turn_customers.append(cid)
                            if "ticket_id" in out and out["ticket_id"]:
                                tid = str(out["ticket_id"]).strip().upper()
                                if tid not in turn_tickets:
                                    turn_tickets.append(tid)
                            if "ticket" in out and isinstance(out["ticket"], dict):
                                tk = out["ticket"]
                                tid = tk.get("ticket_id")
                                if tid:
                                    tid = str(tid).strip().upper()
                                    if tid not in turn_tickets:
                                        turn_tickets.append(tid)
                                if tk.get("order_id"):
                                    oid = str(tk["order_id"]).strip().upper()
                                    if oid not in turn_orders:
                                        turn_orders.append(oid)
                                if tk.get("customer_id"):
                                    cid = str(tk["customer_id"]).strip().upper()
                                    if cid not in turn_customers:
                                        turn_customers.append(cid)
                            if "orders" in out and isinstance(out["orders"], list):
                                for o in out["orders"]:
                                    if isinstance(o, dict) and o.get("order_id"):
                                        oid = str(o["order_id"]).strip().upper()
                                        if oid not in turn_orders:
                                            turn_orders.append(oid)
                                        order_details[oid] = {
                                            "order_id": oid,
                                            "customer_id": out.get("customer_id"),
                                            "item_name": o.get("item_name"),
                                            "total_amount": o.get("total_amount"),
                                            "status": o.get("status"),
                                        }

            # 2. Inspect message text content
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                for m in re.finditer(r"\bORD[-_]?(\d{4})\b", content, re.IGNORECASE):
                    oid = f"ORD{m.group(1)}"
                    if oid not in turn_orders:
                        turn_orders.append(oid)
                for m in re.finditer(r"\bTICKET[-_]?(\d{4})\b", content, re.IGNORECASE):
                    tid = f"TICKET{m.group(1)}"
                    if tid not in turn_tickets:
                        turn_tickets.append(tid)
                for m in re.finditer(r"\bCUST[-_]?(\d{4})\b", content, re.IGNORECASE):
                    cid = f"CUST{m.group(1)}"
                    if cid not in turn_customers:
                        turn_customers.append(cid)

            if turn_orders:
                turns_orders.append(turn_orders)
                for o in turn_orders:
                    if o not in all_orders:
                        all_orders.append(o)
            if turn_tickets:
                turns_tickets.append(turn_tickets)
                for t in turn_tickets:
                    if t not in all_tickets:
                        all_tickets.append(t)
            if turn_customers:
                turns_customers.append(turn_customers)
                for c in turn_customers:
                    if c not in all_customers:
                        all_customers.append(c)

        return {
            "recent_orders": all_orders,
            "recent_tickets": all_tickets,
            "recent_customers": all_customers,
            "turns_orders": turns_orders,
            "turns_tickets": turns_tickets,
            "turns_customers": turns_customers,
            "order_details": order_details,
            "created_tickets_by_order": created_tickets_by_order,
            "pending_ticket_request": pending_ticket_request,
            "last_issue_description": last_issue_description,
            "damaged_orders": damaged_orders,
        }

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract Order, Customer, and Ticket identifiers using standard patterns."""
        entities = {}

        # Order IDs: e.g. ORD1001, ORD-1001, ORD9999
        order_ids = []
        for m in re.finditer(r"\bORD[-_]?(\d{4})\b", query, re.IGNORECASE):
            oid = f"ORD{m.group(1)}"
            if oid not in order_ids:
                order_ids.append(oid)
        if order_ids:
            entities["order_ids"] = order_ids
            entities["order_id"] = order_ids[0]

        # Customer IDs: e.g. CUST1001, CUST-1001
        cust_ids = []
        for m in re.finditer(r"\bCUST[-_]?(\d{4})\b", query, re.IGNORECASE):
            cid = f"CUST{m.group(1)}"
            if cid not in cust_ids:
                cust_ids.append(cid)
        if cust_ids:
            entities["customer_ids"] = cust_ids
            entities["customer_id"] = cust_ids[0]

        # Ticket IDs: e.g. TICKET1001, TICKET-1001
        ticket_ids = []
        for m in re.finditer(r"\bTICKET[-_]?(\d{4})\b", query, re.IGNORECASE):
            tid = f"TICKET{m.group(1)}"
            if tid not in ticket_ids:
                ticket_ids.append(tid)
        if ticket_ids:
            entities["ticket_ids"] = ticket_ids
            entities["ticket_id"] = ticket_ids[0]

        return entities

    def is_general_policy_query(self, q_lower: str) -> bool:
        """Identify if query is a general policy or FAQ inquiry."""
        # 1. Direct policy/FAQ terms and differences
        if any(k in q_lower for k in [
            "return policy", "refund policy", "cancellation policy", "store policy", "company policy",
            "return rules", "refund rules", "how do returns work", "how do refunds work",
            "how does return work", "how does a return work", "how does refund work",
            "faq", "frequently asked", "policy on returns", "policy for returns",
            "difference between return policy", "difference between policy",
            "policy vs return eligibility", "policy vs eligibility", "policy and eligibility",
            "policy versus eligibility",
        ]):
            return True

        # 2. Refund processing, timeline, approval, and refund duration
        if (
            any(k in q_lower for k in [
                "how long does a refund take", "how long does the refund process take",
                "how long does a refund", "how long does the refund", "refund take",
                "refund process", "refund processing", "how long for a refund",
                "when will i receive my refund", "when will i get a refund", "when do i get my refund",
                "when will i get my refund", "when will i get my refud", "when do i get my refud",
                "what happens after a return is approved", "after a return is approved",
                "return is approved", "approved return", "return approved",
            ])
            or bool(re.search(r"\b(?:when|how long|how many days|in how many days|n how many days)\b.*?\b(?:refund|refud)\b", q_lower))
            or bool(re.search(r"\brefuds?\b", q_lower))
        ):
            return True

        # 3. Return window start, duration questions, deadlines, and missed window
        if (any(k in q_lower for k in [
            "when does the return period begin", "when does the return window begin",
            "when does return period begin", "when does return window begin",
            "when does the return period start", "when does the return window start",
            "when does return period start", "when does return window start",
            "return period begin", "return window begin", "return period start", "return window start",
            "how long do i have to return", "how long to return",
            "return window", "return period", "time limit to return", "time limit for return",
            "after 30 days", "past 30 days", "within 30 days", "30-day", "30 days",
            "how many days to return", "days do i have to return", "return deadline",
            "miss the 30-day", "miss the return deadline", "miss the deadline", "if i miss",
            "missed the", "miss the return window",
        ]) or (
            any(k in q_lower for k in ["how many days", "how long"])
            and not bool(re.search(r"\b(?:refund|refud)\b", q_lower))
        ) or bool(re.search(r"\b(after|past|beyond|within)\s+\d+\s+days\b", q_lower))) and any(k in q_lower for k in ["return", "refund", "refud", "send back", "exchange", "product", "item", "it", "deadline", "period", "window"]):
            return True

        # 4. Packaging, item condition, used products, and accessory requirements
        if any(k in q_lower for k in [
            "packaging required", "original packaging", "original box", "tags attached",
            "opened packaging", "opened the packaging", "open packaging", "open the packaging",
            "opened package", "opened the package", "open package", "open the package",
            "condition for return", "conditions for return", "conditions for returning",
            "condition to return", "conditions to return", "return condition", "return requirements",
            "what are the conditions", "requirements for return", "requirements to return",
            "what condition", "condition should", "condition must", "condition of the item",
            "item condition", "product condition", "condition should a returned item",
            "can i return a used", "can you return a used", "return a used product",
            "return a used item", "used product", "used products", "used item", "used items",
            "used goods", "already used", "unopened", "unworn", "undamaged", "accessory", "accessories",
        ]):
            return True

        # 5. Clearance, final sale, exceptions, non-returnable items, hygiene products
        if any(k in q_lower for k in [
            "clearance", "final sale", "non-returnable", "cannot be returned", "can't be returned",
            "non returnable", "not returnable", "exceptions", "hygiene", "hygienic", "personal care",
            "intimate apparel",
        ]) and any(k in q_lower for k in ["return", "returnable", "refund", "item", "product", "products", "order", "policy", "sale"]):
            return True

        # 6. General return questions about a generic item/product/category (not a specific order)
        if any(k in q_lower for k in [
            "how to return", "how do i return", "can i return a product", "can a product be returned",
            "can items be returned", "can i return an item", "can an item be returned",
            "can you return a product", "how does someone return",
            "eligible for return", "eligibility for return",
            "electronic", "electronics",
        ]):
            return True

        # 7. Orders in transit / not yet delivered
        if any(k in q_lower for k in [
            "in transit", "currently in transit", "not yet delivered", "before delivery",
            "on the way",
        ]) and any(k in q_lower for k in ["return", "eligible", "eligibility", "cancel", "refund"]):
            return True

        # 8. Refund and return combination general questions
        if "refund" in q_lower and "return" in q_lower and not any(k in q_lower for k in ["ord", "order"]):
            return True

        # 9. Generic returnability questions (e.g. "Are ... returnable?", "Is ... returnable?")
        if re.search(r"\b(are|is|can)\b.*\breturnable\b", q_lower) and not any(k in q_lower for k in ["ord", "order"]):
            return True

        # 10. Promotional bonuses, cashback, pickup service, and immediate refunds
        if any(k in q_lower for k in [
            "cashback", "cash back", "bonus", "cashback bonus",
            "pickup", "pick up", "pickup service", "free return pickup", "return pickup",
            "two hours", "2 hours", "within two hours", "within 2 hours", "instant refund",
        ]):
            return True

        return False

    def plan(
        self,
        query: str,
        customer_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Plan:
        """Analyze query and formulate a deterministic, validated plan.

        Args:
            query: User's natural language input.
            customer_context: Optional session customer context (e.g. authenticated customer).
            conversation_history: Optional list of past conversation turns for memory resolution.

        Returns:
            Structured Plan object.
        """
        entities = self.extract_entities(query)
        if customer_context and "customer_id" in customer_context and "customer_id" not in entities:
            entities["customer_id"] = customer_context["customer_id"]
            if "customer_ids" not in entities:
                entities["customer_ids"] = [customer_context["customer_id"]]

        q_lower = query.lower().strip()
        memory = self.extract_context_from_history(conversation_history)

        # 1. Customer Privacy Violation (Bulk data / credential dump)
        if self.is_privacy_violation_query(q_lower):
            steps = []
            for oid in entities.get("order_ids", []):
                steps.append(ToolStep(
                    tool_name="get_order",
                    arguments={"order_id": oid},
                    description=f"Look up order {oid}",
                ))
            for tid in entities.get("ticket_ids", []):
                steps.append(ToolStep(
                    tool_name="get_support_ticket",
                    arguments={"ticket_id": tid},
                    description=f"Look up support ticket {tid}",
                ))
            return Plan(
                intent="privacy_restricted",
                extracted_entities=entities,
                steps=steps,
                reasoning="Bulk customer account details or private information cannot be disclosed. Individual record inquiries are checked separately.",
            )

        # 2. Cross-Customer Authorization Check
        auth_cid = customer_context.get("customer_id").strip().upper() if customer_context and customer_context.get("customer_id") else None
        requested_cids = entities.get("customer_ids", [])
        foreign_cids = [cid for cid in requested_cids if auth_cid and cid != auth_cid]
        is_unauthorized_req = (
            bool(foreign_cids)
            or any(k in q_lower for k in [
                "without authorization", "unauthorized access",
                "another customer's orders without authorization", "another customer's order without authorization",
                "other customer's orders without authorization", "information about another customer",
            ])
        )
        if is_unauthorized_req:
            target_cid = foreign_cids[0] if foreign_cids else (requested_cids[0] if requested_cids else "another customer")
            return Plan(
                intent="unauthorized_customer_access",
                extracted_entities=entities,
                reasoning=f"Access denied: unauthorized access attempt for customer {target_cid}.",
            )

        # 3. Unsupported Actions
        unsupported_list = self.detect_all_unsupported_actions(query, entities, memory)
        if len(unsupported_list) > 1:
            entities["unsupported_actions"] = unsupported_list
            return Plan(
                intent="multiple_unsupported_actions",
                extracted_entities=entities,
                steps=[],
                reasoning=f"Identified multiple unsupported actions ({len(unsupported_list)}) across requested orders.",
            )
        elif len(unsupported_list) == 1:
            single = unsupported_list[0]
            target_oid = single.get("order_id") or entities.get("order_id") or (memory["recent_orders"][0] if memory.get("recent_orders") else None)
            if target_oid:
                entities["order_id"] = target_oid
            steps = []
            if single["action_type"] == "delivery_agent_contact" and target_oid:
                steps.append(ToolStep(
                    tool_name="check_delivery_status",
                    arguments={"order_id": target_oid},
                    description=f"Check delivery details for {target_oid}",
                ))
            return Plan(
                intent=f"unsupported_{single['action_type']}",
                extracted_entities=entities,
                steps=steps,
                reasoning=f"Identified unsupported action request '{single['action_type']}' for order {target_oid or 'unspecified'}.",
            )

        # Reference detection patterns
        has_order_ref = bool(re.search(r"\b(that order|this order|the order|that package|this package|the package)\b", q_lower))
        has_product_ref = bool(re.search(r"\b(that product|this product|the product|that item|this item|the item)\b", q_lower))
        has_ticket_ref = bool(re.search(r"\b(that ticket|this ticket|the ticket|my ticket)\b", q_lower))
        has_customer_ref = bool(re.search(r"\b(that customer|this customer|the customer|my customer|that user|this user|the user|their orders|their account)\b", q_lower))
        is_relative_that = bool(re.search(r"\b(product|products|item|items|order|orders|package|packages|anything|something)\s+that\b", q_lower))
        has_pronoun = bool(re.search(r"\b(it|its)\b", q_lower)) or (bool(re.search(r"\b(that|this)\b", q_lower)) and not has_customer_ref and not is_relative_that)
        is_generic_subject = (
            bool(re.search(r"\b(a|an|any)\s+([a-z0-9_-]+\s+)?(product|item|order|package)\b", q_lower))
            or bool(re.search(r"\b(products|items)\b", q_lower))
        )
        is_affirmative_followup = (
            any(q_lower.startswith(w) for w in ["yes", "yeah", "sure", "yep", "ok", "okay", "please"])
            or "i need a refund" in q_lower or "i want a refund" in q_lower or "need a refund" in q_lower or "want a refund" in q_lower
        )
        is_process_followup = any(k in q_lower for k in [
            "what happens next", "what next", "how does it work next", "what do i do next",
            "what should i do next", "what is the next step", "next steps",
        ])

        # Ambiguous order help requests without an explicit order ID
        is_ambiguous_help = (
            not entities.get("order_id")
            and not entities.get("customer_id")
            and not entities.get("ticket_id")
            and any(phrase in q_lower for phrase in [
                "i need help with my order",
                "i need help with an order",
                "i need help with the order",
                "i need help with that order",
                "help with my order",
                "help me with my order",
                "help with an order",
                "need help with my order",
                "need help with an order",
                "help with order",
                "i have an issue with my order",
                "i have a problem with my order",
                "when will my package arrive",
                "when will my order arrive",
            ])
            and not any(k in q_lower for k in [
                "damaged", "broken", "defective", "return", "refund", "cancel", "complaint", "status", "where is", "where's", "tracking", "track"
            ])
        )
        if is_ambiguous_help:
            return Plan(
                intent="ambiguous_order_request",
                extracted_entities=entities,
                needs_clarification=True,
                clarification_question="I would be happy to help with your order! Could you please share your Order ID (for example, ORD1001) so I can look up the details?",
                reasoning="Customer requested help with an order but did not provide an Order ID.",
            )

        # Check for ambiguity: "what is its status" when BOTH order and ticket exist in recent context
        if (
            has_pronoun
            and "status" in q_lower
            and not has_order_ref
            and not has_ticket_ref
            and not entities.get("order_id")
            and not entities.get("ticket_id")
            and memory["recent_orders"]
            and memory["recent_tickets"]
        ):
            latest_order = memory["recent_orders"][0]
            latest_ticket = memory["recent_tickets"][0]
            return Plan(
                intent="ambiguous_status_reference",
                extracted_entities=entities,
                needs_clarification=True,
                clarification_question=f"Are you inquiring about the status of order {latest_order} or support ticket {latest_ticket}?",
                reasoning="Ambiguous reference 'its status' with both an order and ticket in context.",
            )

        is_creating_ticket = self.is_ticket_creation_query(q_lower)

        # Resolve Ticket references (only when inquiring about an existing ticket, not creating one)
        if not entities.get("ticket_id") and not is_creating_ticket and (has_ticket_ref or ("ticket" in q_lower and has_pronoun)):
            if not memory["recent_tickets"]:
                return Plan(
                    intent="ambiguous_ticket_request",
                    extracted_entities=entities,
                    needs_clarification=True,
                    clarification_question="Could you please provide your Support Ticket ID (for example, TICKET1001) so I can check its status for you?",
                    reasoning="Customer referenced a ticket but no ticket was found in conversation history.",
                )
            latest_tickets = memory["turns_tickets"][0] if memory["turns_tickets"] else []
            if len(latest_tickets) > 1:
                return Plan(
                    intent="ambiguous_ticket_reference",
                    extracted_entities=entities,
                    needs_clarification=True,
                    clarification_question=f"You recently discussed multiple support tickets ({', '.join(latest_tickets)}). Which ticket would you like me to check?",
                    reasoning=f"Ambiguous ticket reference with multiple candidate tickets: {latest_tickets}.",
                )
            entities["ticket_id"] = memory["recent_tickets"][0]

        # Resolve Customer references
        if not entities.get("customer_id") and not entities.get("order_id") and has_customer_ref:
            if not memory["recent_customers"]:
                return Plan(
                    intent="ambiguous_customer_request",
                    extracted_entities=entities,
                    needs_clarification=True,
                    clarification_question="Could you please provide your Customer ID (for example, CUST1001) so I can look up the account details?",
                    reasoning="Customer referenced 'this customer' but no Customer ID was found in conversation history.",
                )
            latest_customers = memory["turns_customers"][0] if memory["turns_customers"] else []
            if len(latest_customers) > 1:
                return Plan(
                    intent="ambiguous_customer_reference",
                    extracted_entities=entities,
                    needs_clarification=True,
                    clarification_question=f"You recently discussed multiple customers ({', '.join(latest_customers)}). Which customer would you like me to check?",
                    reasoning=f"Ambiguous customer reference with multiple candidate customers: {latest_customers}.",
                )
            entities["customer_id"] = memory["recent_customers"][0]

        # Customer Inquiry (Name, Profile, or Orders) or Resumption of Pending Ticket Request
        if entities.get("customer_id") and not entities.get("order_id") and not entities.get("ticket_id"):
            is_explicit_customer_query = any(k in q_lower for k in [
                "name", "who is", "profile", "show orders", "all orders", "compare",
                "list orders", "what orders", "past orders", "order history",
                "show all orders", "belonging to", "belonging", "placed", "order", "orders",
            ])
            pending_ticket = memory.get("pending_ticket_request")
            if (
                pending_ticket
                and pending_ticket.get("order_id")
                and pending_ticket["order_id"] not in memory.get("created_tickets_by_order", {})
                and not is_explicit_customer_query
            ):
                target_order = pending_ticket["order_id"]
                cust_id = entities["customer_id"]
                entities["order_id"] = target_order
                desc = pending_ticket.get("description") or memory.get("last_issue_description") or "The order arrived damaged"
                return Plan(
                    intent="damaged_item_support",
                    extracted_entities=entities,
                    steps=[
                        ToolStep(
                            tool_name="get_order",
                            arguments={"order_id": target_order},
                            description=f"Verify order {target_order} exists and obtain customer ID",
                        ),
                        ToolStep(
                            tool_name="create_support_ticket",
                            arguments={
                                "customer_id": cust_id,
                                "order_id": target_order,
                                "issue": "Damaged / Defective item reported",
                                "description": desc,
                            },
                            description=f"Create persistent support ticket for damaged order {target_order}",
                        ),
                    ],
                    reasoning=f"Resuming pending support ticket request for order {target_order} with provided customer ID {cust_id}.",
                )

            if is_explicit_customer_query or not self.is_general_policy_query(q_lower):
                cust_ids = entities.get("customer_ids", [entities["customer_id"]])
                steps = [
                    ToolStep(
                        tool_name="get_customer_orders",
                        arguments={"customer_id": cid},
                        description=f"Fetch profile and order history for customer {cid}",
                    )
                    for cid in cust_ids
                ]
                reason = (
                    f"Identified customer inquiry for customers: {', '.join(cust_ids)}."
                    if len(cust_ids) > 1
                    else f"Identified customer inquiry for customer {cust_ids[0]}."
                )
                return Plan(
                    intent="customer_orders",
                    extracted_entities=entities,
                    steps=steps,
                    reasoning=reason,
                )

        # Check if query is a general policy inquiry without an explicit order ID in the query
        is_policy = self.is_general_policy_query(q_lower)
        if not entities.get("order_id") and is_policy:
            # Contextual policy check: only treat as contextual_policy if the user explicitly refers
            # to a previously discussed order/product using a direct reference (e.g. 'it', 'its', 'that order',
            # 'the product') AND the inquiry is not framed as a general/generic inquiry ('a product', 'an item', etc.).
            is_explicit_order_followup = (
                (bool(re.search(r"\b(it|its)\b", q_lower)) or has_order_ref or has_product_ref)
                and not is_generic_subject
                and not any(phrase in q_lower for phrase in [
                    "opened hygiene", "hygiene", "clearance", "final sale", "used product", "used item",
                    "opened packaging", "opened the packaging", "open packaging", "open the packaging",
                    "opened package", "opened the package", "open package", "open the package",
                    "what condition", "condition should", "condition must",
                    "how long does a refund", "how long does the refund", "refund process",
                    "what happens after a return is approved", "after a return is approved",
                    "when does the return period begin", "when does the return window begin",
                    "when does the return period start", "when does the return window start",
                ])
                and not bool(re.search(r"\b(after|past|beyond)\s+\d+\s+days\b", q_lower))
            )
            if is_explicit_order_followup and memory["recent_orders"]:
                resolved_order = memory["recent_orders"][0]
                entities["order_id"] = resolved_order
                if resolved_order in memory["order_details"]:
                    details = memory["order_details"][resolved_order]
                    if details.get("item_name"):
                        entities["item_name"] = details["item_name"]
                    if details.get("customer_id") and "customer_id" not in entities:
                        entities["customer_id"] = details["customer_id"]
                return Plan(
                    intent="contextual_policy",
                    extracted_entities=entities,
                    steps=[
                        ToolStep(
                            tool_name="get_order",
                            arguments={"order_id": resolved_order},
                            description=f"Retrieve order details for {resolved_order}",
                        ),
                        ToolStep(
                            tool_name="get_return_policy",
                            arguments={},
                            description="Retrieve company return policy guidelines",
                            is_resource=True,
                            resource_uri="support://policies/returns",
                        ),
                    ],
                    reasoning=f"Customer asked a policy question regarding contextual order {resolved_order}.",
                )
            else:
                # Pure general policy inquiry (no order in context, or asking generally about 'a product'/'an item')
                return Plan(
                    intent="faq_policy",
                    extracted_entities=entities,
                    steps=[
                        ToolStep(
                            tool_name="get_return_policy",
                            arguments={},
                            description="Retrieve company return policy guidelines",
                            is_resource=True,
                            resource_uri="support://policies/returns",
                        )
                    ],
                    reasoning="Customer asked general policy/FAQ question without referencing an order ID.",
                )

        # Resolve Order / Product references
        order_action_cues = [
            "track", "tracking", "delivery", "deliver", "delivered", "delivering",
            "arrive", "arrived", "arriving", "arrival", "ship", "shipped", "shipping", "shipment",
            "where is", "where's", "where", "when", "sent", "send", "package", "return", "returned",
            "returning", "refund", "refunded", "damaged", "broken", "defective", "order", "item",
            "product", "status", "cancel", "cost", "price", "bought", "purchased", "receive", "received",
        ]
        is_order_followup = (
            has_order_ref
            or has_product_ref
            or (has_pronoun and any(c in q_lower for c in order_action_cues))
            or any(k in q_lower for k in ["what product", "what is the product", "which product", "what item", "what is the item", "tell me about the product", "what did i order"])
            or is_affirmative_followup
            or is_process_followup
            or (is_creating_ticket and bool(memory["recent_orders"]))
        )

        # Only resolve order if query didn't already have an explicit order_id and is not a ticket query
        if not entities.get("order_id") and is_order_followup and not entities.get("ticket_id"):
            # Check general FAQ first: e.g. "What is your return policy?"
            is_policy_query = (
                any(k in q_lower for k in [
                    "return policy", "how to return", "refund policy", "return window",
                    "what is your return", "can items be returned", "faq", "frequently asked",
                    "refunds and returns", "returns work", "how do returns work",
                ])
                or ("refund" in q_lower and "return" in q_lower)
            )
            if not (is_policy_query and not (has_order_ref or has_product_ref or has_pronoun)):
                if not memory["recent_orders"]:
                    return Plan(
                        intent="ambiguous_order_request",
                        extracted_entities=entities,
                        needs_clarification=True,
                        clarification_question="I would be happy to help with your order! Could you please share your Order ID (for example, ORD1001) so I can look up the details?",
                        reasoning="Customer used a follow-up reference ('it'/'that order') but no Order ID was found in conversation history.",
                    )
                if is_creating_ticket and memory.get("damaged_orders"):
                    if len(memory["damaged_orders"]) == 1:
                        resolved_order = memory["damaged_orders"][0]
                    else:
                        return Plan(
                            intent="ambiguous_order_reference",
                            extracted_entities=entities,
                            needs_clarification=True,
                            clarification_question=f"You recently discussed multiple damaged orders ({', '.join(memory['damaged_orders'])}). Which order would you like me to create a support ticket for?",
                            reasoning=f"Ambiguous ticket request with multiple damaged orders: {memory['damaged_orders']}.",
                        )
                else:
                    latest_orders = memory["turns_orders"][0] if memory["turns_orders"] else []
                    if len(latest_orders) > 1:
                        clarify_action = "create a support ticket for" if is_creating_ticket else "check"
                        return Plan(
                            intent="ambiguous_order_reference",
                            extracted_entities=entities,
                            needs_clarification=True,
                            clarification_question=f"You recently discussed multiple orders ({', '.join(latest_orders)}). Which order would you like me to {clarify_action}?",
                            reasoning=f"Ambiguous follow-up reference with multiple candidate orders: {latest_orders}.",
                        )
                    resolved_order = memory["recent_orders"][0]
                entities["order_id"] = resolved_order
                if resolved_order in memory["order_details"]:
                    details = memory["order_details"][resolved_order]
                    if "customer_id" not in entities and details.get("customer_id"):
                        entities["customer_id"] = details["customer_id"]
                    if details.get("item_name"):
                        entities["item_name"] = details["item_name"]
                elif memory["recent_customers"] and "customer_id" not in entities:
                    entities["customer_id"] = memory["recent_customers"][0]

        # Preserve customer context if known from order details
        if entities.get("order_id") and "customer_id" not in entities:
            oid = entities["order_id"]
            if oid in memory["order_details"] and memory["order_details"][oid].get("customer_id"):
                entities["customer_id"] = memory["order_details"][oid]["customer_id"]

        # Process / Next Steps Follow-Up (e.g. "What happens next?")
        if is_process_followup and entities.get("order_id"):
            oid = entities["order_id"]
            return Plan(
                intent="contextual_policy",
                extracted_entities=entities,
                steps=[
                    ToolStep(
                        tool_name="get_order",
                        arguments={"order_id": oid},
                        description=f"Retrieve order details for {oid}",
                    ),
                    ToolStep(
                        tool_name="get_return_policy",
                        arguments={},
                        description="Retrieve company return policy guidelines",
                        is_resource=True,
                        resource_uri="support://policies/returns",
                    ),
                ],
                reasoning=f"Customer inquired about next steps for order {oid}.",
            )

        # Product Inquiry (e.g. "What is the product?", "Tell me about the product")
        is_product_inquiry = any(k in q_lower for k in [
            "what product", "what is the product", "which product", "what item",
            "what is the item", "which item", "tell me about the product", "what did i order",
        ]) and not any(k in q_lower for k in ["return", "refund", "damaged", "broken", "defective", "ticket"])
        if is_product_inquiry and entities.get("order_id"):
            order_ids = entities.get("order_ids", [entities["order_id"]])
            if len(order_ids) > 1:
                return Plan(
                    intent="product_inquiry",
                    extracted_entities=entities,
                    steps=[
                        ToolStep(
                            tool_name="get_order",
                            arguments={"order_id": oid},
                            description=f"Retrieve product details for order {oid}",
                        )
                        for oid in order_ids
                    ],
                    reasoning=f"Customer inquired about products in multiple orders: {', '.join(order_ids)}.",
                )
            return Plan(
                intent="product_inquiry",
                extracted_entities=entities,
                steps=[
                    ToolStep(
                        tool_name="get_order",
                        arguments={"order_id": entities["order_id"]},
                        description=f"Retrieve product details for order {entities['order_id']}",
                    ),
                ],
                reasoning=f"Customer inquired about the product/item in order {entities['order_id']}.",
            )

        # 1. Existing Support Ticket Inquiry
        if entities.get("ticket_id"):
            ticket_id = entities["ticket_id"]
            return Plan(
                intent="get_ticket",
                extracted_entities=entities,
                steps=[
                    ToolStep(
                        tool_name="get_support_ticket",
                        arguments={"ticket_id": ticket_id},
                        description=f"Look up existing support ticket {ticket_id}",
                    )
                ],
                reasoning=f"Identified inquiry regarding support ticket {ticket_id}.",
            )

        # 2. General Return Policy or FAQ Inquiry (No Order ID provided)
        is_policy_query = self.is_general_policy_query(q_lower)
        if is_policy_query and not entities.get("order_id"):
            return Plan(
                intent="faq_policy",
                extracted_entities=entities,
                steps=[
                    ToolStep(
                        tool_name="get_return_policy",
                        arguments={},
                        description="Retrieve company return policy guidelines",
                        is_resource=True,
                        resource_uri="support://policies/returns",
                    )
                ],
                reasoning="Customer asked general policy/FAQ question without referencing an order ID.",
            )

        # 4. Ambiguous Request without Order ID
        order_action_keywords = [
            "order", "track", "tracking", "delivery", "deliver", "delivered", "arrive", "arrived",
            "ship", "shipped", "where is", "where's", "package", "return", "refund", "damaged", "broken", "ticket",
            "complaint", "raise", "file",
        ]
        if not entities.get("order_id") and (any(k in q_lower for k in order_action_keywords) or self.is_ticket_creation_query(q_lower)):
            # Check if user wanted to create a ticket or report damage
            if self.is_ticket_creation_query(q_lower):
                return Plan(
                    intent="ambiguous_order_request",
                    extracted_entities=entities,
                    needs_clarification=True,
                    clarification_question="I would be happy to help file a support ticket! Could you please share your Order ID (for example, ORD1001) so I can look up your order details?",
                    reasoning="Customer requested to create a support ticket or report damage but did not provide an Order ID.",
                )
            # Check if user mentioned ticket without ID
            if "ticket" in q_lower:
                return Plan(
                    intent="ambiguous_ticket_request",
                    extracted_entities=entities,
                    needs_clarification=True,
                    clarification_question="Could you please provide your Support Ticket ID (for example, TICKET1001) so I can check its status for you?",
                    reasoning="Customer inquired about a ticket but did not provide a ticket ID.",
                )
            return Plan(
                intent="ambiguous_order_request",
                extracted_entities=entities,
                needs_clarification=True,
                clarification_question="I would be happy to help with your order! Could you please share your Order ID (for example, ORD1001) so I can look up the details?",
                reasoning="Customer inquired about an order or delivery but did not provide an Order ID.",
            )

        # Guard against missing order_id
        if not entities.get("order_id"):
            return Plan(
                intent="general_assistance",
                extracted_entities=entities,
                needs_clarification=True,
                clarification_question="Hello! How may I assist you today? You can ask me to track an order (e.g. 'Where is order ORD1001?'), check delivery status, review return eligibility, or inquire about return policies.",
                reasoning="General inquiry without an order, ticket, or customer identifier.",
            )

        order_id = entities["order_id"]
        order_ids = entities.get("order_ids", [order_id])

        # Multiple Orders Inquiry or Comparison
        if len(order_ids) > 1:
            # A. Return eligibility inquiry / comparison
            is_return_query = any(k in q_lower for k in [
                "return", "eligible", "eligibility", "send back", "exchange", "refund"
            ])
            if is_return_query:
                return Plan(
                    intent="return_eligibility",
                    extracted_entities=entities,
                    steps=[
                        ToolStep(
                            tool_name="check_return_eligibility",
                            arguments={"order_id": oid},
                            description=f"Evaluate return window and policy rules for {oid}",
                        )
                        for oid in order_ids
                    ],
                    reasoning=f"Compare return eligibility for multiple orders: {', '.join(order_ids)}.",
                )

            # B. Delivery status / tracking / general comparison for multiple orders
            return Plan(
                intent="delivery_followup",
                extracted_entities=entities,
                steps=[
                    ToolStep(
                        tool_name="check_delivery_status",
                        arguments={"order_id": oid},
                        description=f"Check carrier delivery status for {oid}",
                    )
                    for oid in order_ids
                ],
                reasoning=f"Check delivery status for multiple orders: {', '.join(order_ids)}.",
            )

        # 5. Damaged Item / Ticket Creation Request
        is_damaged_or_issue = self.is_ticket_creation_query(q_lower)
        if is_damaged_or_issue:
            # Single-creation guarantee: Ticket creation must occur exactly once
            if order_id in memory.get("created_tickets_by_order", {}):
                existing_ticket_id = memory["created_tickets_by_order"][order_id]
                entities["ticket_id"] = existing_ticket_id
                return Plan(
                    intent="existing_ticket_confirmed",
                    extracted_entities=entities,
                    steps=[
                        ToolStep(
                            tool_name="get_support_ticket",
                            arguments={"ticket_id": existing_ticket_id},
                            description=f"Retrieve existing support ticket {existing_ticket_id} for order {order_id}",
                        )
                    ],
                    reasoning=f"A support ticket ({existing_ticket_id}) has already been created for order {order_id} in this conversation. Confirming existing ticket without duplicate creation.",
                )

            # Check if specific issue is actually described in the query or conversation history
            specific_issue_keywords = [
                "damaged", "broken", "defective", "faulty", "cracked", "shattered", "torn", "scratched",
                "missing", "wrong item", "wrong product", "incorrect item", "not working", "doesn't work",
                "does not work", "malfunction", "failed", "late delivery", "not delivered", "haven't received",
                "have not received", "never arrived", "arrived late", "poor quality", "leaking", "stained",
            ]
            has_explicit_issue = any(k in q_lower for k in specific_issue_keywords)
            has_history_issue = bool(memory.get("last_issue_description")) or (
                bool(memory.get("pending_ticket_request")) and bool(memory["pending_ticket_request"].get("description"))
            )

            # Determine issue description and category
            if any(k in q_lower for k in ["damaged", "broken", "defective", "faulty", "cracked", "shattered", "torn", "scratched"]):
                issue_category = "Damaged / Defective item reported"
                desc = query if any(k in q_lower for k in ["damaged", "broken", "defective"]) else "The order arrived damaged"
            elif any(k in q_lower for k in ["missing", "not delivered", "haven't received", "have not received", "never arrived"]):
                issue_category = "Delivery non-receipt / Missing package"
                desc = query
            elif any(k in q_lower for k in ["wrong item", "wrong product", "incorrect item"]):
                issue_category = "Incorrect item received"
                desc = query
            elif any(k in q_lower for k in ["not working", "doesn't work", "does not work", "malfunction", "failed"]):
                issue_category = "Product malfunctioning / not working"
                desc = query
            elif memory.get("pending_ticket_request") and memory["pending_ticket_request"].get("issue"):
                issue_category = memory["pending_ticket_request"]["issue"]
                desc = memory["pending_ticket_request"].get("description") or query
            elif memory.get("last_issue_description"):
                issue_category = "Customer reported issue"
                desc = memory["last_issue_description"]
            else:
                issue_category = "Customer reported issue"
                desc = query

            # Customer ID resolution: check query, memory, or mark for order lookup
            raw_cids = []
            for m in re.finditer(r"\bCUST[-_]?(\d{4})\b", query, re.IGNORECASE):
                raw_cids.append(f"CUST{m.group(1)}")

            if raw_cids:
                cid = raw_cids[0]
            elif order_id in memory.get("order_details", {}) and memory["order_details"][order_id].get("customer_id"):
                cid = memory["order_details"][order_id]["customer_id"]
            else:
                cid = "LOOKUP_FROM_ORDER"
            entities["customer_id"] = cid

            # We first retrieve the order to verify customer and item details, then create ticket, and verify creation if requested
            ticket_steps = [
                ToolStep(
                    tool_name="get_order",
                    arguments={"order_id": order_id},
                    description=f"Verify order {order_id} exists and obtain customer ID",
                ),
                ToolStep(
                    tool_name="create_support_ticket",
                    arguments={
                        "customer_id": cid,
                        "order_id": order_id,
                        "issue": issue_category,
                        "description": desc,
                    },
                    description=f"Create persistent support ticket for order {order_id}",
                ),
            ]
            if any(k in q_lower for k in ["check", "verify", "details", "status", "show", "confirm"]):
                ticket_steps.append(
                    ToolStep(
                        tool_name="get_support_ticket",
                        arguments={"ticket_id": "LOOKUP_FROM_CREATED_TICKET"},
                        description=f"Verify created support ticket for order {order_id}",
                    )
                )

            return Plan(
                intent="damaged_item_support",
                extracted_entities=entities,
                steps=ticket_steps,
                reasoning=f"Customer reported a damaged item or requested a support ticket for order {order_id}. Plan retrieves order details first, creates ticket, and verifies creation.",
            )

        # 6. Multi-Step Request (Check order + delivery + return)
        multi_step_cues = [
            ("delivery" in q_lower or "track" in q_lower or "status" in q_lower) and ("return" in q_lower or "eligible" in q_lower),
            "and" in q_lower and ("check" in q_lower or "tell me" in q_lower),
        ]
        if any(multi_step_cues):
            return Plan(
                intent="multi_step_order_inquiry",
                extracted_entities=entities,
                steps=[
                    ToolStep(
                        tool_name="get_order",
                        arguments={"order_id": order_id},
                        description=f"Retrieve core order details for {order_id}",
                    ),
                    ToolStep(
                        tool_name="check_delivery_status",
                        arguments={"order_id": order_id},
                        description=f"Check carrier delivery status for {order_id}",
                    ),
                    ToolStep(
                        tool_name="check_return_eligibility",
                        arguments={"order_id": order_id},
                        description=f"Verify return policy eligibility for {order_id}",
                    ),
                ],
                reasoning=f"Identified composite request requiring order verification, delivery tracking, and return eligibility evaluation for {order_id}.",
            )

        # 7. Return Eligibility Request
        is_return_query = any(k in q_lower for k in ["return", "eligible", "send back", "exchange", "refund"])
        if is_return_query:
            return Plan(
                intent="return_eligibility",
                extracted_entities=entities,
                steps=[
                    ToolStep(
                        tool_name="get_order",
                        arguments={"order_id": order_id},
                        description=f"Retrieve order details for {order_id}",
                    ),
                    ToolStep(
                        tool_name="check_return_eligibility",
                        arguments={"order_id": order_id},
                        description=f"Evaluate return window and policy rules for {order_id}",
                    ),
                ],
                reasoning=f"Customer asked if order {order_id} can be returned.",
            )

        # 8. Delivery Follow-Up (Expected delivery date / has not arrived / delivery status inquiry)
        is_delivery_followup = any(k in q_lower for k in [
            "not arrived", "hasn't arrived", "when will it be delivered", "when will it arrive",
            "when was it delivered", "when did it arrive", "when was it shipped", "when did it ship",
            "was it delivered", "did it arrive", "has it arrived", "has it been delivered",
            "is it delivered", "delivery date", "estimated delivery", "actual delivery",
            "where is my package", "delayed", "late", "delivery status",
        ]) or (
            any(v in q_lower for v in ["deliver", "arrive", "ship"])
            and any(w in q_lower for w in ["when", "date", "time", "status"])
        )
        if is_delivery_followup:
            return Plan(
                intent="delivery_followup",
                extracted_entities=entities,
                steps=[
                    ToolStep(
                        tool_name="get_order",
                        arguments={"order_id": order_id},
                        description=f"Fetch order status for {order_id}",
                    ),
                    ToolStep(
                        tool_name="check_delivery_status",
                        arguments={"order_id": order_id},
                        description=f"Check tracking and estimated delivery date for {order_id}",
                    ),
                ],
                reasoning=f"Customer inquiring about delayed arrival or expected delivery date for order {order_id}.",
            )

        # 9. General Order Tracking
        return Plan(
            intent="order_tracking",
            extracted_entities=entities,
            steps=[
                ToolStep(
                    tool_name="get_order",
                    arguments={"order_id": order_id},
                    description=f"Retrieve order and tracking information for {order_id}",
                ),
                ToolStep(
                    tool_name="check_delivery_status",
                    arguments={"order_id": order_id},
                    description=f"Fetch live tracking details for {order_id}",
                ),
            ],
            reasoning=f"Standard order tracking inquiry for {order_id}.",
        )

    def validate_tool_name(self, tool_name: str) -> bool:
        """Ensure tool exists in discovered MCP tools."""
        return tool_name in self.discovered_tools
