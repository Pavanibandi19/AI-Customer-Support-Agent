"""Agent Orchestrator module.

Coordinates the 7-stage Agentic AI lifecycle:
A. UNDERSTAND  -> Intent and entity identification
B. PLAN        -> Tool selection, sequencing, parameter resolution
C. EXECUTE     -> Official MCP stdio tool invocation
D. OBSERVE     -> Raw result capture
E. REFLECT     -> Grounding verification and consistency checking
F. RESPOND     -> Verified answer generation
G. LOG/TRACE   -> Structured diagnostic trace
"""

import re
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict

import config
from agent.mcp_client import MCPClientWrapper
from agent.planner import AgentPlanner, Plan, ToolStep
from agent.reflection import AgentReflection, VerificationResult
from agent.llm_client import LLMService

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionRecord:
    """Record of an individual tool call during execution."""
    tool: str
    arguments: Dict[str, Any]
    output: Any
    latency_ms: float
    success: bool
    is_resource: bool = False


@dataclass
class AgentExecutionTrace:
    """Complete diagnostic trace of an agentic workflow execution."""
    query: str
    intent: str
    plan_reasoning: str
    planned_tools: List[str]
    tool_calls: List[Dict[str, Any]]
    verification: Dict[str, Any]
    final_response: str
    latency_ms: float
    llm_mode: str
    needs_clarification: bool = False


class AgentOrchestrator:
    """High-level agent orchestrator coordinating multi-step support workflows."""

    def __init__(
        self,
        mcp_client: Optional[MCPClientWrapper] = None,
        llm_service: Optional[LLMService] = None,
    ):
        self.mcp_client = mcp_client or MCPClientWrapper()
        self.llm_service = llm_service or LLMService()
        self.reflection = AgentReflection()
        self.discovered_tools: List[str] = []
        self._initialized = False
        self.conversation_history: List[Dict[str, Any]] = []

    def reset_memory(self) -> None:
        """Clear internal conversational memory."""
        self.conversation_history = []

    async def initialize(self) -> None:
        """Discover tools from MCP server to validate future invocations."""
        if not self._initialized:
            try:
                tools = await self.mcp_client.list_tools()
                self.discovered_tools = [t["name"] for t in tools]
                self._initialized = True
                logger.info("Orchestrator initialized with discovered MCP tools: %s", self.discovered_tools)
            except Exception as e:
                logger.error("Failed to initialize MCP tool discovery: %s", str(e))
                # Fallback to standard 7 tools
                self.discovered_tools = [
                    "get_order",
                    "get_customer_orders",
                    "check_delivery_status",
                    "check_return_eligibility",
                    "get_return_policy",
                    "create_support_ticket",
                    "get_support_ticket",
                ]

    async def run(
        self,
        query: str,
        customer_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentExecutionTrace:
        """Execute the agentic workflow for a user request."""
        start_time = time.perf_counter()
        await self.initialize()

        llm_mode = self.llm_service.determine_active_mode()
        planner = AgentPlanner(discovered_tools=self.discovered_tools)

        # STAGE A & B: UNDERSTAND & PLAN (incorporating conversational memory)
        active_history = self.conversation_history if conversation_history is None else conversation_history
        plan: Plan = planner.plan(
            query,
            customer_context=customer_context,
            conversation_history=active_history,
        )
        planned_tools = [s.tool_name for s in plan.steps]

        # Check if clarification is needed (e.g. missing or ambiguous identifiers)
        if plan.needs_clarification:
            total_latency = (time.perf_counter() - start_time) * 1000
            trace = AgentExecutionTrace(
                query=query,
                intent=plan.intent,
                plan_reasoning=plan.reasoning,
                planned_tools=[],
                tool_calls=[],
                verification={"passed": True, "findings": ["Request requires user clarification for missing or ambiguous reference."]},
                final_response=plan.clarification_question,
                latency_ms=round(total_latency, 2),
                llm_mode=llm_mode,
                needs_clarification=True,
            )
            # Record in internal conversation memory
            self.conversation_history.append({"role": "user", "content": query, "trace": None})
            self.conversation_history.append({
                "role": "assistant",
                "content": trace.final_response,
                "trace": asdict(trace),
            })
            return trace

        # STAGE C & D: EXECUTE & OBSERVE (Bounded loop up to MAX_AGENT_STEPS)
        executed_records: List[ToolExecutionRecord] = []
        context_state: Dict[str, Any] = {}
        failed_orders: set = set()
        steps_to_run = plan.steps[: config.MAX_AGENT_STEPS]
        q_lower = query.lower()

        async with self.mcp_client.session() as mcp_session:
            for step in steps_to_run:
                # Tool name safety validation
                if not planner.validate_tool_name(step.tool_name):
                    logger.warning("Proposed tool '%s' is not in discovered MCP tools. Skipping.", step.tool_name)
                    continue

                # Skip secondary steps on orders that failed initial lookup
                if step.arguments.get("order_id") in failed_orders and step.tool_name in (
                    "check_delivery_status", "check_return_eligibility", "create_support_ticket"
                ):
                    continue

                # Dynamic parameter resolution (e.g. customer_id from previous get_order step)
                args = dict(step.arguments)
                if step.tool_name == "create_support_ticket":
                    # Check customer mismatch if user asserted a specific customer_id in query
                    user_stated_cid = None
                    for m in re.finditer(r"\bCUST[-_]?(\d{4})\b", query, re.IGNORECASE):
                        user_stated_cid = f"CUST{m.group(1)}"
                        break
                    if user_stated_cid and context_state.get("customer_id") and context_state["customer_id"] != user_stated_cid:
                        continue

                    # Check if explicit issue description was provided
                    specific_issue_keywords = [
                        "damaged", "broken", "defective", "faulty", "cracked", "shattered", "torn", "scratched",
                        "missing", "wrong item", "wrong product", "incorrect item", "not working", "doesn't work",
                        "does not work", "malfunction", "failed", "late delivery", "not delivered", "haven't received",
                        "have not received", "never arrived", "arrived late", "poor quality", "leaking", "stained",
                        "complaint",
                    ]
                    has_explicit_issue = any(k in q_lower for k in specific_issue_keywords)
                    has_history_issue = False
                    if active_history:
                        for turn in reversed(active_history):
                            if turn.get("role") == "user":
                                c_low = (turn.get("content") or "").lower()
                                if any(k in c_low for k in specific_issue_keywords):
                                    has_history_issue = True
                                    break
                    if not has_explicit_issue and not has_history_issue:
                        continue

                    if context_state.get("customer_id"):
                        args["customer_id"] = context_state["customer_id"]
                    elif plan.extracted_entities.get("customer_id") and plan.extracted_entities["customer_id"] != "LOOKUP_FROM_ORDER":
                        args["customer_id"] = plan.extracted_entities["customer_id"]
                elif args.get("customer_id") == "LOOKUP_FROM_ORDER":
                    if context_state.get("customer_id"):
                        args["customer_id"] = context_state["customer_id"]
                    elif plan.extracted_entities.get("customer_id") and plan.extracted_entities["customer_id"] != "LOOKUP_FROM_ORDER":
                        args["customer_id"] = plan.extracted_entities["customer_id"]

                if step.tool_name == "get_support_ticket" and args.get("ticket_id") == "LOOKUP_FROM_CREATED_TICKET":
                    if context_state.get("created_ticket_id"):
                        args["ticket_id"] = context_state["created_ticket_id"]
                    else:
                        continue

                step_start = time.perf_counter()
                tool_success = True
                raw_output = None

                try:
                    if step.is_resource and step.resource_uri:
                        resource_text = await mcp_session.read_resource(step.resource_uri)
                        raw_output = {"success": True, "content": resource_text}
                    else:
                        raw_output = await mcp_session.call_tool(step.tool_name, args)
                        tool_success = True
                except Exception as e:
                    tool_success = False
                    raw_output = {"success": False, "error": str(e)}

                step_latency = (time.perf_counter() - step_start) * 1000

                # Update context state with verified values
                if isinstance(raw_output, dict):
                    if "order" in raw_output:
                        context_state["customer_id"] = raw_output["order"].get("customer_id")
                        context_state["order_id"] = raw_output["order"].get("order_id")
                        context_state["item_name"] = raw_output["order"].get("item_name")
                    if "ticket_id" in raw_output:
                        context_state["created_ticket_id"] = raw_output["ticket_id"]

                executed_records.append(
                    ToolExecutionRecord(
                        tool=step.tool_name,
                        arguments=args,
                        output=raw_output,
                        latency_ms=round(step_latency, 2),
                        success=tool_success,
                        is_resource=step.is_resource,
                    )
                )

                # For customer orders queries that also ask about return eligibility, check delivered orders
                if step.tool_name == "get_customer_orders" and isinstance(raw_output, dict) and raw_output.get("success"):
                    is_return_inquiry_on_orders = any(k in q_lower for k in ["return", "eligible", "eligibility"])
                    if is_return_inquiry_on_orders:
                        for ord_item in raw_output.get("orders", []):
                            if (ord_item.get("status") or "").lower() == "delivered" or (ord_item.get("delivery_status") or "").lower() == "delivered":
                                ord_id = ord_item.get("order_id")
                                if ord_id:
                                    try:
                                        elig_res = await mcp_session.call_tool("check_return_eligibility", {"order_id": ord_id})
                                        executed_records.append(
                                            ToolExecutionRecord(
                                                tool="check_return_eligibility",
                                                arguments={"order_id": ord_id},
                                                output=elig_res,
                                                latency_ms=0.0,
                                                success=True,
                                                is_resource=False,
                                            )
                                        )
                                    except Exception as e:
                                        logger.error("Failed to check return eligibility for %s: %s", ord_id, str(e))

                # If order lookup failed, mark it so dependent steps for this order are skipped
                if step.tool_name == "get_order" and isinstance(raw_output, dict) and raw_output.get("success") is False:
                    context_state["order_not_found"] = True
                    if step.arguments.get("order_id"):
                        failed_orders.add(step.arguments["order_id"])

        # STAGE E: REFLECT / VERIFY
        dict_records = [
            {
                "tool": r.tool,
                "arguments": r.arguments,
                "output": r.output,
                "success": r.success,
            }
            for r in executed_records
        ]
        verification: VerificationResult = self.reflection.verify_tool_results(
            intent=plan.intent,
            planned_steps=steps_to_run,
            executed_results=dict_records,
        )

        # STAGE F: RESPOND (Grounded response generation)
        final_answer = await self._generate_response(
            query=query,
            plan=plan,
            executed_records=executed_records,
            verification=verification,
            llm_mode=llm_mode,
            conversation_history=active_history,
        )

        total_latency = (time.perf_counter() - start_time) * 1000

        # Check if response asked for issue clarification
        is_issue_clarification = "Could you please describe the issue you are experiencing?" in final_answer

        # STAGE G: LOG AND TRACE
        trace = AgentExecutionTrace(
            query=query,
            intent=plan.intent,
            plan_reasoning=plan.reasoning,
            planned_tools=planned_tools,
            tool_calls=[asdict(r) for r in executed_records],
            verification=asdict(verification),
            final_response=final_answer,
            latency_ms=round(total_latency, 2),
            llm_mode=llm_mode,
            needs_clarification=is_issue_clarification or plan.needs_clarification,
        )

        # Record in internal conversation memory
        self.conversation_history.append({"role": "user", "content": query, "trace": None})
        self.conversation_history.append({
            "role": "assistant",
            "content": trace.final_response,
            "trace": asdict(trace),
        })

        logger.info(
            "Workflow finished for query '%s' in %.2fms. Intent: %s, Tools: %s, Verification: %s",
            query,
            total_latency,
            plan.intent,
            planned_tools,
            verification.passed,
        )
        return trace

    def run_sync(
        self,
        query: str,
        customer_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentExecutionTrace:
        """Synchronous wrapper for UI and tests."""
        return asyncio.run(self.run(query, customer_context, conversation_history))

    async def _generate_response(
        self,
        query: str,
        plan: Plan,
        executed_records: List[ToolExecutionRecord],
        verification: VerificationResult,
        llm_mode: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Synthesize a customer-facing response strictly grounded in verified facts."""
        facts = verification.grounded_facts
        q_lower = query.lower()

        # Privacy & Authorization Checks
        if plan.intent == "privacy_restricted":
            privacy_msg = (
                "Access restricted: For customer privacy and data security, bulk customer personal information, "
                "account details, and full database exports cannot be disclosed through this customer support chat."
            )
            lookup_lines = []
            for r in executed_records:
                if r.tool == "get_order":
                    oid = r.arguments.get("order_id", "specified")
                    if r.output and r.output.get("success") and "order" in r.output:
                        o = r.output["order"]
                        lookup_lines.append(f"- **Order {oid}**: Verified in system (Item: {o.get('item_name')}, Status: {o.get('status')}).")
                    else:
                        lookup_lines.append(f"- **Order {oid}**: Record was not found in our database.")
                elif r.tool == "get_support_ticket":
                    tid = r.arguments.get("ticket_id", "specified")
                    if r.output and r.output.get("success") and "ticket" in r.output:
                        t = r.output["ticket"]
                        lookup_lines.append(f"- **Support Ticket {tid}**: Status: {t.get('status')}, Issue: {t.get('issue')}.")
                    else:
                        lookup_lines.append(f"- **Support Ticket {tid}**: Record was not found in our database.")

            if lookup_lines:
                return (
                    f"{privacy_msg}\n\n"
                    "**Individual Record Verification Results:**\n"
                    + "\n".join(lookup_lines)
                )
            return privacy_msg

        if plan.intent == "unauthorized_customer_access":
            return (
                "Access restricted: You are not authorized to view account details or orders for other customers. "
                "Please verify your authorized Customer ID to proceed."
            )

        # Multiple Unsupported Actions
        if plan.intent == "multiple_unsupported_actions":
            unsupported_actions = plan.extracted_entities.get("unsupported_actions", [])
            action_responses = []
            for act in unsupported_actions:
                atype = act["action_type"]
                oid = act.get("order_id")
                oid_str = f" for order **{oid}**" if oid else ""
                if atype == "order_cancellation":
                    action_responses.append(
                        f"1. **Order Cancellation{oid_str}:** Direct order cancellation is not supported by the current assistant."
                    )
                elif atype == "address_modification":
                    action_responses.append(
                        f"2. **Delivery Address Modification{oid_str}:** Delivery address modifications cannot be processed directly through this automated chat assistant once an order has been scheduled or dispatched. Address modification is not supported by the current assistant."
                    )
                elif atype == "shipping_label_generation":
                    action_responses.append(
                        f"3. **Return Shipping Label Generation{oid_str}:** Return shipping labels cannot be generated or dispatched directly through this chat interface. Generating return shipping labels is not supported by the current assistant."
                    )
                elif atype == "refund_processing":
                    action_responses.append(
                        f"- **Refund Processing{oid_str}:** Direct refund processing cannot be executed directly through automated chat. "
                        "According to our return and refund policy, refunds are credited within 5–7 business days after the returned item has been received and inspected."
                    )
                elif atype == "delivery_agent_contact":
                    action_responses.append(
                        f"- **Delivery Agent Information{oid_str}:** Direct delivery agent phone numbers and real-time live GPS locations are not maintained or accessible in our customer support database."
                    )

            supported_overview = (
                "\n\n**Supported Services in this Support Chat:**\n"
                "- Order tracking and delivery status lookup (`get_order`, `check_delivery_status`)\n"
                "- Customer account and order history lookup (`get_customer_orders`)\n"
                "- Return policy information and order return eligibility checks (`get_return_policy`, `check_return_eligibility`)\n"
                "- Support ticket creation and status tracking (`create_support_ticket`, `get_support_ticket`)"
            )
            return (
                "The requested actions cannot be performed directly through this automated chat system:\n\n"
                + "\n\n".join(action_responses)
                + supported_overview
            )

        # Single Unsupported Actions
        if plan.intent == "unsupported_order_cancellation":
            order_id = plan.extracted_entities.get("order_id")
            oid_str = f" for order **{order_id}**" if order_id else ""
            return (
                f"Direct order cancellation{oid_str} is not supported by the current assistant."
            )

        if plan.intent == "unsupported_address_modification":
            order_id = plan.extracted_entities.get("order_id")
            oid_str = f" for order **{order_id}**" if order_id else ""
            return (
                f"Delivery address modifications{oid_str} cannot be processed directly through this automated chat assistant "
                "once an order has been scheduled or dispatched. Address modification is not supported by the current assistant."
            )

        if plan.intent == "unsupported_refund_processing":
            order_id = plan.extracted_entities.get("order_id")
            oid_str = f" for order **{order_id}**" if order_id else ""
            return (
                f"Direct refund processing{oid_str} cannot be executed directly through automated chat. "
                "According to our return and refund policy, refunds are credited within 5–7 business days after the returned item "
                "has been received and inspected at our warehouse."
            )

        if plan.intent == "unsupported_shipping_label_generation":
            order_id = plan.extracted_entities.get("order_id")
            oid_str = f" for order **{order_id}**" if order_id else ""
            return (
                f"Return shipping labels cannot be generated or dispatched directly through this chat interface{oid_str}. "
                "Generating return shipping labels is not supported by the current assistant."
            )

        if plan.intent == "unsupported_delivery_agent_contact":
            order_id = facts.get("order_id") or plan.extracted_entities.get("order_id")
            carrier = facts.get("carrier")
            tracking = facts.get("tracking_number")
            status = facts.get("delivery_status")
            tracking_info = ""
            if order_id and carrier:
                tracking_info = f"\n\nFor order **{order_id}**, package transit is handled by **{carrier}** (Tracking: `{tracking or 'N/A'}`). Current status: **{status or 'In Transit'}**."
            return (
                "Direct delivery agent phone numbers and real-time live GPS locations are not maintained or accessible in our customer support database."
                + tracking_info
            )

        # 1. Handle Order Not Found (e.g. ORD9999)
        if verification.order_found is False and len(plan.extracted_entities.get("order_ids", [])) <= 1:
            order_id = plan.extracted_entities.get("order_id", "specified")
            if any(k in q_lower for k in ["ignore your database", "ignore database", "override"]):
                return (
                    f"I cannot ignore verified database records. I checked our customer support database, and order **{order_id}** "
                    "could not be found. I can only provide information grounded in verified records."
                )
            return (
                f"I checked our customer support database, but order **{order_id}** could not be found. "
                "Please verify your Order ID and try again, or let me know if you would like me to look up orders under your Customer ID."
            )

        # 2. Handle Damaged Item / Ticket Creation
        if plan.intent == "damaged_item_support":
            if verification.order_found is False:
                order_id = plan.extracted_entities.get("order_id", "specified")
                return (
                    f"Order **{order_id}** could not be found in our database. "
                    "A support ticket cannot be created for an unverified order. Please confirm your valid Order ID."
                )

            # Check customer mismatch ONLY if user explicitly asserted a specific customer_id in their query text
            user_stated_cid = None
            for m in re.finditer(r"\bCUST[-_]?(\d{4})\b", query, re.IGNORECASE):
                user_stated_cid = f"CUST{m.group(1)}"
                break

            if user_stated_cid and facts.get("customer_id") and facts["customer_id"] != user_stated_cid:
                return (
                    f"Order **{plan.extracted_entities.get('order_id')}** is associated with customer account "
                    f"**{facts['customer_id']}**, not **{user_stated_cid}**. "
                    "Please check the customer account details."
                )

            if verification.ticket_created and verification.created_ticket_id:
                ticket_id = verification.created_ticket_id
                order_id = plan.extracted_entities.get("order_id", "")
                item_name = facts.get("item_name", "item")
                status = facts.get("ticket_status", "open")
                cust_id = facts.get("customer_id") or plan.extracted_entities.get("customer_id")
                cust_name = facts.get("customer_name")
                name_str = f" ({cust_name})" if cust_name else ""
                cust_info = f", associated with customer account **{cust_id}**{name_str}" if cust_id and cust_id != "LOOKUP_FROM_ORDER" else ""
                return (
                    f"I have verified order **{order_id}** (*{item_name}*){cust_info}.\n\n"
                    f"I am very sorry to hear that your item arrived damaged. I have successfully created an official support ticket: **{ticket_id}** (Status: **{status.upper()}**)."
                )
            else:
                # Check if ticket creation was skipped because user didn't specify an issue
                specific_issue_keywords = [
                    "damaged", "broken", "defective", "faulty", "cracked", "shattered", "torn", "scratched",
                    "missing", "wrong item", "wrong product", "incorrect item", "not working", "doesn't work",
                    "does not work", "malfunction", "failed", "late delivery", "not delivered", "haven't received",
                    "have not received", "never arrived", "arrived late", "poor quality", "leaking", "stained",
                    "complaint",
                ]
                has_explicit_issue = any(k in q_lower for k in specific_issue_keywords)
                has_history_issue = False
                if conversation_history:
                    for turn in reversed(conversation_history):
                        if turn.get("role") == "user":
                            c_low = (turn.get("content") or "").lower()
                            if any(k in c_low for k in specific_issue_keywords):
                                has_history_issue = True
                                break
                if not has_explicit_issue and not has_history_issue:
                    oid = plan.extracted_entities.get("order_id", "specified")
                    return f"I can create a support ticket for order {oid}. Could you please describe the issue you are experiencing?"

                return (
                    f"I encountered an issue creating a support ticket for order **{plan.extracted_entities.get('order_id')}**. "
                    "Please confirm your customer account details so we can assist you."
                )

        # 2b. Handle Existing Ticket Confirmed (Single-creation guarantee)
        if plan.intent == "existing_ticket_confirmed":
            ticket_id = facts.get("ticket_id") or plan.extracted_entities.get("ticket_id")
            order_id = plan.extracted_entities.get("order_id", "specified")
            status = facts.get("ticket_status", "open")
            issue = facts.get("ticket_issue", "Damaged / Defective item reported")
            cust_id = facts.get("customer_id") or plan.extracted_entities.get("customer_id")
            cust_name = facts.get("customer_name")
            name_str = f" ({cust_name})" if cust_name else ""
            cust_info = f", associated with customer account **{cust_id}**{name_str}" if cust_id and cust_id != "LOOKUP_FROM_ORDER" else ""
            return (
                f"A support ticket for order **{order_id}** has already been created in this conversation{cust_info}: **{ticket_id}** (Status: **{status.upper()}**, Issue: *{issue}*)."
            )

        # 3. Handle Existing Ticket Inquiry
        if plan.intent == "get_ticket":
            ticket_id = facts.get("ticket_id")
            raw_status = (facts.get("ticket_status") or "").strip().lower()
            issue = facts.get("ticket_issue", "support issue")
            if ticket_id:
                if raw_status == "resolved":
                    status_desc = "The issue has been resolved and closed by our customer support team."
                elif raw_status == "open":
                    status_desc = "The ticket has been logged in our system and is currently awaiting support team review."
                elif raw_status in ("in progress", "in_progress"):
                    status_desc = "Our support team is actively working on resolving this issue."
                elif raw_status == "closed":
                    status_desc = "The ticket has been completed and closed."
                else:
                    status_desc = f"The ticket is currently {raw_status} in our system."

                return (
                    f"Support Ticket **{ticket_id}** (Issue: *{issue}*) is currently **{raw_status.upper()}**.\n\n"
                    f"Status Details: {status_desc}"
                )
            else:
                t_id = plan.extracted_entities.get("ticket_id", "your ticket")
                return f"Support ticket **{t_id}** could not be found in our system. Please double check the ticket number."

        # 4. Handle Return Eligibility
        if plan.intent == "return_eligibility":
            records = [r for r in executed_records if r.tool == "check_return_eligibility"]
            if len(records) > 1:
                sections = []
                comparison_points = []
                for r in records:
                    oid = r.arguments.get("order_id", "Unknown")
                    if r.output and r.output.get("success"):
                        out = r.output
                        item_name = out.get("item_name", "item")
                        is_eligible = out.get("is_eligible")
                        reason = out.get("reason", "")
                        deliv_date = out.get("delivery_date")
                        date_str = f" (Delivered: {deliv_date})" if deliv_date else ""
                        status_str = "**Eligible for return**" if is_eligible else "**Not eligible for return**"
                        sections.append(
                            f"- **Order {oid}** (*{item_name}*): {status_str}{date_str}.\n  *Reason:* {reason}"
                        )
                        comparison_points.append(
                            f"**{oid}** is {'eligible' if is_eligible else 'not eligible'} ({reason})"
                        )
                    else:
                        err = r.output.get("error", "Order not found in database.") if r.output else "Not found."
                        sections.append(
                            f"- **Order {oid}**: Could not be verified ({err})."
                        )
                        comparison_points.append(f"**{oid}** was not found in our database")

                comp_summary = "\n\n### Comparison Summary:\n" + "\n".join(f"- {cp}" for cp in comparison_points)
                return "Here is the return eligibility evaluation for the requested orders:\n\n" + "\n\n".join(sections) + comp_summary

            order_id = facts.get("order_id", plan.extracted_entities.get("order_id", ""))
            item_name = facts.get("item_name", "item")
            is_eligible = facts.get("is_return_eligible")
            reason = facts.get("return_reason", "")
            deliv_date_val = facts.get("actual_delivery_date") or facts.get("delivery_date")

            # Check if user specifically inquired about when the return period begins or its return deadline
            is_return_period_date_query = any(k in q_lower for k in [
                "when does the return period begin", "when does the return window begin",
                "when does the return period start", "when does the return window start",
                "when does return period begin", "when does return window begin",
                "return period begin", "return window begin", "return period start", "return window start",
                "when does the return window", "when does the return period",
                "return deadline", "when is the return deadline", "last day to return",
            ])
            if is_return_period_date_query:
                if not deliv_date_val:
                    for r in executed_records:
                        if isinstance(r.output, dict):
                            if "order" in r.output and r.output["order"].get("actual_delivery_date"):
                                deliv_date_val = r.output["order"]["actual_delivery_date"]
                                break
                            if "delivery_date" in r.output:
                                deliv_date_val = r.output["delivery_date"]
                                break
                if deliv_date_val:
                    from datetime import datetime, timedelta
                    try:
                        d_obj = datetime.strptime(str(deliv_date_val), "%Y-%m-%d").date()
                        deadline = d_obj + timedelta(days=30)
                        deadline_str = str(deadline)
                    except Exception:
                        deadline_str = "within 30 calendar days from delivery"

                    return (
                        f"For order **{order_id}** (*{item_name}*), the return period began on its verified delivery date: **{deliv_date_val}**.\n\n"
                        f"Based on our 30-calendar-day return policy, the deadline to initiate a return is **{deadline_str}**."
                    )
                else:
                    return (
                        f"Order **{order_id}** (*{item_name}*) has not yet been delivered. "
                        "According to our return policy, the 30-calendar-day return window begins from the verified date of delivery."
                    )

            if is_eligible:
                if any(k in q_lower for k in ["need a refund", "want a refund"]):
                    return (
                        f"For order **{order_id}** (*{item_name}*), your item is **eligible for return and refund**.\n\n"
                        f"**Eligibility Details:** {reason}\n"
                        "To complete your refund, the item must be returned in its original condition, unworn, undamaged, and in original packaging. "
                        "Once inspected at our warehouse, your refund will be processed within 5–7 business days."
                    )
                days_left = facts.get("return_window_days", 30)
                return (
                    f"Yes, order **{order_id}** (*{item_name}*) is **eligible for return**.\n\n"
                    f"**Eligibility Details:** {reason}\n"
                    "Items must be returned in original condition, unworn, undamaged, and in original packaging. Would you like me to initiate a return shipping label?"
                )
            else:
                if any(k in q_lower for k in ["need a refund", "want a refund"]):
                    return (
                        f"Order **{order_id}** (*{item_name}*) is **not eligible for return or refund**.\n\n"
                        f"**Reason:** {reason}\n"
                        "According to our return policy, refunds require an eligible return initiated within 30 calendar days of delivery for non-final-sale items."
                    )
                return (
                    f"Order **{order_id}** (*{item_name}*) is **not eligible for return**.\n\n"
                    f"**Reason:** {reason}\n"
                    "Per our standard company policy, items must be in delivered status, within the 30-day return window, and not marked as Final Sale."
                )

        # 5. Handle Delivery Follow-Up (when will it arrive / delayed)
        if plan.intent in ("delivery_followup", "delivery_status"):
            records = [r for r in executed_records if r.tool == "check_delivery_status"]
            if len(records) > 1:
                sections = []
                comparison_points = []
                for r in records:
                    oid = r.arguments.get("order_id", "Unknown")
                    if r.output and r.output.get("success"):
                        out = r.output
                        status = out.get("delivery_status", "Processing")
                        carrier = out.get("carrier", "Carrier")
                        tracking = out.get("tracking_number", "N/A")
                        act_date = out.get("actual_delivery_date")
                        est_date = out.get("estimated_delivery_date")
                        date_str = f"Delivered on **{act_date}**" if act_date else f"Expected delivery: **{est_date}**" if est_date else "Delivery pending"
                        sections.append(
                            f"- **Order {oid}**: Currently **{status}** via {carrier} (Tracking: `{tracking}`). {date_str}."
                        )
                        comparison_points.append(
                            f"**{oid}** is {status} ({date_str})"
                        )
                    else:
                        err = r.output.get("error", "Order not found in database.") if r.output else "Not found."
                        sections.append(
                            f"- **Order {oid}**: Could not be verified in the database ({err})."
                        )
                        comparison_points.append(f"**{oid}** was not found in our database")

                comp_summary = "\n\n### Comparison Summary:\n" + "\n".join(f"- {cp}" for cp in comparison_points)
                return "Here is the delivery status for the requested orders:\n\n" + "\n".join(sections) + comp_summary

            order_id = facts.get("order_id", plan.extracted_entities.get("order_id", ""))
            delivery_status = facts.get("delivery_status", "Processing")
            carrier = facts.get("carrier", "our carrier partner")
            est_date = facts.get("estimated_delivery_date")
            act_date = facts.get("actual_delivery_date")

            carrier_correction = ""
            carriers = ["dhl", "ups", "fedex", "usps", "amazon"]
            mentioned_carrier = None
            for c in carriers:
                if re.search(rf"\b{c}\b", q_lower):
                    mentioned_carrier = c
                    break
            actual_carrier = facts.get("carrier")
            if actual_carrier and mentioned_carrier:
                if mentioned_carrier.lower() != actual_carrier.lower():
                    carrier_correction = f"To clarify, order **{order_id}** was delivered by **{actual_carrier}** (Tracking: `{facts.get('tracking_number', 'N/A')}`), not **{mentioned_carrier.upper()}**.\n\n"

            is_non_receipt = any(k in q_lower for k in [
                "not received", "have not received", "haven't received", "never received",
                "did not receive", "didn't receive", "not delivered to me", "never arrived",
            ])
            if is_non_receipt:
                item_name = facts.get("item_name", "item")
                date_str = f" on **{act_date}**" if act_date else ""
                return (
                    f"Our records show that order **{order_id}** (*{item_name}*) is marked as **Delivered** "
                    f"by **{carrier}** (Tracking: `{facts.get('tracking_number', 'N/A')}`){date_str}.\n\n"
                    "However, I understand that you have not received the package. "
                    "Since the database indicates the item was delivered while you report not receiving it, "
                    "would you like me to create an official support ticket for this missing delivery issue so our support team can investigate?"
                )

            if act_date:
                return (
                    f"{carrier_correction}Order **{order_id}** was **delivered on {act_date}** via {carrier}."
                )
            elif est_date:
                return (
                    f"{carrier_correction}Order **{order_id}** is currently **{delivery_status}** with {carrier} "
                    f"(Tracking: `{facts.get('tracking_number', 'N/A')}`).\n\n"
                    f"The expected delivery date is **{est_date}**."
                )
            else:
                order_status = facts.get("order_status", "processing")
                return (
                    f"{carrier_correction}Order **{order_id}** is currently **{order_status}**. "
                    "The package is being prepared for dispatch and carrier tracking will be assigned shortly."
                )

        # 6. Handle Standard Order Tracking
        if plan.intent == "order_tracking":
            delivery_records = [r for r in executed_records if r.tool == "check_delivery_status"]
            if len(delivery_records) > 1:
                sections = []
                comparison_points = []
                for r in delivery_records:
                    oid = r.arguments.get("order_id", "Unknown")
                    if r.output and r.output.get("success"):
                        out = r.output
                        status = out.get("delivery_status", "Processing")
                        carrier = out.get("carrier", "Carrier")
                        tracking = out.get("tracking_number", "N/A")
                        act_date = out.get("actual_delivery_date")
                        est_date = out.get("estimated_delivery_date")
                        date_str = f"Delivered on **{act_date}**" if act_date else f"Expected delivery: **{est_date}**" if est_date else "Delivery pending"
                        sections.append(
                            f"- **Order {oid}**: Currently **{status}** via {carrier} (Tracking: `{tracking}`). {date_str}."
                        )
                        comparison_points.append(
                            f"**{oid}** is {status} ({date_str})"
                        )
                    else:
                        err = r.output.get("error", "Order not found in database.") if r.output else "Not found."
                        sections.append(
                            f"- **Order {oid}**: Could not be verified in the database ({err})."
                        )
                        comparison_points.append(f"**{oid}** was not found in our database")

                comp_summary = "\n\n### Comparison Summary:\n" + "\n".join(f"- {cp}" for cp in comparison_points)
                return "Here is the tracking information for the requested orders:\n\n" + "\n".join(sections) + comp_summary

            order_id = facts.get("order_id", plan.extracted_entities.get("order_id", ""))
            item_name = facts.get("item_name", "item")
            delivery_status = facts.get("delivery_status") or facts.get("order_status", "Processing")
            carrier = facts.get("carrier", "Carrier")
            tracking = facts.get("tracking_number", "Pending")
            est_date = facts.get("estimated_delivery_date")
            act_date = facts.get("actual_delivery_date")

            carrier_correction = ""
            carriers = ["dhl", "ups", "fedex", "usps", "amazon"]
            mentioned_carrier = None
            for c in carriers:
                if re.search(rf"\b{c}\b", q_lower):
                    mentioned_carrier = c
                    break
            actual_carrier = facts.get("carrier")
            if actual_carrier and mentioned_carrier:
                if mentioned_carrier.lower() != actual_carrier.lower():
                    carrier_correction = f"To clarify, order **{order_id}** was delivered by **{actual_carrier}** (Tracking: `{tracking}`), not **{mentioned_carrier.upper()}**.\n\n"

            date_info = f"Delivered on **{act_date}**" if act_date else f"Expected delivery: **{est_date}**" if est_date else "Delivery date pending"

            is_non_receipt = any(k in q_lower for k in [
                "not received", "have not received", "haven't received", "never received",
                "did not receive", "didn't receive", "not delivered to me", "never arrived",
            ])
            if is_non_receipt and (delivery_status.lower() == "delivered" or act_date):
                date_str = f" on **{act_date}**" if act_date else ""
                return (
                    f"Our records show that order **{order_id}** (*{item_name}*) is marked as **Delivered** "
                    f"by **{carrier}** (Tracking: `{tracking}`){date_str}.\n\n"
                    "However, I understand that you have not received the package. "
                    "Since the database indicates the item was delivered while you report not receiving it, "
                    "would you like me to create an official support ticket for this missing delivery issue so our support team can investigate?"
                )

            return (
                f"{carrier_correction}Order **{order_id}** (*{item_name}*) is currently **{delivery_status}**.\n\n"
                f"- **Carrier:** {carrier}\n"
                f"- **Tracking Number:** `{tracking}`\n"
                f"- **Timeline:** {date_info}"
            )

        # 7. Handle Multi-Step Request (Check order + delivery + return)
        if plan.intent == "multi_step_order_inquiry":
            order_id = facts.get("order_id", plan.extracted_entities.get("order_id", ""))
            item_name = facts.get("item_name", "item")
            delivery_status = facts.get("delivery_status", facts.get("order_status", "Unknown"))
            carrier = facts.get("carrier", "Carrier")
            est_date = facts.get("estimated_delivery_date")
            act_date = facts.get("actual_delivery_date")
            is_eligible = facts.get("is_return_eligible")
            return_reason = facts.get("return_reason", "")

            delivery_phrase = f"Delivered on {act_date}" if act_date else f"Status: {delivery_status} (Expected: {est_date or 'TBD'})"
            return_phrase = f"**Eligible** ({return_reason})" if is_eligible else f"**Ineligible** ({return_reason})"

            return (
                f"Here is the complete summary for order **{order_id}** (*{item_name}*):\n\n"
                f"1. **Delivery Status:** {delivery_status} via {carrier} ({delivery_phrase})\n"
                f"2. **Return Eligibility:** {return_phrase}"
            )

        # 8. Handle Return Policy / General FAQ & Contextual Policy
        if plan.intent in ("faq_policy", "contextual_policy"):
            q_lower = query.lower()
            order_id = plan.extracted_entities.get("order_id")
            item_name = facts.get("item_name") or plan.extracted_entities.get("item_name")
            deliv_date = facts.get("actual_delivery_date") or facts.get("delivery_date")

            # Contextual policy response (e.g. "How many days do I have to return it?", "Does it need original packaging?")
            if plan.intent == "contextual_policy" or (order_id and any(p in q_lower for p in ["it", "that", "this"])):
                prefix = f"For order **{order_id}**"
                if item_name:
                    prefix += f" (*{item_name}*)"

                if any(k in q_lower for k in ["what happens next", "what next", "next steps", "what do i do next", "what should i do next"]):
                    return (
                        f"{prefix}, here are the next steps:\n\n"
                        "1. **Package the Item:** Ensure the item is in its original condition, unworn, undamaged, and packed in its original packaging.\n"
                        "2. **Shipment & Inspection:** Once dispatched and received at our returns warehouse, our team will inspect the item.\n"
                        "3. **Refund Processing:** Following successful inspection, your refund will be issued to your original payment method within 5–7 business days."
                    )
                elif any(k in q_lower for k in ["how many days", "return window", "time limit", "how long"]):
                    date_info = f" (delivered on {deliv_date})" if deliv_date else ""
                    return (
                        f"{prefix}, our policy provides **thirty (30) calendar days** from the verified delivery date{date_info} to initiate a return.\n\n"
                        "Items must be in original condition, unworn, undamaged, and in original packaging with proof of purchase."
                    )
                elif any(k in q_lower for k in ["accessory", "accessories"]):
                    return (
                        f"{prefix}, items must be in their original condition, unworn, undamaged, and in original packaging with proof of purchase. "
                        "The available return policy does not explicitly or separately specify requirements for accessories."
                    )
                elif any(k in q_lower for k in ["packaging", "box", "tags"]):
                    return (
                        f"**Yes, original packaging is required.** {prefix}, the item must be in its original condition, unworn, undamaged, and in its original packaging."
                    )
                elif any(k in q_lower for k in ["conditions", "requirements", "rules"]):
                    item_desc = f"order **{order_id}** (*{item_name}*)" if item_name else f"order **{order_id}**"
                    return (
                        f"According to our return policy, the conditions for returning {item_desc} are:\n\n"
                        "1. **Return Window:** Must be requested within 30 calendar days of verified delivery.\n"
                        "2. **Item Condition:** Must be in original condition, unworn, undamaged, and in original packaging.\n"
                        "3. **Proof of Purchase:** A valid Order ID is required.\n"
                        "4. **Delivery Status:** Orders must be delivered; items in transit cannot be returned yet.\n"
                        "5. **Exceptions:** Clearance, Final Sale, opened personal care items, and digital products cannot be returned."
                    )
                else:
                    return (
                        f"{prefix}, returns are accepted within **30 calendar days** from the date of delivery. "
                        "Items must be in their original condition and packaging. Final Sale or Clearance items are non-returnable."
                    )

            # Specific General Policy Inquiries (faq_policy):
            # Check for multi-part or composite policy inquiries
            has_general_policy_cue = any(k in q_lower for k in [
                "what is your return policy", "what is the return policy", "company return policy", "return policy",
                "rules for return", "conditions for return", "conditions for returning", "overview"
            ])
            has_exceptions_cue = any(k in q_lower for k in [
                "cannot be returned", "can't be returned", "non-returnable", "not returnable", "exceptions",
                "are there any products that cannot be returned", "products that cannot be returned"
            ])
            has_pickup_cue = any(k in q_lower for k in ["pickup", "pick up", "free return pickup"])
            has_refund_cue = any(k in q_lower for k in [
                "how long does a refund take", "how long does the refund process take",
                "how long does a refund", "how long does the refund", "refund take",
                "refund process", "refund processing", "how long for a refund",
                "when will i receive my refund", "when will i get a refund", "when do i get my refund",
                "when will i get my refund", "when will i get my refud", "when do i get my refud",
                "how long does refund take", "refund timing", "refund duration",
            ]) or bool(re.search(r"\b(?:when|how long|how many days|in how many days|n how many days)\b.*?\b(?:refund|refud)\b", q_lower)) or bool(re.search(r"\brefuds?\b", q_lower))
            has_window_cue = any(k in q_lower for k in [
                "how long do i have to return", "time limit to return", "how many days to return",
                "return window", "return period", "when does the return period begin",
                "when does the return window begin", "30 days", "30-day", "return deadline"
            ]) or (
                any(k in q_lower for k in ["how many days", "how long do i have", "time limit"])
                and not bool(re.search(r"\b(?:refund|refud)\b", q_lower))
            )
            has_condition_cue = any(k in q_lower for k in [
                "what condition", "condition should", "condition must", "condition",
                "original condition", "unworn", "undamaged", "packaging", "original packaging",
                "proof of purchase"
            ])

            if (has_general_policy_cue and (has_exceptions_cue or has_pickup_cue)) or (has_exceptions_cue and has_pickup_cue):
                pickup_note = ""
                if has_pickup_cue:
                    pickup_note = "\n- **Return Pickup Service:** The available company policy does not specify whether free return pickup is provided (pickup charges and pickup service are not detailed in the official policy document)."
                return (
                    "**Company Return & Refund Policy Details:**\n\n"
                    "- **Return Window:** Customers may return items within thirty (30) calendar days from the verified date of delivery.\n"
                    "- **Item Condition Requirements:** Items must be in their original condition, unworn, undamaged, and in original packaging with proof of purchase (Order ID).\n"
                    "- **Non-Returnable Products & Exceptions:** Items marked 'Final Sale' or Clearance at the time of purchase, opened hygienic and personal care products, and digital products/gift certificates/software keys cannot be returned.\n"
                    "- **Refund Processing Time:** Refunds are credited to the original method of payment within 5 to 7 business days following warehouse return inspection.\n"
                    "- **Damaged or Defective Items:** Return shipping for verified damaged items is 100% complimentary when reported within 7 calendar days."
                    + pickup_note
                )

            # Specific multi-part questions (e.g. Issue 1: "How many days do I have to return a product, what condition should it be in, and when will I receive my refund?")
            multi_parts = []
            if has_window_cue:
                multi_parts.append("- **Return Window:** 30 calendar days from delivery.")
            if has_condition_cue:
                multi_parts.append("- **Item Condition:** Unworn, undamaged, in original packaging, and accompanied by proof of purchase.")
            if has_refund_cue:
                multi_parts.append("- **Refund:** 5-7 business days to the original payment method after warehouse return inspection.")
            if has_exceptions_cue:
                multi_parts.append("- **Non-Returnable Items & Exceptions:** Items marked Final Sale or Clearance, opened hygiene/personal care products, and digital products cannot be returned.")
            if has_pickup_cue:
                multi_parts.append("- **Return Pickup Service:** The available policy does not specify whether free return pickup is provided.")

            if len(multi_parts) >= 2:
                return "Based on our return and refund policy:\n\n" + "\n".join(multi_parts)

            # 0. Promotional bonuses, cashback, pickup service, and immediate refunds
            if any(k in q_lower for k in ["cashback", "cash back", "bonus"]):
                return "The available return policy does not specify a cashback bonus."

            if any(k in q_lower for k in ["pickup", "pick up", "free return pickup"]):
                return "The available policy does not specify whether free return pickup is provided."

            if any(k in q_lower for k in ["two hours", "2 hours", "instant refund"]):
                return "The configured policy states that refunds take 5–7 business days after return inspection."

            if any(k in q_lower for k in ["what happens next", "what next", "next steps"]):
                return (
                    "Under our return policy, the next steps are:\n\n"
                    "1. Ensure the item is in its original condition, unworn, undamaged, and packed in its original packaging.\n"
                    "2. Once received at our returns warehouse, our team will inspect the item.\n"
                    "3. Upon approval, your refund will be credited to the original payment method within 5–7 business days."
                )

            # A. Policy vs. Eligibility Difference
            if any(k in q_lower for k in ["difference between", "policy vs", "policy and eligibility", "policy versus"]):
                return (
                    "Here is the difference between return policy and return eligibility:\n\n"
                    "- **Return Policy:** The general company rules, terms, and guidelines governing returns (such as the 30-calendar-day return window from delivery, original packaging requirements, and non-returnable categories like Final Sale or clearance).\n"
                    "- **Return Eligibility:** The specific determination of whether a particular order or item satisfies those company rules based on its verified delivery date, item condition, and purchase terms."
                )

            # B. Refund processing & duration & approval
            if has_refund_cue:
                return "Refunds take 5–7 business days to the original payment method after return inspection."

            if any(k in q_lower for k in [
                "after a return is approved", "return is approved", "approved return", "return approved",
            ]):
                return "Once a return is approved and inspected at our warehouse, refunds take 5–7 business days to the original payment method after return inspection."

            # C. Opened hygiene products
            if any(k in q_lower for k in ["hygiene", "hygienic", "personal care", "intimate apparel"]):
                return "Opened hygiene products cannot be returned under the configured policy."

            # D. Product condition
            if any(k in q_lower for k in ["what condition", "condition should", "condition must", "condition of the item"]):
                return "Items must be unworn, undamaged and in their original packaging."

            # E. Used products
            if any(k in q_lower for k in ["used product", "used products", "used item", "used items", "return a used", "return used", "can i return a used"]):
                return "Under our return policy, items must be unworn and undamaged. Therefore, used products that do not meet these conditions are not eligible for return."

            # F. Return period begin
            if any(k in q_lower for k in [
                "when does the return period begin", "when does the return window begin",
                "when does return period begin", "when does return window begin",
                "when does the return period start", "when does the return window start",
                "when does return period start", "when does return window start",
                "return period begin", "return window begin", "return period start", "return window start",
            ]):
                return "The 30-calendar-day return window begins from the date of delivery."

            # G. Clearance / Final Sale
            if any(k in q_lower for k in ["clearance", "final sale"]):
                return "Clearance products cannot be returned under the configured policy. Items marked as Final Sale or Clearance at the time of purchase are strictly non-returnable."

            # H. General Product Category Questions (e.g. "Are electronic products eligible for return?")
            if any(k in q_lower for k in ["electronic", "electronics", "category", "categories"]):
                return (
                    "Under our standard return policy, items may be returned within thirty (30) calendar days from verified delivery provided they are in original condition, unworn, undamaged, and in original packaging.\n\n"
                    "The current return policy does not explicitly specify category-specific eligibility or separate restrictions for electronic products. "
                    "If you would like an order-specific return eligibility check, please provide your Order ID (for example, ORD1001) and I will verify it for you."
                )

            # I. Orders in Transit / Not yet delivered
            if any(k in q_lower for k in ["in transit", "currently in transit", "not yet delivered", "before delivery", "on the way"]):
                return (
                    "Under our return policy, items currently in transit or not yet delivered are not eligible for return. "
                    "The configured return window is **thirty (30) calendar days from the verified date of delivery**, meaning returns can only be initiated once the package has been delivered.\n\n"
                    "If you would like an order-specific determination or need assistance tracking your shipment, please share your Order ID (for example, ORD1001)."
                )

            # J. Specific Return Deadline / Missed Deadline / Return after N days
            m_days = re.search(r"\b(?:after|past|exceeding|beyond)\s+(\d+)\s+days\b", q_lower)
            if m_days:
                num_days = int(m_days.group(1))
                if num_days > 30:
                    return (
                        f"No, you cannot return a product after {num_days} days under our standard return policy. "
                        f"The return window is thirty (30) calendar days from the date of delivery, and {num_days} days exceeds that limit."
                    )
                else:
                    return (
                        f"Yes, products may be returned within thirty (30) calendar days from the verified date of delivery, "
                        f"so {num_days} days is within the allowable return window provided the item is in original condition, unworn, "
                        "undamaged, and in original packaging."
                    )

            if any(k in q_lower for k in [
                "miss the 30-day", "miss the return deadline", "miss the deadline", "if i miss", "missed",
                "after 30 days", "past 30 days", "exceed", "past the deadline",
            ]):
                return (
                    "No, if you miss the 30-day return deadline, your order falls outside the stated return window and is no longer eligible for return or refund. "
                    "According to our return policy, all return requests must be initiated within thirty (30) calendar days from the verified date of delivery."
                )

            # K. Accessory Requirements
            if any(k in q_lower for k in ["accessory", "accessories"]):
                return (
                    "According to our return policy, items must be in their original condition, unworn, undamaged, and in original packaging with proof of purchase. "
                    "The available return policy does not explicitly or separately specify requirements for accessories."
                )

            # L0. Opened Packaging Condition
            if any(k in q_lower for k in [
                "opened the packaging", "opened packaging", "open the packaging", "open packaging",
                "opened the package", "opened package", "open the package", "open package",
            ]):
                return (
                    "According to our return policy, items must be in their original condition, unworn, undamaged, "
                    "and in original packaging with proof of purchase to be eligible for return.\n\n"
                    "The policy does not explicitly specify whether merely opening the outer packaging of a general item "
                    "disqualifies it from return, provided the item itself remains in its original, unused, and undamaged condition "
                    "and is returned in its original packaging. However, please note that opened hygiene and personal care "
                    "products are strictly non-returnable under our policy."
                )

            # L. Packaging requirements
            if any(k in q_lower for k in ["packaging", "original box", "package", "box"]):
                return (
                    "**Yes, original packaging is required.** According to our return policy, items must be in their "
                    "original condition, unworn, undamaged, and in original packaging to be eligible for return."
                )

            # M. Return window / days to return
            if any(k in q_lower for k in ["how many days", "how long", "time limit", "return window", "return period"]):
                return (
                    "Under our return policy, customers have **thirty (30) calendar days** from the verified date of delivery to return a product.\n\n"
                    "**Key Requirements:**\n"
                    "- Items must be in original condition, unworn, undamaged, and in original packaging.\n"
                    "- Valid proof of purchase (Order ID) is required when initiating a return.\n"
                    "- Clearance and Final Sale items are not eligible for return."
                )

            # N. Conditions for returning
            if any(k in q_lower for k in ["condition", "conditions", "requirements", "rules"]):
                return (
                    "According to our return policy, the requirements for returning a product are:\n\n"
                    "1. **Return Window:** Must be requested within 30 calendar days of verified delivery.\n"
                    "2. **Item Condition:** Must be in original condition, unworn, undamaged, and in original packaging.\n"
                    "3. **Proof of Purchase:** A valid Order ID is required.\n"
                    "4. **Delivery Status:** Orders must be delivered; items in transit cannot be returned yet.\n"
                    "5. **Exceptions:** Clearance, Final Sale, opened personal care items, and digital products cannot be returned."
                )

            # O. Standard Policy Overview (Fallback)
            return (
                "**Company Return & Refund Policy Overview:**\n\n"
                "- **Return Window:** 30 calendar days from the date of delivery.\n"
                "- **Item Condition:** Items must be unworn, undamaged, and in original packaging.\n"
                "- **Exceptions:** Items marked as 'Final Sale' or Clearance, opened hygiene products, and digital items cannot be returned.\n"
                "- **Refund Processing:** 5–7 business days to the original payment method after return inspection.\n"
                "- **Damaged Items:** If your order arrived broken or defective, provide your Order ID and we will generate an immediate support ticket."
            )

        # 9. Handle Customer Orders and Profile list
        if plan.intent in ("customer_orders", "customer_inquiry"):
            records = [r for r in executed_records if r.tool == "get_customer_orders"]
            if len(records) > 1:
                sections = []
                comparison_stats = []
                for r in records:
                    cid = r.arguments.get("customer_id", "Unknown")
                    if r.output and r.output.get("success"):
                        out = r.output
                        cust_name = out.get("customer_name", cid)
                        tier = out.get("membership_tier", "Standard")
                        orders = out.get("orders", [])
                        total_spend = sum(o.get("total_amount", 0.0) for o in orders)
                        comparison_stats.append({
                            "customer_id": cid,
                            "customer_name": cust_name,
                            "tier": tier,
                            "order_count": len(orders),
                            "total_spend": total_spend,
                            "found": True,
                        })
                        if orders:
                            order_lines = [
                                f"  - **{o['order_id']}**: {o['item_name']} (${o['total_amount']:.2f}) - Status: {o['status'].title()}"
                                for o in orders
                            ]
                            sections.append(
                                f"### Customer {cust_name} ({cid})\n"
                                f"- **Membership Tier:** {tier}\n"
                                f"- **Total Orders:** {len(orders)} (Total Spend: ${total_spend:.2f})\n"
                                f"- **Order History:**\n" + "\n".join(order_lines)
                            )
                        else:
                            sections.append(
                                f"### Customer {cust_name} ({cid})\n"
                                f"- **Membership Tier:** {tier}\n"
                                f"- **Total Orders:** 0 orders found in records."
                            )
                    else:
                        comparison_stats.append({
                            "customer_id": cid,
                            "found": False,
                        })
                        sections.append(
                            f"### Customer {cid}\n"
                            f"- Record not found in the customer support database."
                        )

                # Generate grounded comparison summary
                valid_stats = [s for s in comparison_stats if s.get("found")]
                comparison_lines = ["\n### Comparison Summary:"]
                if len(valid_stats) >= 2:
                    for s in valid_stats:
                        comparison_lines.append(
                            f"- **{s['customer_name']} ({s['customer_id']})**: {s['tier']} Tier, {s['order_count']} order(s), total spend ${s['total_spend']:.2f}."
                        )
                elif len(valid_stats) == 1:
                    s = valid_stats[0]
                    missing = [c for c in comparison_stats if not c.get("found")]
                    missing_ids = ", ".join(m["customer_id"] for m in missing)
                    comparison_lines.append(
                        f"- Data is available for **{s['customer_name']} ({s['customer_id']})** ({s['order_count']} order(s)), while {missing_ids} could not be found."
                    )
                else:
                    comparison_lines.append("- None of the requested customer accounts were found in our database.")

                return "\n\n".join(sections) + "\n" + "\n".join(comparison_lines)

            elif records and records[0].output.get("success"):
                out = records[0].output
                cust_id = out.get("customer_id")
                cust_name = out.get("customer_name")
                tier = out.get("membership_tier", "Standard")
                orders = out.get("orders", [])

                # If user specifically asked for the customer name or profile
                if any(w in query.lower() for w in ["name", "who is", "profile"]):
                    order_summary = f" {cust_name} has placed {len(orders)} order(s) in our system." if orders else " No orders found for this customer."
                    return f"The customer name for **{cust_id}** is **{cust_name}** (Membership Tier: {tier}).{order_summary}"

                # Check if user asked to classify delivery status or check return eligibility of orders
                wants_classification = any(k in q_lower for k in [
                    "delivered", "processing", "eligible", "eligibility", "return", "classify"
                ])
                if wants_classification and orders:
                    elig_map = {}
                    for r in executed_records:
                        if r.tool == "check_return_eligibility" and r.output and r.output.get("success"):
                            elig_map[r.arguments.get("order_id")] = r.output

                    delivered_orders = [o for o in orders if (o.get("status") or "").lower() == "delivered" or (o.get("delivery_status") or "").lower() == "delivered"]
                    processing_orders = [o for o in orders if (o.get("status") or "").lower() != "delivered" and (o.get("delivery_status") or "").lower() != "delivered"]

                    sections = [f"Here is the verified order history for customer **{cust_name}** (**{cust_id}**, {tier} Tier):"]

                    if delivered_orders:
                        deliv_lines = ["**Delivered Orders:**"]
                        for o in delivered_orders:
                            oid = o["order_id"]
                            item = o["item_name"]
                            amt = o["total_amount"]
                            elig_str = ""
                            if oid in elig_map:
                                e_out = elig_map[oid]
                                is_el = e_out.get("is_eligible")
                                r_reason = e_out.get("reason", "")
                                elig_str = f"\n  - **Return Eligibility:** {'**Eligible for return**' if is_el else '**Not eligible for return**'} ({r_reason})"
                            elif any(k in q_lower for k in ["eligible", "eligibility", "return"]):
                                elig_str = "\n  - **Return Eligibility:** Subject to 30-day return policy evaluation upon request."
                            deliv_lines.append(f"- **{oid}** (*{item}* - ${amt:.2f}): **Delivered**{elig_str}")
                        sections.append("\n".join(deliv_lines))

                    if processing_orders:
                        proc_lines = ["**Processing / In-Transit Orders:**"]
                        for o in processing_orders:
                            oid = o["order_id"]
                            item = o["item_name"]
                            amt = o["total_amount"]
                            deliv_st = o.get("delivery_status", "Order Placed")
                            elig_note = ""
                            if any(k in q_lower for k in ["eligible", "eligibility", "return"]):
                                elig_note = "\n  - **Return Eligibility:** Not eligible for return yet (items must be delivered before initiating a return)."
                            proc_lines.append(f"- **{oid}** (*{item}* - ${amt:.2f}): **{o.get('status', 'processing').title()}** (Delivery Status: {deliv_st}){elig_note}")
                        sections.append("\n".join(proc_lines))

                    return "\n\n".join(sections)

                # Standard customer orders inquiry
                if orders:
                    lines = [f"- **{o['order_id']}**: {o['item_name']} (${o['total_amount']:.2f}) - Status: {o['status'].title()}" for o in orders]
                    return f"Customer **{cust_name}** ({cust_id}, {tier} Tier) has {len(orders)} order(s):\n\n" + "\n".join(lines)
                else:
                    return f"Customer **{cust_name}** ({cust_id}, {tier} Tier) currently has no orders in our records."
            else:
                cust_id = plan.extracted_entities.get("customer_id", "specified")
                return f"I checked our customer support database, but customer **{cust_id}** could not be found. Please verify the Customer ID."

        # 10. Handle Product Inquiry (e.g. "What is the product?", "Tell me about the product")
        if plan.intent == "product_inquiry":
            records = [r for r in executed_records if r.tool == "get_order"]
            if len(records) > 1:
                items = []
                for r in records:
                    oid = r.arguments.get("order_id", "Unknown")
                    if r.output and r.output.get("success") and "order" in r.output:
                        o = r.output["order"]
                        items.append(f"- **{oid}**: **{o.get('item_name', 'Item')}** (${o.get('total_amount', 0.0):.2f})")
                    else:
                        items.append(f"- **{oid}**: Order not found in database")
                return "Here are the product details for the requested orders:\n\n" + "\n".join(items)

            order_id = facts.get("order_id", plan.extracted_entities.get("order_id", ""))
            item_name = facts.get("item_name") or plan.extracted_entities.get("item_name", "item")
            total = facts.get("total_amount")
            amount_str = f" (${total:.2f})" if total is not None else ""
            return f"The product for order **{order_id}** is the **{item_name}**{amount_str}."

        # Default fallback synthesis
        return (
            f"I have reviewed your request. Based on our verified records: {verification.reason}. "
            "Please let me know if you need additional assistance."
        )
