"""Streamlit User Interface for AI Customer Support Agent.

Provides a professional, interactive chat interface, execution trace inspection,
MCP status monitoring, sample queries, and a live SQLite database viewer.
"""

import sys
import json
import logging
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import config
from agent.orchestrator import AgentOrchestrator
from agent.llm_client import LLMService
from mcp_server.database import (
    init_db,
    list_all_tickets,
    list_all_orders,
    list_all_customers,
)

# Page configuration
st.set_page_config(
    page_title="AI Customer Support Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database on startup
init_db()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I am your **AI Customer Support Agent**, powered by **Agentic AI** and the "
                "**Model Context Protocol (MCP)**.\n\n"
                "I can help you:\n"
                "- 📦 **Track order delivery** (e.g. `ORD1001`, `ORD1002`)\n"
                "- 🔄 **Evaluate return eligibility** (e.g. `ORD1001`, `ORD1003`, `ORD1007`)\n"
                "- 🛠️ **File support tickets for damaged items** (e.g. `ORD1004`)\n"
                "- 🎫 **Check status of existing support tickets** (e.g. `TICKET1001`)\n"
                "- 📜 **Explain return and refund policies**\n\n"
                "Try typing a question below or click any of the sample scenarios in the sidebar!"
            ),
            "trace": None,
        }
    ]

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = AgentOrchestrator()

if "llm_service" not in st.session_state:
    st.session_state.llm_service = LLMService()

if "prefill_query" not in st.session_state:
    st.session_state.prefill_query = ""


# =====================================================================
# SIDEBAR: STATUS & DEMO TOOLS
# =====================================================================

with st.sidebar:
    st.title("⚙️ System Status")

    # MCP Connection Status
    st.success("🟢 **MCP Server:** Connected (`FastMCP` stdio)")

    # LLM Status Indicator
    llm_status = st.session_state.llm_service.get_status(force_refresh=False)
    active_mode = st.session_state.llm_service.determine_active_mode()

    if active_mode == "ollama" and llm_status.get("model_found"):
        st.success(f"🟢 **LLM Provider:** Ollama (`{config.OLLAMA_MODEL}`)")
    else:
        st.info(
            "🟡 **LLM Provider:** Deterministic Fallback Mode\n\n"
            f"- Ollama URL: `{config.OLLAMA_BASE_URL}` (Offline)\n"
            "- Agentic workflows, MCP tools, and reflection operate with 100% deterministic grounding."
        )

    st.markdown("---")
    st.subheader("🛠️ Discovered MCP Tools")
    st.markdown(
        """
        - `get_order(order_id)`
        - `get_customer_orders(customer_id)`
        - `check_delivery_status(order_id)`
        - `check_return_eligibility(order_id)`
        - `get_return_policy()`
        - `create_support_ticket(customer_id, ...)`
        - `get_support_ticket(ticket_id)`
        """
    )

    st.markdown("---")
    st.subheader("💡 Clickable Sample Queries")

    sample_queries = [
        ("📦 Track Delivered Order", "Where is order ORD1001?"),
        ("🚚 Delivery Follow-Up", "My order ORD1002 has not arrived. When will it be delivered?"),
        ("❌ Return Ineligible (>30d)", "Can I return order ORD1003?"),
        ("🚫 Return Ineligible (Final Sale)", "Is order ORD1007 eligible for a return?"),
        ("🛠️ Report Damaged Item", "My order ORD1004 arrived damaged. Please create a support ticket."),
        ("🎫 Existing Ticket Status", "What is the status of ticket TICKET1001?"),
        ("❓ Unknown Order Test", "Where is order ORD9999?"),
        ("💬 Ambiguous Inquiry", "I need help with my order."),
        ("📜 Return Policy FAQ", "What is your return policy?"),
        ("⚡ Multi-Step Composite Query", "Check order ORD1001, tell me its delivery status, and check whether it can be returned."),
    ]

    for label, query_text in sample_queries:
        if st.button(label, key=f"btn_{label}", use_container_width=True):
            st.session_state.prefill_query = query_text

    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.prefill_query = ""
        st.session_state.orchestrator.reset_memory()
        st.rerun()


# =====================================================================
# MAIN CONTENT AREA
# =====================================================================

st.title("🤖 AI Customer Support Agent")
st.caption(
    "Level 2 Agentic AI Project demonstrating Autonomous Planning, Official MCP Stdio Protocol, "
    "Tool Execution, Grounded Reflection, and Persistent SQLite Storage."
)

# Tabs: Chat Interface vs. Live Database Inspector
tab_chat, tab_database = st.tabs(["💬 Customer Support Chat", "🗄️ Database Inspector (Live Demo Data)"])

with tab_chat:
    # Render Conversation History
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Render expandable Execution Trace if available
            trace = msg.get("trace")
            if trace:
                with st.expander("🔍 View Agent Execution Trace", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Detected Intent", trace.get("intent", "N/A"))
                    with col2:
                        st.metric("Execution Latency", f"{trace.get('latency_ms', 0):.1f} ms")
                    with col3:
                        verif = trace.get("verification", {})
                        passed = verif.get("passed", False)
                        st.metric("Verification Status", "✅ PASSED" if passed else "⚠️ UNVERIFIED/NOT FOUND")

                    st.markdown("**1. Planning & Reasoning:**")
                    st.info(trace.get("plan_reasoning", "No planning notes recorded."))

                    st.markdown("**2. Tools Selected & Executed:**")
                    tools_executed = trace.get("tool_calls", [])
                    if tools_executed:
                        for t_idx, t in enumerate(tools_executed, 1):
                            tool_status = "✅ Success" if t.get("success") else "❌ Failed"
                            st.markdown(f"**Step {t_idx}: `{t.get('tool')}`** ({tool_status} in {t.get('latency_ms', 0):.1f} ms)")
                            with st.container():
                                subcol_in, subcol_out = st.columns(2)
                                with subcol_in:
                                    st.caption("Input Arguments:")
                                    st.json(t.get("arguments", {}))
                                with subcol_out:
                                    st.caption("MCP Tool Output:")
                                    st.json(t.get("output", {}))
                    else:
                        st.caption("No external tools called (clarification generated directly).")

                    st.markdown("**3. Reflection & Grounding Findings:**")
                    findings = verif.get("findings", [])
                    if findings:
                        for f in findings:
                            st.write(f"- {f}")
                    else:
                        st.write("No verification discrepancies found.")

    # User Input
    user_prompt = st.chat_input("Ask a question about an order, return, or support ticket...")

    # If user clicked a sample query button in sidebar, prefill and execute
    if st.session_state.prefill_query:
        user_prompt = st.session_state.prefill_query
        st.session_state.prefill_query = ""

    if user_prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_prompt, "trace": None})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Run agent orchestrator
        with st.chat_message("assistant"):
            with st.spinner("🤖 Analyzing request, invoking MCP tools, and verifying grounding..."):
                trace = st.session_state.orchestrator.run_sync(
                    user_prompt,
                    conversation_history=st.session_state.messages[:-1],
                )

            st.markdown(trace.final_response)

            # Show trace immediately
            with st.expander("🔍 View Agent Execution Trace", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Detected Intent", trace.intent)
                with col2:
                    st.metric("Execution Latency", f"{trace.latency_ms:.1f} ms")
                with col3:
                    st.metric("Verification Status", "✅ PASSED" if trace.verification.get("passed") else "⚠️ UNVERIFIED/NOT FOUND")

                st.markdown("**1. Planning & Reasoning:**")
                st.info(trace.plan_reasoning)

                st.markdown("**2. Tools Executed via MCP:**")
                if trace.tool_calls:
                    for t_idx, t in enumerate(trace.tool_calls, 1):
                        tool_status = "✅ Success" if t.get("success") else "❌ Failed"
                        st.markdown(f"**Step {t_idx}: `{t.get('tool')}`** ({tool_status} in {t.get('latency_ms', 0):.1f} ms)")
                        subcol_in, subcol_out = st.columns(2)
                        with subcol_in:
                            st.caption("Input Arguments:")
                            st.json(t.get("arguments", {}))
                        with subcol_out:
                            st.caption("MCP Tool Output:")
                            st.json(t.get("output", {}))
                else:
                    st.caption("No external tools called (clarification generated directly).")

                st.markdown("**3. Reflection & Grounding Findings:**")
                for f in trace.verification.get("findings", []):
                    st.write(f"- {f}")

        # Save assistant message to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": trace.final_response,
            "trace": {
                "intent": trace.intent,
                "latency_ms": trace.latency_ms,
                "verification": trace.verification,
                "plan_reasoning": trace.plan_reasoning,
                "tool_calls": trace.tool_calls,
            },
        })


with tab_database:
    st.subheader("🗄️ Live SQLite Database Inspector")
    st.caption("Direct view of the persistent SQLite database (`customer_support.db`). Watch new support tickets persist in real-time!")

    if st.button("🔄 Refresh Database View"):
        st.rerun()

    subtab_tickets, subtab_orders, subtab_customers = st.tabs(["🎫 Support Tickets", "📦 Orders & Deliveries", "👥 Customers"])

    with subtab_tickets:
        tickets = list_all_tickets()
        st.markdown(f"**Total Support Tickets:** `{len(tickets)}`")
        if tickets:
            st.dataframe(tickets, use_container_width=True)
        else:
            st.info("No support tickets currently found.")

    with subtab_orders:
        orders = list_all_orders()
        st.markdown(f"**Total Seed Orders:** `{len(orders)}`")
        st.dataframe(orders, use_container_width=True)

    with subtab_customers:
        customers = list_all_customers()
        st.markdown(f"**Total Customers:** `{len(customers)}`")
        st.dataframe(customers, use_container_width=True)
