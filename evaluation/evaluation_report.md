# AI Customer Support Agent - Benchmark Evaluation Report

## 1. Overview and Evaluation Objectives
This evaluation benchmarks the AI Customer Support Agent across multi-step agentic workflows, evaluating intent classification, MCP tool selection, ground-truth verification, factual grounding, and execution latency across 22 representative customer scenarios.

## 2. Key Performance Metrics

| Metric | Formula / Definition | Value | Benchmark Target | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Tool Selection Accuracy** | Percentage of test cases where selected tools exactly matched ground-truth plan | **100.0%** | > 90% | PASS |
| **Task Success Rate** | Percentage of queries resolved correctly with proper clarification or grounded evidence | **100.0%** | > 95% | PASS |
| **Grounded Verification Pass Rate** | Percentage of reflection checks correctly confirming grounding or detecting missing data | **100.0%** | > 90% | PASS |
| **Average Latency** | Mean roundtrip execution time across stdio MCP transport | **6170.9 ms** | < 10000 ms | PASS |
| **Tool Execution Failures** | Total unexpected exceptions encountered during MCP tool execution | **0** | 0 | PASS |

## 3. Detailed Case-by-Case Results

| ID | Category | Query | Tools Executed | Tool Match | Success | Latency |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| TC01 | Order Tracking | Where is order ORD1001? | `check_delivery_status, get_order` | Yes | Yes | 7860.15 ms |
| TC02 | Order Tracking | Can you track my order ORD1008? | `check_delivery_status, get_order` | Yes | Yes | 6687.29 ms |
| TC03 | Delivery Follow-Up | My order ORD1002 has not arrived. When will it be delivered? | `check_delivery_status, get_order` | Yes | Yes | 6392.46 ms |
| TC04 | Delivery Follow-Up | When will ORD1005 arrive? | `check_delivery_status, get_order` | Yes | Yes | 6685.82 ms |
| TC05 | Return Eligibility | Can I return order ORD1001? | `get_order, check_return_eligibility` | Yes | Yes | 6464.12 ms |
| TC06 | Return Eligibility | Can I return order ORD1003? | `get_order, check_return_eligibility` | Yes | Yes | 7016.87 ms |
| TC07 | Return Eligibility | Is order ORD1007 eligible for a return? | `get_order, check_return_eligibility` | Yes | Yes | 5812.02 ms |
| TC08 | Return Eligibility | Can I return order ORD1006? | `get_order, check_return_eligibility` | Yes | Yes | 6207.07 ms |
| TC09 | Return Eligibility | Can I return order ORD1002? | `get_order, check_return_eligibility` | Yes | Yes | 6221.24 ms |
| TC10 | Ticket Creation | My order ORD1004 arrived damaged. Please create a support ticket. | `get_order, create_support_ticket` | Yes | Yes | 6205.8 ms |
| TC11 | Ticket Creation | The item in ORD1009 was defective and broken upon opening. Please open a ticket. | `get_order, create_support_ticket` | Yes | Yes | 5990.1 ms |
| TC12 | Ticket Status | What is the status of ticket TICKET1001? | `get_support_ticket` | Yes | Yes | 6256.05 ms |
| TC13 | Ticket Status | Check the progress of my ticket TICKET1003 | `get_support_ticket` | Yes | Yes | 5861.99 ms |
| TC14 | Ticket Status | What is the status of ticket TICKET9999? | `get_support_ticket` | Yes | Yes | 6239.66 ms |
| TC15 | Unknown Order | Where is order ORD9999? | `get_order` | Yes | Yes | 6168.05 ms |
| TC16 | Unknown Order | Track package ORD8888 | `get_order` | Yes | Yes | 6080.62 ms |
| TC17 | Ambiguous Request | I need help with my order. | `None (Clarification)` | Yes | Yes | 4341.17 ms |
| TC18 | Ambiguous Request | When will my package arrive? | `None (Clarification)` | Yes | Yes | 4366.26 ms |
| TC19 | General Policy FAQ | What is your return policy? | `get_return_policy` | Yes | Yes | 6257.45 ms |
| TC20 | General Policy FAQ | How do refunds and returns work? | `get_return_policy` | Yes | Yes | 5947.16 ms |
| TC21 | Multi-Step Workflow | Check order ORD1001, tell me its delivery status, and check whether it can be returned. | `check_delivery_status, get_order, check_return_eligibility` | Yes | Yes | 6479.02 ms |
| TC22 | Multi-Step Workflow | Check ORD1003, tell me if it arrived and if I can return it. | `check_delivery_status, get_order, check_return_eligibility` | Yes | Yes | 6219.31 ms |

## 4. Architectural Analysis and Takeaways
- **Zero Hallucination on Non-Existent Records**: Test cases TC14, TC15, and TC16 (unknown ticket/orders) verified that the agent correctly short-circuits and explains that the record does not exist rather than inventing delivery dates or tracking numbers.
- **Clarification Before Guessing**: TC17 and TC18 confirmed that missing required identifiers trigger natural clarification requests before attempting ungrounded database queries.
- **FastMCP Stdio Transport Performance**: -The agent completed all benchmark scenarios with an average end-to-end response latency of 6170.9 ms (approximately 6.17 seconds), remaining within the 10-second benchmark target.
- **State Persistence**: Ticket creation in TC10 and TC11 generated verifiable, persistent SQLite tickets (`TICKET1004`, `TICKET1005`) that persisted across calls.