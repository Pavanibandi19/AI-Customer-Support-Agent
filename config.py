"""Configuration module for AI Customer Support Agent.

Reads settings from environment variables and provides centralized access
to database paths, LLM configuration, and agent runtime settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Database configuration
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(BASE_DIR / "data" / "customer_support.db")))
if not DATABASE_PATH.is_absolute():
    DATABASE_PATH = BASE_DIR / DATABASE_PATH

# Ollama LLM configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Runtime mode: 'auto' (use Ollama if up, else fallback demo), 'true' (force demo), 'false' (strict Ollama)
DEMO_MODE = os.getenv("DEMO_MODE", "auto").lower()

# Agent execution bounds
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "5"))

# Log level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Server script path for stdio MCP client
MCP_SERVER_SCRIPT = BASE_DIR / "mcp_server" / "server.py"
