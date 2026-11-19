"""LLM Client for Ollama integration with transparent Demo Mode fallback.

Allows configurable local LLM inference while maintaining system resilience
and clear observability when Ollama is offline.
"""

import logging
from typing import Dict, Any, Optional
import httpx
import config

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for local Ollama LLM provider with health checks and structured requests."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """Verify Ollama server accessibility and model presence.

        Returns:
            Dict containing 'available', 'models', 'model_found', and 'error' details.
        """
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m.get("name") for m in data.get("models", [])]
                    model_found = any(self.model in m for m in models)
                    return {
                        "available": True,
                        "models": models,
                        "model_found": model_found,
                        "current_model": self.model,
                        "base_url": self.base_url,
                        "error": None,
                    }
                return {
                    "available": False,
                    "models": [],
                    "model_found": False,
                    "current_model": self.model,
                    "base_url": self.base_url,
                    "error": f"Ollama returned HTTP status {resp.status_code}",
                }
        except httpx.ConnectError:
            return {
                "available": False,
                "models": [],
                "model_found": False,
                "current_model": self.model,
                "base_url": self.base_url,
                "error": f"Cannot connect to Ollama at {self.base_url}. Ensure 'ollama serve' is running.",
            }
        except Exception as e:
            return {
                "available": False,
                "models": [],
                "model_found": False,
                "current_model": self.model,
                "base_url": self.base_url,
                "error": str(e),
            }

    async def generate(self, prompt: str, system: str = "") -> Optional[str]:
        """Generate a completion using Ollama.

        Returns string completion, or None if Ollama is unreachable.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/generate", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("response", "").strip()
                logger.warning("Ollama API error: status %s, body: %s", resp.status_code, resp.text)
                return None
        except Exception as e:
            logger.warning("Ollama connection failed: %s", str(e))
            return None


class LLMService:
    """Manages LLM routing between Ollama and deterministic rule engine."""

    def __init__(self):
        self.ollama = OllamaClient()
        self._cached_health = None

    def get_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get current provider status."""
        if self._cached_health is None or force_refresh:
            self._cached_health = self.ollama.check_health()
        return self._cached_health

    def is_ollama_available(self) -> bool:
        """Return True if Ollama is running and responsive."""
        status = self.get_status(force_refresh=True)
        return status["available"] and status["model_found"]

    def determine_active_mode(self) -> str:
        """Determine whether to use 'ollama' or 'demo_fallback'."""
        if config.DEMO_MODE == "true":
            return "demo_fallback"
        elif config.DEMO_MODE == "false":
            return "ollama"
        else:  # 'auto'
            return "ollama" if self.is_ollama_available() else "demo_fallback"
