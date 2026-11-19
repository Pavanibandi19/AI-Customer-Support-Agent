"""Official MCP Client communicating with FastMCP Server over stdio transport.

Uses the official MCP Python SDK (stdio_client, ClientSession) to discover
and execute tools, read resources, and fetch registered prompts.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession
import config

logger = logging.getLogger(__name__)


class MCPSessionHandle:
    """Active session handle wrapping an open ClientSession connection."""

    def __init__(self, session: ClientSession):
        self._session = session

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List tools on this active session."""
        response = await self._session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema or {},
            }
            for tool in response.tools
        ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke tool on this active session."""
        logger.info("Calling MCP tool '%s' with arguments: %s", tool_name, arguments)
        result = await self._session.call_tool(tool_name, arguments)
        if result.content and len(result.content) > 0:
            first_content = result.content[0]
            text = getattr(first_content, "text", str(first_content))
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return {"raw_text": text}
        return {"success": True, "result": "Empty response"}

    async def list_resources(self) -> List[Dict[str, Any]]:
        """List resources on this active session."""
        response = await self._session.list_resources()
        return [
            {
                "uri": str(r.uri),
                "name": r.name,
                "description": r.description or "",
                "mimeType": r.mimeType or "text/plain",
            }
            for r in response.resources
        ]

    async def read_resource(self, uri: str) -> str:
        """Read resource on this active session."""
        logger.info("Reading MCP resource: %s", uri)
        result = await self._session.read_resource(uri)
        if result.contents and len(result.contents) > 0:
            first_content = result.contents[0]
            return getattr(first_content, "text", str(first_content))
        return ""

    async def list_prompts(self) -> List[Dict[str, Any]]:
        """List prompts on this active session."""
        response = await self._session.list_prompts()
        return [
            {
                "name": p.name,
                "description": p.description or "",
                "arguments": [
                    {"name": arg.name, "description": arg.description or "", "required": arg.required}
                    for arg in (p.arguments or [])
                ],
            }
            for p in response.prompts
        ]

    async def get_prompt(self, name: str, arguments: Dict[str, str]) -> str:
        """Get prompt template on this active session."""
        logger.info("Fetching MCP prompt '%s'", name)
        result = await self._session.get_prompt(name, arguments)
        if result.messages and len(result.messages) > 0:
            first_msg = result.messages[0]
            content = first_msg.content
            return getattr(content, "text", str(content))
        return ""


class MCPClientWrapper:
    """Client wrapper for communicating with the local MCP server over stdio."""

    def __init__(self, server_script: Optional[Path | str] = None, python_executable: Optional[str] = None):
        self.server_script = Path(server_script or config.MCP_SERVER_SCRIPT).resolve()
        self.python_executable = python_executable or sys.executable
        self.server_params = StdioServerParameters(
            command=self.python_executable,
            args=[str(self.server_script)],
        )

    @asynccontextmanager
    async def session(self):
        """Context manager creating a persistent ClientSession for workflow execution."""
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield MCPSessionHandle(session)

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Discover all available tools exposed by the MCP server."""
        async with self.session() as s:
            return await s.list_tools()

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke an MCP tool by name using the official SDK client."""
        async with self.session() as s:
            return await s.call_tool(tool_name, arguments)

    async def list_resources(self) -> List[Dict[str, Any]]:
        """Discover all registered resources on the MCP server."""
        async with self.session() as s:
            return await s.list_resources()

    async def read_resource(self, uri: str) -> str:
        """Read content of a registered MCP resource."""
        async with self.session() as s:
            return await s.read_resource(uri)

    async def list_prompts(self) -> List[Dict[str, Any]]:
        """Discover all registered prompt templates on the MCP server."""
        async with self.session() as s:
            return await s.list_prompts()

    async def get_prompt(self, name: str, arguments: Dict[str, str]) -> str:
        """Retrieve a registered prompt populated with arguments."""
        async with self.session() as s:
            return await s.get_prompt(name, arguments)
