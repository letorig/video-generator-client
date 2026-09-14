"""Minimal stdio MCP server for Unified Video Gen."""
 
from __future__ import annotations

import asyncio
import json
from typing import Any

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
except ImportError as exc:
    raise SystemExit("Install MCP support with: pip install -e '.[mcp]'") from exc

from video_gen.client import VideoClient

server = Server("unified-video-gen")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_video",
            description="Generate a video and wait for the result.",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {"type": "string"},
                    "prompt": {"type": "string"},
                    "model": {"type": "string"},
                    "image_url": {"type": "string"},
                    "duration": {"type": "integer", "default": 5},
                    "aspect_ratio": {"type": "string", "default": "16:9"},
                },
                "required": ["provider", "prompt"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name != "generate_video":
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async with VideoClient() as client:
        task = await client.generate(
            arguments["provider"],
            prompt=arguments["prompt"],
            model=arguments.get("model"),
            image_url=arguments.get("image_url"),
            duration=arguments.get("duration", 5),
            aspect_ratio=arguments.get("aspect_ratio", "16:9"),
        )
        result = await task.wait()
        return [
            TextContent(
                type="text",
                text=json.dumps(result.model_dump(), indent=2),
            )
        ]


async def main() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
