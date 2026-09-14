"""Video download helper."""

from __future__ import annotations

from pathlib import Path

import httpx


async def download_video(url: str, dest: str) -> str:
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with path.open("wb") as file:
                async for chunk in response.aiter_bytes():
                    file.write(chunk)

    return str(path)
