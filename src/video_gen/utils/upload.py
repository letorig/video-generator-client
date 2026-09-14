"""Generic file-upload helpers."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import httpx


async def upload_file(
    client: httpx.AsyncClient,
    url: str,
    path: str,
    headers: dict | None = None,
) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"

    with p.open("rb") as file:
        files = {"file": (p.name, file, mime)}
        response = await client.post(url, files=files, headers=headers or {})

    response.raise_for_status()
    data = response.json()
    result = data.get("url") or data.get("data", {}).get("url")
    if not result:
        raise ValueError("Upload response did not contain a URL")
    return result


def file_to_data_uri(path: str) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"
