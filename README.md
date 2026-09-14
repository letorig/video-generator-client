# Unified Video Gen

One async Python client for multiple video-generation providers: **Seedance, Kling, MiniMax/Hailuo and Wan**.

## Interfaces

- **CLI:** `video-gen generate ...`
- **Web UI:** `video-gen serve`, then open `http://127.0.0.1:8000`
- **Python SDK:** `from video_gen import VideoClient`
- **MCP server:** optional `mcp` extra

## Install

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[web]"
```

Copy `.env.example` to `.env` and add the credentials for the providers you use.

## CLI

```bash
video-gen --help
video-gen providers
video-gen models wan
video-gen generate wan --prompt "A snowy forest at dawn" --model 2.1 --duration 5 --output output/snow.mp4
video-gen status wan TASK_ID
video-gen serve
```

Web server options:

```bash
video-gen serve --host 127.0.0.1 --port 8000
video-gen serve --host 0.0.0.0 --port 9000 --reload
```

## Web UI

The web interface is bundled directly into the Python package; there is **no npm, Node.js or frontend build step**. FastAPI serves the HTML and exposes JSON endpoints for provider discovery, generation, task polling and MP4 download.

The UI provides:

- provider/model selection
- text-to-video and image-to-video input
- negative prompt
- duration, aspect ratio and resolution
- live task status polling
- video preview and MP4 download
- in-memory task history for the current server session

## Python SDK

```python
import asyncio
from video_gen import VideoClient

async def main() -> None:
    async with VideoClient() as client:
        task = await client.generate(
            provider="wan",
            model="2.1",
            prompt="A cinematic shot of a snowy forest at dawn",
            duration=5,
        )
        result = await task.wait()
        print(result.video_url)
        await task.download("output/snow.mp4")

asyncio.run(main())
```

## Project layout

```text
unified-video-gen/
├── src/video_gen/
│   ├── client.py
│   ├── cli.py
│   ├── web.py
│   ├── catalog.py
│   ├── static/index.html
│   ├── providers/
│   └── utils/
├── examples/
├── tests/
├── mcp_server/
├── .env.example
├── .gitignore
├── LICENSE
└── pyproject.toml
```

## Important

Provider APIs and model IDs change independently of this SDK. Treat the provider adapters as integration code that may need updates when a provider changes its API.

No `bootstrap.sh` is included in this project.
