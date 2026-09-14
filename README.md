# Video Gen Client

One async Python client for multiple video-generation providers: **Seedance, Kling, MiniMax/Hailuo and Wan**.


## Install

```bash
git clone https://github.com/letorig/video-generator-client
cd video-generator-client
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


