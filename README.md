# Video Gen Client

One async Python client for multiple video-generation providers: **Seedance, Kling, MiniMax/Hailuo and Wan**.


# Supported Models

# Supported Models

| Provider | Version | API Model ID | Notes |
|---|---|---|---|
| **Seedance** (ByteDance) | 2.5 | `doubao-seedance-2.5` | 30s one-take, native audio |
| Seedance | 2.0 | `doubao-seedance-2-0-260128` | Full 2.0, multimodal input |
| Seedance | 2.0 Mini | `doubao-seedance-2-0-mini-260615` | Budget tier, 480p/720p, 4–15s |
| Seedance | 2.0 Fast | `doubao-seedance-2-0-fast-260128` | Faster 2.0, lower quality |
| **Kling** (Kuaishou) | 3.0 / 3.0 Omni | `kling-v3` / `kling-v3-omni` | Flagship, multi-shot, native audio |
| Kling | 2.6 | `kling-v2-6` | First with synced audio + video |
| Kling | 2.5 Turbo | `kling-v2-5-turbo` | 1080p, cheap for high volume |
| Kling | 2.1 | `kling-v2-1-master` | Cinematic control, better semantics |
| **MiniMax / Hailuo** | H3 | `MiniMax-H3` | 4K, 60fps, 30s, native audio |
| MiniMax | 2.3 | `MiniMax-Hailuo-2.3` | 1080p, T2V + I2V |
| MiniMax | 2.3 Fast | `MiniMax-Hailuo-2.3-Fast` | Faster 2.3, I2V only |
| **Wan** (Alibaba) | 3.0 | `wan3.0` | All-in-one, 30s, up to 1080p |
| Wan | 2.7 | `wan2.7-t2v` / `wan2.7-i2v` / `wan2.7-r2v` | Specialized models |
| Wan | 2.6 | `wan2.6-t2v` / `wan2.6-i2v` | Multi-shot, longer clips |
| Wan | 2.5 | `wan2.5-t2v-preview` | 1080p, audio support |


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

## License
MIT.


