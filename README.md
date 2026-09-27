# Video Gen Client

One async Python client for multiple video-generation providers: **Seedance, Kling, MiniMax/Hailuo and Wan**.

## Supported Models

| Provider | Short name (`--model`) | API model ID | Notes |
|---|---|---|---|
| **Seedance** (ByteDance) | `2.5` | `seedance-2.5` | |
| Seedance | `2.0-mini` | `seedance-2.0-mini` | |
| Seedance | `fast` | `seedance-fast` | |
| **Kling** (Kuaishou) | `2.1` | `kling-v2-1` | |
| Kling | `2.0` | `kling-v2` | |
| Kling | `1.6` | `kling-v1-6` | |
| **MiniMax / Hailuo** | `hailuo` | `MiniMax-Hailuo-02` | Text-to-video |
| MiniMax | `t2v-01` | `T2V-01` | Text-to-video |
| MiniMax | `i2v-01` | `I2V-01` | Image-to-video |
| **Wan** (Alibaba DashScope) | `2.2` | `wanx2.2-t2v-plus` | |
| Wan | `2.1` | `wanx2.1-t2v-turbo` | Default for text-to-video |
| Wan | `2.1-plus` | `wanx2.1-t2v-plus` | |
| Wan | `i2v` | `wanx2.1-i2v-turbo` | Image-to-video |

All four providers support text-to-video and image-to-video. The `--model` flag takes the short names listed above; run `video-gen models <provider>` to print the current mapping from `src/video_gen/catalog.py`.


## Install

```bash
git clone https://github.com/letorig/video-generator-client
cd video-generator-client
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -e ".[web]"
```

Copy `.env.example` to `.env` and add the credentials for the providers you use.

## CLI
Output is saved to the output folder.

```bash
video-gen --help
video-gen providers
video-gen models wan
video-gen generate wan --prompt "A snowy forest at dawn" --model 2.1 --duration 5
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


