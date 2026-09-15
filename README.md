# Video Gen Client

One async Python client for multiple video-generation providers: **Seedance, Kling, MiniMax/Hailuo, Wan and OrcaRouter**.


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

## OrcaRouter

[OrcaRouter](https://www.orcarouter.ai) is an OpenAI-compatible AI gateway that
also serves asynchronous video generation on the same inference origin. It is
registered as a first-class provider named `orcarouter`, alongside Seedance,
Kling, MiniMax and Wan.

There are **two ways to authenticate**, and they are separate, explicit choices:

| Choice | How | Where the key comes from |
|---|---|---|
| `OrcaRouter - API` | Paste an existing `sk-orca-…` key | [OrcaRouter console](https://www.orcarouter.ai/console/token) |
| `OrcaRouter - Auth` | Sign in with your OrcaRouter account (OAuth 2.0 + PKCE) | The consent screen issues the key |

Both produce the same kind of OrcaRouter API key, and both end at the same
credential seam — the inference code never knows which one was used.

```bash
# Sign in with an account. Opens your browser; Flow A (loopback redirect).
video-gen orcarouter login

# Or paste a key you already have.
video-gen orcarouter key

# Check what is stored, and how much of the catalog it can reach.
video-gen orcarouter status

# Remove the stored credential.
video-gen orcarouter logout

# List the video models this key is actually entitled to.
video-gen models orcarouter
video-gen models orcarouter --image     # only models that accept an image input
```

In the web UI, both choices appear in the **OrcaRouter account** panel. The
model list is a live dropdown; the API key never leaves the server process.

### No client secret, and no redirect to register

The connect flow uses PKCE with `S256`, so an intercepted authorization code
cannot be redeemed by anyone who does not hold the verifier — which never leaves
the process. There is nothing to apply for and no callback URL to pre-register.

Two flows are implemented, chosen by where the client runs:

- **Flow A (loopback redirect)** for `video-gen orcarouter login`, which runs on
  your own machine and can listen on `127.0.0.1`.
- **Flow B (out-of-band code)** for the web UI, which may be served from a host
  reached over the network, so a loopback callback would name the wrong machine.

### The key is durable, not a refresh token

PKCE returns a long-lived OrcaRouter API key, not an access/refresh token pair.
It is reused until you revoke it; nothing here refreshes it, and re-authorizing
on every launch would hit OrcaRouter's issuance cap. If OrcaRouter rejects a
stored key with `401`, the exact credential generation that made the request is
marked as needing reauthentication and you are told to sign in again — the
client does not retry or fabricate a refresh.

Revoke access at any time from
<https://www.orcarouter.ai/console/authorized-apps>.

### Origins

Authentication and inference live on two different origins.

```
auth:      https://www.orcarouter.ai        (authorize /auth, exchange /api/v1/auth/keys)
inference: https://api.orcarouter.ai/v1     (video, models, chat)
```

Both are configurable for self-hosted deployments. `ORCA_BASE_URL` sets one
shared origin; `ORCA_AUTH_BASE_URL` and `ORCA_API_BASE_URL` override the two
separately and take precedence. Remote origins must be HTTPS; plain HTTP is
accepted only for loopback development.

### Model selection

The model dropdown is built from the live `GET /v1/models` catalog, requested
with your key so it lists the models your workspace can actually call. Options
are filtered by capability, never by model name:

- text-to-video shows every model whose catalog entry declares the
  `openai-video` endpoint;
- attaching an image narrows the list to models that additionally declare
  `image` in `architecture.input_modalities`.

A model that does not declare a capability is excluded, not assumed capable. If
the catalog cannot be reached, a small verified fallback set is used and the UI
says so; a live catalog that simply lists no video model for your key is
reported honestly rather than papered over with the fallback.

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


## Verification

```bash
pytest                                              # full suite, fake credentials only
video-gen orcarouter status                         # is a credential stored, and does it work
ORCAROUTER_API_KEY=sk-orca-… pytest tests/test_orcarouter_live.py   # opt-in live checks
```

`tests/test_orcarouter_gui.py` boots the real FastAPI app, drives the page with
Playwright and writes the review evidence to `orca-evidence/` (`manifest.json`
plus the screenshots). It runs a throwaway credential store and a fake
`sk-orca-testonly-…` key, so it never touches a real account. It is a pytest
module, so the plain `pytest` run above already includes it. The evidence is
generated output, not source: it is git-ignored and produced fresh by whoever
reviews the change, so a screenshot always matches the code it was taken from.
The catalog it renders comes from `tests/fixtures/orcarouter_catalog.json`,
parsed through the same `parse_models` the live path uses, so the browser is
exercised against the real wire shape rather than a hand-built object graph.

`tests/test_orcarouter_lint.py` carries the static gates for this integration:
the secret audit (no client secret, no fixed PKCE verifier, no real `sk-orca-`
key), the endpoint audit (nothing builds a URL from the wrong `/v1/auth`
exchange origin) and the structural audit (HTTP stays inside the credential
seam). They are written against the standard library's `ast` and run in-process
under `pytest`, so they cannot pass because of a different tool version or a
`PATH` entry than the one under review. `ruff` remains the project's formatter
and linter and runs on its own:

```bash
ruff check src tests
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
│   ├── orcarouter/          # origins, credentials, PKCE connect, model catalog
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


