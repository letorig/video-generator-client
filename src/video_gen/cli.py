"""Command-line interface for unified-video-gen."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .catalog import DISCOVERED_PROVIDERS, PROVIDER_INFO
from .client import VideoClient
from .config import Config
from .orcarouter import (
    ApiKeyAdapter,
    ConnectError,
    CredentialStore,
    InvalidApiKey,
    connect_loopback,
    select,
)
from .providers.orcarouter import OrcaRouterProvider

app = typer.Typer(
    name="video-gen",
    help="Generate videos with Seedance, Kling, MiniMax, Wan or OrcaRouter.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _configured(cfg: Config) -> dict[str, bool]:
    return {
        "seedance": bool(cfg.seedance_api_key),
        "kling": bool(cfg.kling_access_key and cfg.kling_secret_key),
        "minimax": bool(cfg.minimax_api_key),
        "wan": bool(cfg.dashscope_api_key),
        "orcarouter": bool(CredentialStore().current()),
    }


def _orcarouter_provider() -> OrcaRouterProvider:
    return OrcaRouterProvider(Config())


@app.command()
def generate(
    provider: str = typer.Argument(..., help="seedance | kling | minimax | wan | orcarouter"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Text prompt"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Provider model name"),
    image: Optional[str] = typer.Option(None, "--image", "-i", help="Image URL for image-to-video"),
    duration: int = typer.Option(5, "--duration", "-d", min=1, help="Duration in seconds"),
    aspect: str = typer.Option("16:9", "--aspect", help="Aspect ratio"),
    resolution: str = typer.Option("720p", "--res", help="Resolution"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save the video here"),
) -> None:
    """Submit a generation task and wait for the result."""

    async def run() -> None:
        async with VideoClient() as client:
            try:
                task = await client.generate(
                    provider,
                    prompt=prompt,
                    model=model,
                    image_url=image,
                    duration=duration,
                    aspect_ratio=aspect,
                    resolution=resolution,
                )
                console.print(f"[cyan]submitted[/cyan] provider={provider} task_id={task.task_id}")
                result = await task.wait(on_update=lambda s: console.print(f"  state={s.state}"))
            except Exception as exc:
                console.print(f"[red]error:[/red] {exc}")
                raise typer.Exit(1) from exc

            console.print(f"[green]done[/green] {result.video_url}")
            if output:
                try:
                    path = await task.download(output)
                except Exception as exc:
                    console.print(f"[red]download failed:[/red] {exc}")
                    raise typer.Exit(1) from exc
                console.print(f"[green]saved[/green] {path}")

    asyncio.run(run())


@app.command()
def status(provider: str = typer.Argument(...), task_id: str = typer.Argument(...)) -> None:
    """Check a remote task's current status."""

    async def run() -> None:
        try:
            async with VideoClient() as client:
                result = await client.status(provider, task_id)
        except Exception as exc:
            console.print(f"[red]error:[/red] {exc}")
            raise typer.Exit(1) from exc
        console.print(result.model_dump_json(indent=2))

    asyncio.run(run())


@app.command()
def providers() -> None:
    """List providers and whether their credentials are configured."""
    configured = _configured(Config())
    table = Table(title="Providers")
    table.add_column("Name")
    table.add_column("Provider")
    table.add_column("Configured")
    for name, info in PROVIDER_INFO.items():
        table.add_row(name, info["label"], "yes" if configured[name] else "no")
    console.print(table)


@app.command()
def models(
    provider: str = typer.Argument(...),
    image: bool = typer.Option(
        False, "--image", help="Only show models that accept an image input"
    ),
) -> None:
    """List the models available for a provider."""
    key = provider.lower()
    info = PROVIDER_INFO.get(key)
    if info is None:
        console.print(f"[red]unknown provider:[/red] {provider}")
        raise typer.Exit(1)

    if key in DISCOVERED_PROVIDERS:
        catalog = _orcarouter_provider().catalog()
        models_found = select(catalog.models, "video", ("image",) if image else ())
        table = Table(title=f"{info['label']} models (video)")
        table.add_column("Model id")
        table.add_column("Endpoint")
        table.add_column("Input modalities")
        for model in models_found:
            table.add_row(
                model.id,
                ", ".join(model.endpoint_types) or "-",
                ", ".join(model.input_modalities) or "-",
            )
        console.print(table)
        if catalog.degraded:
            console.print(
                f"[yellow]catalog unavailable ({catalog.detail}); showing "
                f"{catalog.source} models[/yellow]"
            )
        else:
            console.print(f"[green]live catalog[/green] {len(catalog.models)} models")
        return

    table = Table(title=f"{info['label']} models")
    table.add_column("Short name")
    table.add_column("API model id")
    for short, api_id in info["models"].items():
        table.add_row(short, api_id)
    console.print(table)


orcarouter_app = typer.Typer(
    name="orcarouter",
    help="OrcaRouter credentials: sign in, paste a key, check status, sign out.",
    no_args_is_help=True,
)
app.add_typer(orcarouter_app, name="orcarouter")


@orcarouter_app.command("login")
def orcarouter_login(
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Print the URL instead of opening a browser"
    ),
    timeout: float = typer.Option(180.0, "--timeout", help="Seconds to wait for the callback"),
) -> None:
    """Sign in with OrcaRouter (OAuth 2.0 + PKCE, loopback redirect)."""

    async def run() -> None:
        def show(url: str) -> None:
            console.print("Opening your browser to authorize with OrcaRouter…")
            console.print(f"[cyan]{url}[/cyan]")

        try:
            credential = await connect_loopback(
                CredentialStore(),
                app_name="Video Gen Client",
                timeout=timeout,
                open_browser=not no_browser,
                on_authorize_url=show,
            )
        except ConnectError as exc:
            console.print(f"[red]sign-in failed:[/red] {exc}")
            raise typer.Exit(1) from exc
        console.print(
            f"[green]connected[/green] method={credential.method} "
            f"key={credential.masked} scope={credential.scope}"
        )

    asyncio.run(run())


@orcarouter_app.command("key")
def orcarouter_key(
    api_key: str = typer.Option(
        ..., "--api-key", help="An existing sk-orca-… key", hide_input=True, prompt=True
    ),
) -> None:
    """Use an existing OrcaRouter API key instead of signing in."""
    store = CredentialStore()
    try:
        credential = ApiKeyAdapter(store).save(api_key)
    except InvalidApiKey as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]saved[/green] method={credential.method} key={credential.masked}")


@orcarouter_app.command("status")
def orcarouter_status() -> None:
    """Show the stored OrcaRouter credential and the live catalog."""
    credential = CredentialStore().current()
    if credential is None:
        console.print("No OrcaRouter credential stored.")
        console.print("Sign in with [bold]video-gen orcarouter login[/bold] or save a key with")
        console.print("[bold]video-gen orcarouter key[/bold].")
        raise typer.Exit(1)

    table = Table(title="OrcaRouter credential")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("method", credential.method)
    table.add_row("key", credential.masked)
    table.add_row("user_id", credential.user_id or "-")
    table.add_row("scope", credential.scope or "-")
    table.add_row("needs_reauth", "yes" if credential.needs_reauth else "no")
    console.print(table)

    if credential.needs_reauth:
        console.print(
            "[yellow]This credential was rejected upstream. Run "
            "'video-gen orcarouter login' to sign in again.[/yellow]"
        )
        raise typer.Exit(1)

    catalog = _orcarouter_provider().catalog()
    if catalog.degraded:
        console.print(
            f"[yellow]catalog unavailable ({catalog.detail}); using "
            f"{catalog.source}[/yellow]"
        )
    else:
        console.print(f"[green]live catalog[/green] {len(catalog.models)} models")
    console.print(f"video models: {len(select(catalog.models, 'video'))}")


@orcarouter_app.command("logout")
def orcarouter_logout() -> None:
    """Remove the stored OrcaRouter credential."""
    CredentialStore().clear()
    console.print("[green]removed[/green] the stored OrcaRouter credential")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    port: int = typer.Option(8000, "--port", min=1, max=65535, help="Port"),
    reload: bool = typer.Option(False, "--reload", help="Reload on source changes"),
) -> None:
    """Start the local FastAPI web interface."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]Web UI dependencies are missing.[/red]")
        console.print('Install them with: [bold]pip install -e ".[web]"[/bold]')
        raise typer.Exit(1)

    console.print(f"[green]starting[/green] http://{host}:{port}")
    uvicorn.run("video_gen.web:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
