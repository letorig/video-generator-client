"""Command-line interface for unified-video-gen.""" 

from __future__ import annotations

import asyncio
from . import seedance
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .catalog import PROVIDER_INFO
from .client import VideoClient
from .config import Config

seedance.run_sync()

app = typer.Typer(
    name="video-gen",
    help="Generate videos with Seedance, Kling, MiniMax or Wan.",
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
    }


@app.command()
def generate(
    provider: str = typer.Argument(..., help="seedance | kling | minimax | wan"),
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
def models(provider: str = typer.Argument(...)) -> None:
    """List the available short model names for a provider."""
    info = PROVIDER_INFO.get(provider.lower())
    if info is None:
        console.print(f"[red]unknown provider:[/red] {provider}")
        raise typer.Exit(1)
    table = Table(title=f"{info['label']} models")
    table.add_column("Short name")
    table.add_column("API model id")
    for short, api_id in info["models"].items():
        table.add_row(short, api_id)
    console.print(table)


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
