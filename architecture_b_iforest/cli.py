from __future__ import annotations
import typer
from .train import train_iforest

app = typer.Typer()

@app.command()
def main(config: str = "configs/iforest.yaml", data_config: str = "configs/data.yaml") -> None:
    path = train_iforest(config, data_config)
    typer.echo(f"Saved IF bundle -> {path}")

if __name__ == "__main__":
    app()
