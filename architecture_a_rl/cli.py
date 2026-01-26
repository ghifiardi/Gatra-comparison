from __future__ import annotations
import typer
from .train import train_ppo

app = typer.Typer()


@app.command()
def main(config: str = "configs/ppo.yaml", data_config: str = "configs/data.yaml") -> None:
    path = train_ppo(config, data_config)
    typer.echo(f"Saved PPO checkpoint -> {path}")


if __name__ == "__main__":
    app()
