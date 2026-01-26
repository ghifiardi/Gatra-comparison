from __future__ import annotations
import typer
from .head_to_head import run_head_to_head

app = typer.Typer()

@app.command()
def main(
    eval_config: str = "configs/eval.yaml",
    data_config: str = "configs/data.yaml",
    iforest_config: str = "configs/iforest.yaml",
    ppo_config: str = "configs/ppo.yaml",
) -> None:
    path = run_head_to_head(eval_config, data_config, iforest_config, ppo_config)
    typer.echo(f"Saved report -> {path}")

if __name__ == "__main__":
    app()
