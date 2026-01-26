from __future__ import annotations
import typer
from .contract import InferenceRequest
from .router import handle_request

app = typer.Typer()


@app.command()
def main(config: str = "configs/serving.yaml", data_config: str = "configs/data.yaml") -> None:
    typer.echo("Serving stub. Wire FastAPI later if needed.")
    typer.echo(f"Config: {config}")


@app.command()
def demo() -> None:
    class DummyIForest:
        def score(self, features_v7):
            return 0.42

    class DummyPPO:
        def action_probs(self, features_v128):
            return [0.1, 0.2, 0.3, 0.4]

    req_ok = InferenceRequest(
        event_id="evt_demo_ok",
        ts="2025-01-01T00:00:00",
        features_v7=[1.0, 2.0, 3.0, 80.0, 1.0, 12.0, 3.0],
        features_v128=[0.0] * 128,
        request_id="req_ok",
    )
    status_ok, payload_ok = handle_request(
        req_ok,
        mode="shadow",
        active="iforest",
        iforest=DummyIForest(),
        ppo=DummyPPO(),
    )
    typer.echo(f"OK status={status_ok} payload={payload_ok}")

    req_bad = InferenceRequest(
        event_id="evt_demo_bad",
        ts="2025-01-01T00:00:00",
        features_v7=[1.0, 2.0],
        features_v128=[0.0] * 128,
        request_id="req_bad",
    )
    status_bad, payload_bad = handle_request(
        req_bad,
        mode="shadow",
        active="iforest",
        iforest=DummyIForest(),
        ppo=DummyPPO(),
    )
    typer.echo(f"BAD status={status_bad} payload={payload_bad}")


if __name__ == "__main__":
    app()
