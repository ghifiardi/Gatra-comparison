from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List
import numpy as np
import joblib
import torch
from architecture_a_rl.networks import Actor
from architecture_b_iforest.model import IForestModel


@dataclass
class IForestAdapter:
    preprocessor: Any
    model: IForestModel

    @classmethod
    def load(cls, bundle_path: str) -> "IForestAdapter":
        bundle = joblib.load(bundle_path)
        return cls(preprocessor=bundle["preprocessor"], model=bundle["model"])

    def score(self, features_v7: List[float]) -> float:
        x = np.array(features_v7, dtype=np.float32)[None, :]
        x = self.preprocessor.transform(x)
        return float(self.model.score(x)[0])


@dataclass
class PPOAdapter:
    actor: Actor

    @classmethod
    def load(
        cls,
        ckpt_path: str,
        state_dim: int,
        hidden_sizes: list[int],
        action_dim: int,
    ) -> "PPOAdapter":
        ckpt = torch.load(ckpt_path, map_location="cpu")
        actor = Actor(state_dim=state_dim, hidden=hidden_sizes, action_dim=action_dim)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        return cls(actor=actor)

    def action_probs(self, features_v128: List[float]) -> List[float]:
        x = torch.tensor(features_v128, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(x).squeeze(0).numpy()
        return [float(p) for p in probs]
