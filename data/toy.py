from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from .schemas import RawEvent, Label

@dataclass
class ToyDataset:
    n: int = 5000
    seed: int = 42

    def generate(self) -> tuple[list[RawEvent], list[Label]]:
        rng = np.random.default_rng(self.seed)
        t0 = datetime(2025, 1, 1)
        year_minutes = 365 * 24 * 60
        step_minutes = max(1, int(year_minutes / max(1, self.n)))
        events: list[RawEvent] = []
        labels: list[Label] = []

        for i in range(self.n):
            ts = t0 + timedelta(minutes=int(i * step_minutes))
            # benign baseline
            duration = float(rng.exponential(2.0))
            bytes_sent = float(rng.lognormal(mean=6.0, sigma=0.8))
            bytes_recv = float(rng.lognormal(mean=6.2, sigma=0.9))
            port = int(rng.choice([80, 443, 22, 53, 3389, 8080], p=[.3, .3, .1, .1, .1, .1]))
            protocol = str(rng.choice(["tcp", "udp"], p=[.85, .15]))

            # inject threats with higher exfil + odd ports
            is_threat = rng.random() < 0.08
            if is_threat:
                bytes_sent *= float(rng.uniform(5, 20))
                port = int(rng.choice([4444, 1337, 5555, 3389], p=[.4, .2, .2, .2]))

            eid = f"evt_{i:07d}"
            events.append(RawEvent(
                event_id=eid,
                ts=ts,
                port=port,
                protocol=protocol,
                duration=duration,
                bytes_sent=bytes_sent,
                bytes_received=bytes_recv,
                user_id=f"user_{int(rng.integers(1, 200))}",
                host_id=f"host_{int(rng.integers(1, 100))}",
            ))
            labels.append(Label(
                event_id=eid,
                label="threat" if is_threat else "benign",
                severity=float(rng.uniform(0.4, 1.0) if is_threat else 0.0),
                source="toy",
            ))
        return events, labels
