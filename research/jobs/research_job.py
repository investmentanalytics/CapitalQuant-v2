from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class ResearchJob:
    job_id: str
    asset: str
    timeframe: str
    objective: str
    regime: Optional[str] = None
    status: str = "queued"
    progress: float = 0.0
    message: str = "En cola"
    error: Optional[str] = None
    result: Optional[dict] = None
    # Candidatos del Hall of Fame visibles mientras la generación avanza.
    # Se mantienen como datos simples para que la UI pueda pintarlos en vivo.
    live_candidates: Optional[list[dict]] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d
