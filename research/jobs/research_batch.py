from __future__ import annotations
from dataclasses import dataclass, field
from .research_job import ResearchJob

@dataclass
class ResearchBatch:
    batch_id: str
    jobs: list[ResearchJob] = field(default_factory=list)
    status: str = "queued"

    @property
    def total(self) -> int:
        return len(self.jobs)

    @property
    def completed(self) -> int:
        return sum(j.status == "completed" for j in self.jobs)

    @property
    def failed(self) -> int:
        return sum(j.status == "failed" for j in self.jobs)

    @property
    def running(self) -> int:
        return sum(j.status == "running" for j in self.jobs)

    def progress(self) -> float:
        return (self.completed + self.failed) / self.total if self.total else 1.0
