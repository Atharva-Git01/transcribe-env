"""Run context shared by stages. Stages read params ONLY from their plan entry; data flows through this object
and through files in work/."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .records import Custody, Decisions


@dataclass
class Track:
    index: int                      # 0-based
    label: str                      # speaker label for this track: "SPK?" (unknown) or SPK1/SPK2 (true stereo)
    wav48: Path | None = None       # 48 kHz mono after convert (input to denoise)
    wav16: Path | None = None       # 16 kHz mono that recognition uses (after denoise if any)
    speech: list = field(default_factory=list)   # [(start_s, end_s), …] from the vad stage
    language: str | None = None


@dataclass
class Ctx:
    run_dir: Path
    input_file: Path
    passport: dict
    plan: dict
    log: Decisions
    custody: Custody
    tracks: list[Track] = field(default_factory=list)
    language: str | None = None            # file-level language decision
    language_source: str = ""
    segment_languages: list = field(default_factory=list)  # per VAD group when lid.per_segment
    segments: list[dict] = field(default_factory=list)     # recognised segments (all tracks), time-ordered
    turns: list[dict] = field(default_factory=list)        # diarization turns {start, end, speaker} (phase 4)
    engines_used: list[dict] = field(default_factory=list)
    review: list[dict] = field(default_factory=list)       # items for out/review_list.md
    verdict: str = "PASS"
    verdict_reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def work(self) -> Path:
        return self.run_dir / "work"

    @property
    def out(self) -> Path:
        return self.run_dir / "out"

    def stage_params(self, name: str) -> dict:
        for s in self.plan["stages"]:
            if s["name"] == name:
                return s["params"]
        return {}

    def has_stage(self, name: str) -> bool:
        return any(s["name"] == name for s in self.plan["stages"])
