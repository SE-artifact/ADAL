from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


METHOD_VERSION = " "


@dataclass
class PathConfig:
    gumtree: str = ""
    joern_parse: str = ""
    joern_export: str = ""
    artifact_root: str = ""
    model_path: str = ""
    input_path: str = ""
    output_path: str = ""


@dataclass
class ModelConfig:
    provider: str = "qwen3-14b"
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = ""
    qwen_model: str = "Qwen3-14B"
    qwen_enable_thinking: bool = False
    deepseek_model: str = "deepseek-chat"
    deepseek_endpoint: str = "https://api.deepseek.com/chat/completions"
    deepseek_api_key: str = ""
    openai_model: str = "gpt-4.1"
    openai_api_key: str = ""
    timeout_seconds: int = 60
    max_retries: int = 3
    temperature: float = 0.0
    max_output_tokens: int = 4096


@dataclass
class AnalysisConfig:
    run_gumtree: bool = True
    run_joern: bool = True
    run_llm: bool = True
    random_seed: int = 43
    static_timeout_seconds: int = 60


@dataclass
class AggregationConfig:

    learning_rate: float = None
    max_iter: int = 180
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 12
    filter_l2: float = 0.05
    ranking_C: float = None

    max_pairs_per_pool: int | None = None
    target_false_acceptance_rate: float = 0.35
    alpha_grid: list[float] = field(default_factory=lambda: [
        0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.35, 1.75, 2.25,
    ])


@dataclass
class ADALConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ADALConfig":
        return cls(
            paths=PathConfig(**value.get("paths", {})),
            models=ModelConfig(**value.get("models", {})),
            analysis=AnalysisConfig(**value.get("analysis", {})),
            aggregation=AggregationConfig(**value.get("aggregation", {})),
        )

    @classmethod
    def load(cls, path: str) -> "ADALConfig":
        if not path:
            return cls()
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
