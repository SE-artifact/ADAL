from __future__ import annotations

import argparse
import joblib
import json
from pathlib import Path
from typing import Any

from .config import ADALConfig
from .model import ADALAggregator
from .pipeline import ADAL
from .baselines import BaselineReviewer


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    if not path:
        raise ValueError("Input path is empty.")
    values = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                values.append(json.loads(line))
    return values


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        raise ValueError("Output path is empty.")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _config(path: str) -> ADALConfig:
    return ADALConfig.load(path)


def analyze(args: argparse.Namespace) -> None:
    config = _config(args.config)
    if args.no_llm:
        config.analysis.run_llm = False
    model_path = args.model or config.paths.model_path
    aggregator = ADALAggregator.load(model_path) if model_path else None
    method = ADAL(config, aggregator)
    rows = _read_jsonl(args.input)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("project", "")), str(row.get("tool", "")), str(row.get("bug_uid", ""))
        grouped.setdefault(key, []).append(row)
    output = []
    for key in sorted(grouped):
        output.extend(method.analyze_pool(grouped[key]))
    _write_jsonl(args.output, output)


def train(args: argparse.Namespace) -> None:
    config = _config(args.config)
    rows = _read_jsonl(args.input)
    aggregator = ADAL(config).train(rows)
    aggregator.save(args.model_output)


def lopo(args: argparse.Namespace) -> None:
    config = _config(args.config)
    rows = _read_jsonl(args.input)
    predictions, models = ADAL(config).train_lopo(rows)
    _write_jsonl(args.output, predictions)
    if not args.models_output:
        raise ValueError("Model collection output path is empty.")
    target = Path(args.models_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, target)
    for project, model in models.items():
        filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in project)
        model.save(str(target.with_suffix("") / (filename + ".joblib")))
    target.with_suffix(".training.json").write_text(
        json.dumps({project: model.training_record for project, model in models.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def baseline(args: argparse.Namespace) -> None:
    reviewer = BaselineReviewer(args.method, _config(args.config).models)
    rows = _read_jsonl(args.input)
    if args.prompts_only:
        output = [reviewer.prepare(row) for row in rows]
    else:
        output = [reviewer.assess(row) for row in rows]
    _write_jsonl(args.output, output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adal")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--input", default="")
    analyze_parser.add_argument("--output", default="")
    analyze_parser.add_argument("--config", default="")
    analyze_parser.add_argument("--model", default="")
    analyze_parser.add_argument("--no-llm", action="store_true")
    analyze_parser.set_defaults(handler=analyze)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--input", default="")
    train_parser.add_argument("--model-output", default="")
    train_parser.add_argument("--config", default="")
    train_parser.set_defaults(handler=train)
    lopo_parser = subparsers.add_parser("lopo")
    lopo_parser.add_argument("--input", default="")
    lopo_parser.add_argument("--models-output", required=True)
    lopo_parser.add_argument("--output", default="")
    lopo_parser.add_argument("--config", default="")
    lopo_parser.set_defaults(handler=lopo)
    baseline_parser = subparsers.add_parser("baseline")
    baseline_parser.add_argument("--method", choices=("patch", "specrover"), required=True)
    baseline_parser.add_argument("--input", required=True)
    baseline_parser.add_argument("--output", required=True)
    baseline_parser.add_argument("--config", default="")
    baseline_parser.add_argument("--prompts-only", action="store_true")
    baseline_parser.set_defaults(handler=baseline)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
