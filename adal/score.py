from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import requests


DIMENSIONS = {
    "FC": "factual_consistency",
    "LC": "logical_clarity",
    "DHQ": "defeater_handling_quality",
    "R": "readability",
}
GRAPH_FIELDS = (
    "claim", "supporting_evidence", "rebuttals", "undercutters", "missing_evidence",
    "relations", "argument_states", "subclaim_states", "claim_state", "semantic_schema_candidates",
)
REVISION_FIELDS = (
    "semantic_schema_candidates", "rebuttal_refinements", "new_rebuttals",
    "missing_evidence_updates", "suggested_action", "action_confidence",
    "effective_action", "effective_action_confidence", "validation_adjustments",
)
GOLD_FIELDS = {"truth_label", "gold_label", "ground_truth", "developer_patch", "reference_patch"}
SCORE_PATTERN = re.compile(r"(?m)^\s*EVALUATION_SCORE:\s*([1-5])\s*$")


@dataclass
class Case:
    patch_uid: str
    evidence: Any
    report: Any
    source: str
    backbone: str


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def without_gold(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: without_gold(item) for key, item in value.items() if key not in GOLD_FIELDS}
    if isinstance(value, list):
        return [without_gold(item) for item in value]
    return value


def load_records(source: Path) -> tuple[list[tuple[dict[str, Any], str, str]], set[Path]]:
    files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}) if source.is_dir() else [source]
    records = []
    for path in files:
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() == ".jsonl":
            values = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            value = json.loads(text)
            values = value if isinstance(value, list) else [value]
        for index, value in enumerate(values, 1):
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{index}.")
            fallback = path.stem if len(values) == 1 else f"{path.stem}:{index}"
            records.append((value, str(path.resolve()), fallback))
    return records, {path.resolve() for path in files}


def prepare_case(record: dict[str, Any], source: str, fallback: str, expected_backbone: str) -> Case:
    graph = record.get("evidence_graph")
    if graph is None and "immutable_facts" in record and "claim" in record:
        graph = record
    revision = record.get("semantic_revision") or {}
    request = revision.get("request") or {}
    backbone = str(record.get("source_backbone") or record.get("backbone") or request.get("provider") or "")
    if backbone and backbone.lower() != expected_backbone.lower():
        raise ValueError(f"Source backbone mismatch in {source}: {backbone}.")
    if "report_to_evaluate" in record:
        report = without_gold(record["report_to_evaluate"])
    elif isinstance(graph, dict):
        report = {"refined_reasoning": {key: graph[key] for key in GRAPH_FIELDS if key in graph}}
        initial = record.get("initial_evidence_graph")
        if isinstance(initial, dict):
            report = {
                "initial_reasoning": {key: initial[key] for key in GRAPH_FIELDS if key in initial},
                **report,
            }
        if revision:
            report["semantic_revision"] = {key: revision[key] for key in REVISION_FIELDS if key in revision}
        if record.get("predicted_label") in {"correct", "incorrect"}:
            report["final_assessment"] = {"predicted_label": record["predicted_label"]}
        report = without_gold(report)
    else:
        raise ValueError(f"No report_to_evaluate or ADAL evidence graph in {source}.")
    evidence = record.get("input_evidence")
    if evidence is None and isinstance(graph, dict):
        evidence = {"immutable_facts": graph.get("immutable_facts")}
        buggy_method = graph.get("claim", {}).get("assessment_boundary", {}).get("buggy_method")
        if buggy_method:
            evidence["buggy_method"] = buggy_method
    if not report or not evidence:
        raise ValueError(f"Both a reasoning report and input evidence are required in {source}.")
    evidence = without_gold(evidence)
    if not evidence or evidence == {"immutable_facts": None}:
        raise ValueError(f"The input evidence is empty in {source}.")
    return Case(str(record.get("patch_uid") or fallback), evidence, report, source, backbone or expected_backbone)


def load_prompts(directory: Path) -> dict[str, str]:
    return {
        dimension: (directory / (name + "_system_prompt.txt")).read_text(encoding="utf-8-sig")
        for dimension, name in DIMENSIONS.items()
    }


def user_prompt(case: Case, dimension: str) -> str:
    value = {"report_to_evaluate": case.report}
    if dimension in {"FC", "DHQ"}:
        value = {"input_evidence": case.evidence, **value}
    return "According to the rules you are set, evaluate the following input data.\n### INPUT DATA ###\n" + json_text(value)


def parse_score(text: str) -> tuple[int, str]:
    matches = list(SCORE_PATTERN.finditer(text))
    if len(matches) != 1 or text[matches[0].end():].strip():
        raise ValueError("Expected one final EVALUATION_SCORE integer from 1 to 5.")
    reason = text[:matches[0].start()].strip()
    if not reason:
        raise ValueError("The score justification is missing.")
    return int(matches[0].group(1)), reason


def response_text(data: dict[str, Any]) -> str:
    if data.get("status") != "completed":
        raise ValueError("Incomplete judge response: " + str(data.get("status")))
    parts = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "refusal":
                raise ValueError("The judge declined to evaluate this input.")
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    text = data.get("output_text") or "".join(parts)
    if not text.strip():
        raise ValueError("The judge response contains no evaluation text.")
    return text


def evaluate(
    case: Case, dimension: str, system: str, user: str,
    args: argparse.Namespace, api_key: str,
) -> dict[str, Any]:
    result = {
        "patch_uid": case.patch_uid, "dimension": dimension, "dimension_name": DIMENSIONS[dimension],
        "source": case.source, "source_backbone": case.backbone,
        "requested_model": args.model, "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "score": None, "status": "error", "attempts": [],
    }
    for attempt in range(args.max_retries + 1):
        started = time.monotonic()
        entry = {"attempt": attempt + 1}
        retryable = True
        try:
            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
                json={
                    "model": args.model, "instructions": system, "input": user,
                    "temperature": args.temperature, "max_output_tokens": args.max_output_tokens,
                    "store": False,
                }, timeout=args.timeout,
            )
            entry["http_status"] = response.status_code
            entry["request_id"] = response.headers.get("x-request-id", "")
            retryable = response.status_code in {408, 409, 429} or response.status_code >= 500
            response.raise_for_status()
            retryable = True
            data = response.json()
            entry["response"] = data
            text = response_text(data)
            score, reason = parse_score(text)
            result.update({
                "score": score, "reason": reason, "status": "valid",
                "model": data.get("model", args.model), "response_id": data.get("id", ""),
                "usage": data.get("usage", {}),
            })
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            entry["error"] = str(exc)
        entry["seconds"] = time.monotonic() - started
        result["attempts"].append(entry)
        if result["status"] == "valid" or not retryable:
            break
        if attempt < args.max_retries:
            time.sleep(min(2 ** attempt, 8))
    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def read_previous(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid result JSONL at line {number}; preserve or repair the interrupted final line.") from exc
            result[(value["patch_uid"], value["dimension"])] = value
    return result


def write_summary(
    output: Path, cases: list[Case], results: dict[tuple[str, str], dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    dimensions = {}
    for dimension, name in DIMENSIONS.items():
        scores = [results[(case.patch_uid, dimension)]["score"] for case in cases
                  if results.get((case.patch_uid, dimension), {}).get("status") == "valid"]
        dimensions[dimension] = {
            "name": name, "valid_count": len(scores), "missing_or_failed": len(cases) - len(scores),
            "mean": sum(scores) / len(scores) if scores else None,
            "distribution": {str(score): scores.count(score) for score in range(1, 6)},
        }
    complete_count = sum(all(results.get((case.patch_uid, d), {}).get("status") == "valid" for d in DIMENSIONS) for case in cases)
    summary = {
        "model": args.model, "source_backbone": args.source_backbone,
        "patches": len(cases), "expected_count": args.expected_count,
        "complete_patches": complete_count, "complete": complete_count == len(cases),
        "dimensions": dimensions,
        "temperature": args.temperature, "max_output_tokens": args.max_output_tokens,
        "max_retries": args.max_retries,
    }
    output.with_suffix(".summary.json").write_text(json_text(summary) + "\n", encoding="utf-8")
    with output.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patch_uid", *DIMENSIONS, "complete"])
        writer.writeheader()
        for case in cases:
            row = {"patch_uid": case.patch_uid}
            row.update({dimension: results.get((case.patch_uid, dimension), {}).get("score") for dimension in DIMENSIONS})
            row["complete"] = all(row[dimension] is not None for dimension in DIMENSIONS)
            writer.writerow(row)


def api_key_from(args: argparse.Namespace) -> str:
    if args.key_file:
        key = Path(args.key_file).read_text(encoding="utf-8-sig").strip()
    else:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key and args.config:
            config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
            key = config.get("models", {}).get("openai_api_key", "").strip()
    if not key:
        raise ValueError("Set OPENAI_API_KEY or provide --key-file or --config.")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description="Score ADAL reasoning with GPT-4.1 on four five-point dimensions.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "LLM_as_judge")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--source-backbone", default="deepseek-v3")
    parser.add_argument("--expected-count", type=int, default=2280)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--key-file")
    parser.add_argument("--config")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"gpt-4\.1(?:-\d{4}-\d{2}-\d{2})?", args.model):
        parser.error("The judge must be GPT-4.1 or an explicitly selected GPT-4.1 snapshot.")
    if not 1 <= args.workers <= 32 or args.max_retries < 0 or args.expected_count < 0:
        parser.error("Invalid workers, max-retries or expected-count.")
    if args.max_output_tokens <= 0 or args.timeout <= 0:
        parser.error("Token budget and timeout must be positive.")
    if args.output.suffix.lower() != ".jsonl":
        parser.error("--output must be a JSONL filename.")
    output_paths = [args.output, args.output.with_suffix(".summary.json"), args.output.with_suffix(".csv")]
    if args.source.is_dir() and args.output.parent.resolve() == args.source.resolve():
        parser.error("Place result files outside the input directory.")
    records, input_files = load_records(args.source)
    if any(path.resolve() in input_files for path in output_paths):
        parser.error("Result files must not overwrite input files.")
    cases = [prepare_case(record, source, fallback, args.source_backbone) for record, source, fallback in records]
    del records
    if not cases or len({case.patch_uid for case in cases}) != len(cases):
        raise ValueError("The source must contain nonempty, unique patch identifiers.")
    if args.expected_count and len(cases) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} patches, found {len(cases)}.")
    prompts = load_prompts(args.prompt_dir)
    if any(path.exists() for path in output_paths) and not args.resume:
        raise FileExistsError("Result files exist; use --resume to continue the evaluation.")
    results = read_previous(args.output) if args.resume and args.output.exists() else {}
    pairs = {(case.patch_uid, dimension) for case in cases for dimension in DIMENSIONS}
    if set(results) - pairs:
        raise ValueError("Existing output belongs to a different patch set.")
    jobs = []
    for case in cases:
        for dimension in DIMENSIONS:
            previous = results.get((case.patch_uid, dimension))
            if previous and previous.get("status") == "valid":
                continue
            jobs.append((case, dimension))
    api_key = api_key_from(args) if jobs else ""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_summary(args.output, cases, results, args)
    last_summary = time.monotonic()
    iterator = iter(jobs)
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = set()
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < args.workers * 2:
                job = next(iterator, None)
                if job is None:
                    exhausted = True
                    break
                case, dimension = job
                pending.add(pool.submit(
                    evaluate, case, dimension, prompts[dimension], user_prompt(case, dimension), args, api_key,
                ))
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                results[(result["patch_uid"], result["dimension"])] = result
            if time.monotonic() - last_summary >= 5:
                write_summary(args.output, cases, results, args)
                last_summary = time.monotonic()
            valid = sum(value.get("status") == "valid" for value in results.values())
            print(f"Completed scores: {valid}/{len(cases) * len(DIMENSIONS)}", file=sys.stderr)
    write_summary(args.output, cases, results, args)
    if any(results.get(pair, {}).get("status") != "valid" for pair in pairs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
