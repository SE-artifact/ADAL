from __future__ import annotations

from dataclasses import dataclass, asdict
import difflib
import time
from typing import Any

from .baseline_prompts import build_baseline_prompt
from .config import ModelConfig
from .model_client import ModelClient, parse_json_object


def _text(record: dict[str, Any], name: str) -> str:
    value = record.get(name, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(name + " must contain text.")
    return value


def _file_map(record: dict[str, Any], name: str) -> dict[str, str]:
    values = record.get(name) or {}
    if not isinstance(values, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in values.items()):
        raise ValueError(name + " must map source filenames to source text.")
    return values


@dataclass
class ReviewArtifacts:
    triggering_test: str
    failure_message: str
    stack_trace: str
    buggy_context: str
    patch_diff: str

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "ReviewArtifacts":
        old_files = _file_map(record, "buggy_files")
        new_files = _file_map(record, "patched_files")
        if not old_files and not new_files:
            name = _text(record, "buggy_file") or "Input.java"
            old, new = _text(record, "buggy_source"), _text(record, "patched_source")
            if old or new:
                old_files[name], new_files[name] = old, new
        context = _text(record, "buggy_context")
        if not context:
            context = "\n\n".join("[File] " + name + "\n" + source for name, source in sorted(old_files.items()))
            method = _text(record, "buggy_method")
            if method and method not in context:
                context += ("\n\n" if context else "") + "[Buggy Method]\n" + method
        diff = _text(record, "patch_diff")
        if not diff:
            diffs = []
            for name in sorted(set(old_files) | set(new_files)):
                diffs.extend(difflib.unified_diff(
                    old_files.get(name, "").splitlines(), new_files.get(name, "").splitlines(),
                    fromfile="a/" + name, tofile="b/" + name, lineterm="",
                ))
            diff = "\n".join(diffs)
        if not diff:
            raise ValueError("Supply patch_diff or differing buggy/patched source text.")
        return cls(
            triggering_test=_text(record, "triggering_test"),
            failure_message=_text(record, "failure_message") or _text(record, "failure_context"),
            stack_trace=_text(record, "stack_trace"), buggy_context=context, patch_diff=diff,
        )


class BaselineReviewer:
    def __init__(self, method: str, config: ModelConfig | None = None):
        self.method = method.lower()
        if self.method not in {"patch", "specrover"}:
            raise ValueError("Baseline must be patch or specrover.")
        self.client = ModelClient(config or ModelConfig())

    def prepare(self, record: dict[str, Any]) -> dict[str, Any]:
        artifacts = asdict(ReviewArtifacts.from_record(record))
        system, user = build_baseline_prompt(self.method, artifacts)
        return {
            "method": "PATCH" if self.method == "patch" else "SpecRover",
            "patch_uid": str(record.get("patch_uid", "")),
            "bug_uid": str(record.get("bug_uid", "")),
            "project": str(record.get("project", "")), "tool": str(record.get("tool", "")),
            "patch_rank": record.get("patch_rank"),
            "review_protocol": "single_patch_review" if self.method == "patch" else "failure_then_patch_review_in_one_request",
            "source_figure": 6 if self.method == "patch" else 7,
            "model_settings": self.client.settings(),
            "system_prompt": system, "user_prompt": user,
            "missing_artifacts": [name for name, value in artifacts.items() if not value],
        }

    def assess(self, record: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        result = {
            "method": "PATCH" if self.method == "patch" else "SpecRover",
            "patch_uid": str(record.get("patch_uid", "")),
            "predicted_label": None, "status": "pending", "model_calls": 0, "response": None,
        }
        try:
            result.update(self.prepare(record))
            result["model_calls"] = 1
            response = self.client.complete(result["system_prompt"], result["user_prompt"])
            result["response"] = response.to_dict()
            value = parse_json_object(response)
            fields = {"reason", "label"} if self.method == "patch" else {"test_assessment", "patch_assessment", "label"}
            if set(value) != fields:
                raise ValueError("Response fields do not match the appendix prompt.")
            if value["label"] not in {"correct", "incorrect"}:
                raise ValueError("Baseline label must be correct or incorrect.")
            if any(not isinstance(value[field], str) or not value[field].strip() for field in fields - {"label"}):
                raise ValueError("Every assessment field must contain text.")
            result["assessment"] = value
            result["predicted_label"] = value["label"]
            result["status"] = "valid"
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
        result["seconds"] = time.perf_counter() - started
        return result


class PATCHReviewer(BaselineReviewer):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__("patch", config)


class SpecRoverReviewer(BaselineReviewer):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__("specrover", config)
