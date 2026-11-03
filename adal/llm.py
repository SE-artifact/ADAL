from __future__ import annotations

from copy import deepcopy
import json
import time
from typing import Any

from .model_client import ModelClient, parse_json_object

from .config import ModelConfig
from .prompt import (
    ALLOWED_REBUTTAL_OPERATORS,
    ALLOWED_REPAIR_SCHEMAS,
    SYSTEM_PROMPT,
)


ACTIONS = {"no_change", "ranking_boost", "promote_to_candidate"}
CONFIDENCE = {"low", "medium", "high"}
REBUTTAL_STATES = {"supported", "undercut", "contradicted", "unresolved"}
SCHEMA_STATES = {"supported", "unsupported", "uncertain"}
REQUIRED_FIELDS = {
    "patch_uid",
    "semantic_schema_candidates",
    "rebuttal_refinements",
    "new_rebuttals",
    "missing_evidence_updates",
    "suggested_action",
    "action_confidence",
}


def no_change(patch_uid: str, reason: str = "") -> dict[str, Any]:
    value = {
        "patch_uid": patch_uid,
        "semantic_schema_candidates": [],
        "rebuttal_refinements": [],
        "new_rebuttals": [],
        "missing_evidence_updates": [],
        "suggested_action": "no_change",
        "action_confidence": "low",
        "effective_action": "no_change",
        "effective_action_confidence": "low",
        "raw_response": None,
        "validated": False,
    }
    if reason:
        value["failure_reason"] = reason
    return value


def _extract_json(text: str) -> dict[str, Any]:
    value = json.loads(text.strip())
    if not isinstance(value, dict):
        raise ValueError("The model output must be one JSON object.")
    return value


class LanguageModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = ModelClient(config)
        self._last_transport_response = None

    def revise(self, patch_uid: str, user_prompt: str, graph: dict[str, Any]) -> dict[str, Any]:
        last_error = ""
        attempts = []
        for attempt in range(self.config.max_retries + 1):
            started = time.perf_counter()
            response_text = ""
            parsed = None
            self._last_transport_response = None
            try:
                response_text = self._call(user_prompt)
                parsed = _extract_json(response_text)
                result = validate_revision(parsed, patch_uid, user_prompt, graph)
                attempts.append({
                    "attempt": attempt + 1, "response_text": response_text,
                    "parsed_response": parsed, "valid": True,
                    "transport_response": self._last_transport_response,
                    "seconds": time.perf_counter() - started,
                })
                result["attempts"] = attempts
                result["request"] = self._request_record(user_prompt)
                return result
            except Exception as exc:
                last_error = str(exc)
                attempts.append({
                    "attempt": attempt + 1, "response_text": response_text,
                    "parsed_response": parsed, "valid": False, "error": last_error,
                    "transport_response": self._last_transport_response,
                    "seconds": time.perf_counter() - started,
                })
        result = no_change(patch_uid, last_error)
        result["attempts"] = attempts
        result["raw_response"] = attempts[-1]["parsed_response"] if attempts else None
        result["request"] = self._request_record(user_prompt)
        return result

    def _request_record(self, user_prompt: str) -> dict[str, Any]:
        return {
            **self.client.settings(),
            "timeout_seconds": self.config.timeout_seconds,
            "system_prompt": SYSTEM_PROMPT, "user_prompt": user_prompt,
        }

    def _call(self, user_prompt: str) -> str:
        response = self.client.complete(SYSTEM_PROMPT, user_prompt)
        self._last_transport_response = response.to_dict()
        parse_json_object(response)
        return response.content


def _grounding_valid(values: Any, user_prompt: str) -> bool:
    return (
        isinstance(values, list) and bool(values)
        and all(isinstance(value, str) and value.strip()
                and (value in user_prompt or json.dumps(value, ensure_ascii=False)[1:-1] in user_prompt)
                for value in values)
    )


def should_revise(graph: dict[str, Any]) -> bool:
    unresolved = any(item["status"] == "unresolved" for item in graph["rebuttals"])
    weak_support = (
        graph.get("claim_state") == "unresolved"
        or not any(item["status"] == "supported" and item["target_id"] == "M" for item in graph["supporting_evidence"])
    )
    missing = any(item["new_status"] != "resolved" and item["criticality"] == "high" for item in graph["missing_evidence"])
    return unresolved or weak_support or missing


def _predicate_map(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in graph["immutable_facts"]["predicates"]}


def _cited_predicates(values: list[str], graph: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for predicate in graph["immutable_facts"]["predicates"]:
        representations = {
            predicate["name"], predicate.get("id", "P:" + predicate["name"]),
            json.dumps(predicate, ensure_ascii=False),
            json.dumps(predicate, ensure_ascii=False, sort_keys=True),
            json.dumps(predicate, ensure_ascii=False, sort_keys=True, indent=2),
        }
        if any(value in representations for value in values):
            output.append(predicate)
    return output


def _grounding_roles(values: list[str], graph: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    immutable = graph["immutable_facts"]
    diff = immutable["diff_ast_facts"]
    spans = [
        diff.get("unified_diff", ""),
        immutable["failure_predicates"].get("observed_failure", ""),
        graph["claim"]["assessment_boundary"].get("buggy_method", ""),
        *diff.get("removed_lines", []), *diff.get("added_lines", []),
    ]
    # Names/IDs cite predicates; a separate exact code/failure span is also required.
    cited = _cited_predicates(values, graph)
    identifiers = {p["name"] for p in cited} | {p.get("id", "") for p in cited}
    has_span = any(
        value not in identifiers and any(value in span for span in spans if span)
        for value in values
    )
    return cited, has_span


def _list_of_objects(value: dict[str, Any], field: str) -> list[dict[str, Any]]:
    items = value[field]
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError(field + " must be an array of objects.")
    return deepcopy(items)


def _require(item: dict[str, Any], keys: set[str]) -> None:
    if set(item) != keys:
        raise ValueError("Unexpected or missing fields: " + repr(sorted(set(item) ^ keys)))


def _grounded(item: dict[str, Any], user_prompt: str, graph: dict[str, Any]) -> list[dict[str, Any]]:
    values = item["grounding"]
    if not _grounding_valid(values, user_prompt):
        raise ValueError("Grounding must quote exact input strings.")
    cited, has_span = _grounding_roles(values, graph)
    if not cited or not has_span:
        raise ValueError("Each update needs an input predicate and an exact code/failure span.")
    return cited


def validate_revision(
    value: dict[str, Any], patch_uid: str, user_prompt: str, graph: dict[str, Any],
) -> dict[str, Any]:
    from .evidence import SCHEMA_PREDICATES
    _require(value, REQUIRED_FIELDS)
    if value["patch_uid"] != patch_uid:
        raise ValueError("patch_uid does not match the requested patch.")
    if value["suggested_action"] not in ACTIONS or value["action_confidence"] not in CONFIDENCE:
        raise ValueError("Invalid revision action or confidence.")
    predicates = _predicate_map(graph)
    existing = {item["id"]: item for item in graph["rebuttals"]}

    editable = {uid: item for uid, item in existing.items() if item["status"] in {"supported", "unresolved"}}
    targets = {"C0"} | {item["id"] for item in graph["claim"]["subclaims"]}
    targets |= {item["id"] for item in graph["relations"]}
    schemas = _list_of_objects(value, "semantic_schema_candidates")
    refinements = _list_of_objects(value, "rebuttal_refinements")
    new_rebuttals = _list_of_objects(value, "new_rebuttals")
    missing = _list_of_objects(value, "missing_evidence_updates")
    adjustments = []
    for item in schemas:
        _require(item, {"schema", "status", "confidence", "grounding"})
        if item["schema"] not in ALLOWED_REPAIR_SCHEMAS or item["status"] not in SCHEMA_STATES or item["confidence"] not in CONFIDENCE:
            raise ValueError("Invalid repair-schema candidate.")
        cited = _grounded(item, user_prompt, graph)
        if item["status"] == "supported":
            names = SCHEMA_PREDICATES[item["schema"]]
            cited_names = {predicate["name"] for predicate in cited}
            if not all(predicates.get(name, {}).get("value") == 1 for name in names) or not cited_names.intersection(names):
                item["status"] = "uncertain"
                adjustments.append("Unsupported schema retained as uncertain: " + item["schema"])
    seen_refinements = set()
    for item in refinements:
        _require(item, {"rebuttal_id", "original_status", "new_status", "severity", "reason", "confidence", "grounding"})
        original = editable.get(item["rebuttal_id"])
        if original is None or item["rebuttal_id"] in seen_refinements:
            raise ValueError("Unknown, non-editable or duplicate rebuttal.")
        seen_refinements.add(item["rebuttal_id"])
        if item["original_status"] != original["status"] or item["severity"] != original["severity"]:
            raise ValueError("Rebuttal state/severity must match the supplied argument.")
        if item["new_status"] not in REBUTTAL_STATES or item["confidence"] not in CONFIDENCE or not isinstance(item["reason"], str):
            raise ValueError("Invalid rebuttal revision.")
        cited = _grounded(item, user_prompt, graph)
        required = set(original.get("safe_predicate_names", original["predicate_names"]))
        evidence = [predicate for predicate in cited if predicate["name"] in required]
        if item["new_status"] == "contradicted" and not (
            evidence and all(predicates.get(name, {}).get("value") == 1 for name in required)
        ):
            item["new_status"] = "unresolved"
            adjustments.append("No immutable disproof for " + item["rebuttal_id"])
        if item["new_status"] == "supported" and not any(predicate["value"] == 0 for predicate in evidence):
            item["new_status"] = "unresolved"
            adjustments.append("Unsupported objection kept unresolved: " + item["rebuttal_id"])
        item["predicate_ids"] = [predicate.get("id", "P:" + predicate["name"]) for predicate in cited]
    seen_new = set()
    for item in new_rebuttals:
        _require(item, {"operator", "target_id", "severity", "hypothesis", "grounding"})
        if item["operator"] not in ALLOWED_REBUTTAL_OPERATORS or item["target_id"] not in targets:
            raise ValueError("New rebuttal must target an existing claim or inference relation.")
        if item["severity"] not in CONFIDENCE or not isinstance(item["hypothesis"], str):
            raise ValueError("Invalid new-rebuttal severity or hypothesis.")
        cited = _grounded(item, user_prompt, graph)
        signature = (item["operator"], item["target_id"], item["hypothesis"])
        if signature in seen_new:
            raise ValueError("Duplicate new rebuttal.")
        seen_new.add(signature)
        item["predicate_names"] = [predicate["name"] for predicate in cited]
        item["initial_status"] = "unresolved"
    existing_missing = {item["id"]: item for item in graph["missing_evidence"]}
    seen_missing = set()
    for item in missing:
        _require(item, {"missing_id", "new_status", "criticality", "reason", "grounding"})
        if not isinstance(item["missing_id"], str) or item["missing_id"] in seen_missing:
            raise ValueError("Invalid or duplicate missing-evidence ID.")
        seen_missing.add(item["missing_id"])
        if item["new_status"] not in {"resolved", "unresolved"} or item["criticality"] not in CONFIDENCE or not isinstance(item["reason"], str):
            raise ValueError("Invalid missing-evidence update.")
        cited = _grounded(item, user_prompt, graph)
        if item["new_status"] == "resolved":
            original = existing_missing.get(item["missing_id"])
            required = original.get("predicate_name") if original else None
            aliases = {"external_caller_behavior": "external_behavior_preserved"}
            required = aliases.get(required, required)
            # Known false is also a resolved question; it never supplies positive support.
            if not required or not any(p["name"] == required and p["value"] in (0, 1) for p in cited):
                item["new_status"] = "unresolved"
                adjustments.append("Required immutable fact unavailable for " + item["missing_id"])
    grounded_rebuttal = any(item["severity"] in {"medium", "high"} and item["new_status"] in {"undercut", "contradicted"} for item in refinements)
    supported_schema = any(item["status"] == "supported" for item in schemas)
    contradicted = any(item["severity"] in {"medium", "high"} and item["new_status"] == "contradicted" for item in refinements)
    new_high = any(item["severity"] == "high" for item in new_rebuttals)
    effective = value["suggested_action"]
    if effective == "ranking_boost" and not (grounded_rebuttal or supported_schema):
        effective = "no_change"
    if effective == "promote_to_candidate" and not ((grounded_rebuttal or supported_schema) and contradicted and not new_high):
        effective = "ranking_boost" if grounded_rebuttal or supported_schema else "no_change"

    return {
        "patch_uid": patch_uid, "semantic_schema_candidates": schemas,
        "rebuttal_refinements": refinements, "new_rebuttals": new_rebuttals,
        "missing_evidence_updates": missing,
        "suggested_action": value["suggested_action"], "action_confidence": value["action_confidence"],
        "effective_action": effective, "effective_action_confidence": value["action_confidence"] if effective != "no_change" else "low",
        "raw_response": deepcopy(value), "validation_adjustments": adjustments, "validated": True,
    }
