from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
import re
from typing import Any

from .types import PatchInput


SEVERITY_WEIGHT = {"low": 4.0, "medium": 10.0, "high": 20.0}
ARGUMENT_STRENGTH = {"low": 0.35, "medium": 0.65, "high": 1.0}
UNDERCUT_FACTOR = {"low": 0.15, "medium": 0.35, "high": 0.60}
# The paper defines the ordering and formula; numeric defaults are implementation details.
SUBCLAIMS = {
    "F": "The observed failure and triggering behavior are characterized.",
    "T": "The modification targets the implicated program entity or behavior.",
    "M": "The proposed repair mechanism explains the observed failure.",
    "P": "Required returns, non-failing paths, dependencies and side effects are preserved.",
    "B": "The assessment is bounded by the available code and behavioral evidence.",
}
# Path-wide obligations are subclaim-scoped; only expression-local evidence has level 2.
SCOPE_LEVEL = {"claim": 0, "subclaim": 1, "expression": 2, "execution_path": 1}

# Table 1: operator, activation, ALL targets, severity, name, predicate, hypothesis.
REBUTTAL_LIBRARY = (
    ("RO1", {"condition_change", "null_check_change"}, ("M", "P"), "high",
     "Predicate Polarity", "condition_consistent", "The condition may invert the intended predicate and skip required behavior."),
    ("RO2", {"guard_insertion", "early_return_or_default_return"}, ("M", "P"), "medium",
     "Guard Overgeneralization", "guard_preserves_valid_paths", "The patch may bypass valid computation to avoid the failing path."),
    ("RO3", {"exception_handling_change", "deletion"}, ("F", "M"), "medium",
     "Symptom Suppression", "failure_explained", "The patch may hide the observed symptom without addressing its cause."),
    ("RO4", {"constant_change"}, ("M", "B"), "medium",
     "Trigger Overfitting", "not_trigger_specific", "The new value may specialize the patch to triggering values."),
    ("RO5", {"api_call_change", "method_signature_change", "method_change"}, ("P", "B"), "medium",
     "Regression Surface", "external_behavior_preserved", "The patch may affect callers outside the available tests."),
    ("RO6", {"assignment_or_value_change", "api_call_change", "receiver_change"}, ("M", "P"), "high",
     "Value Substitution", "semantic_role_preserved", "The new expression may alter the original data dependency or semantic role."),
    ("RO7", {"return_change", "early_return_or_default_return", "guard_insertion"}, ("P",), "medium",
     "Branch Control Impact", "required_computation_preserved", "The changed branch may skip required computation."),
    ("RO8", {"deletion"}, ("P",), "medium",
     "Side Effect Deletion", "side_effects_preserved", "The deleted code may contain required side effects."),
    ("RO9", {"assignment_or_value_change", "expression_change", "condition_change"}, ("M", "P"), "medium",
     "Expression Completeness", "operands_preserved", "The new expression may omit required operands."),
    ("RO10", {"condition_change", "guard_insertion", "null_check_change"}, ("M", "P"), "high",
     "Path Feasibility", "repair_path_feasible", "The new condition may be infeasible or fail to control the repair path."),
    ("RO11", {"return_change", "early_return_or_default_return", "deletion"}, ("P",), "medium",
     "Return Contract Mismatch", "return_contract_aligned", "The new return behavior may violate the local method contract."),
)
SCHEMA_PREDICATES = {
    "null_or_guard_repair": ("null_guard_before_dereference", "failure_explained", "repair_path_feasible"),
    "guarded_exception_contract_repair": ("return_contract_aligned", "failure_explained"),
    "early_return_contract_repair": ("return_contract_aligned", "required_computation_preserved"),
    "loop_bound_role_repair": ("condition_consistent", "repair_path_feasible"),
    "api_result_role_repair": ("semantic_role_preserved",),
    "constructor_context_repair": ("semantic_role_preserved",),
    "expression_role_repair": ("operands_preserved", "semantic_role_preserved"),
    "side_effect_free_deletion_repair": ("side_effects_preserved",),
}
SCHEMA_ORDER = tuple(SCHEMA_PREDICATES)


def _identifiers(text: str) -> set[str]:
    keywords = {
        "if", "else", "for", "while", "return", "new", "null", "true", "false",
        "int", "long", "double", "float", "boolean", "char", "byte", "short",
        "class", "public", "protected", "private", "static", "final", "void",
    }
    return {
        token for token in re.findall(r"\b[A-Za-z_]\w*\b", text)
        if token not in keywords
    }


def _cpg_value(cpg: dict[str, Any], name: str) -> Any:
    for item in cpg.get("predicates", []):
        if item.get("name") == name:
            return item.get("value")
    if name in cpg.get("positive_predicates", []):
        return 1
    if name in cpg.get("negative_predicates", []):
        return 0
    return None


def _conjunction(values: list[Any]) -> int | None:
    if any(value == 0 for value in values):
        return 0
    return 1 if values and all(value == 1 for value in values) else None


def _local_null_guard(patch: PatchInput, diff: dict[str, Any]) -> int | None:
    if not {"guard_insertion", "null_check_change"} & set(diff.get("patch_types", [])):
        return None
    # Narrow source rule: exactly the guarded non-null operand is dereferenced in
    # the next statement. General control-flow and other callers remain unknown.
    for _, _, source in patch.source_pairs():
        for match in re.finditer(
            r"if\s*\(\s*([\w.$\[\]]+)\s*!=\s*null\s*\)\s*\{?\s*([^;{}]+;)", source):
            operand, statement = match.groups()
            edited = "\n".join(diff.get("added_lines", []))
            if re.search(re.escape(operand) + r"\s*!=\s*null", edited) and re.search(re.escape(operand) + r"\s*\.", statement):
                return 1
    return None


def _semantic_facts(
    patch: PatchInput, diff: dict[str, Any], cpg: dict[str, Any], failure: dict[str, Any],
) -> dict[str, Any]:
    removed, added = "\n".join(diff.get("removed_lines", [])), "\n".join(diff.get("added_lines", []))
    old_ids, new_ids = _identifiers(removed), _identifiers(added)
    old_calls, new_calls = set(diff.get("removed_calls", [])), set(diff.get("added_calls", []))
    types = set(diff.get("patch_types", []))
    def relation(*names: str) -> int | None:
        for name in names:
            value = _cpg_value(cpg, name)
            if value in (0, 1):
                return int(value)
        return None
    null_guard = relation("null_guard_before_dereference")
    if null_guard is None:
        null_guard = _local_null_guard(patch, diff)
    failure_explained = relation("failure_explained")
    if failure_explained is None and failure.get("has_null_failure") and null_guard == 1:
        failure_explained = 1
    trigger_specific = bool(
        (set(diff.get("added_literals", [])) - set(diff.get("removed_literals", [])))
        & set(re.findall(r"\w+", patch.triggering_test))
    )
    facts = {
        "identifier_overlap_ratio": len(old_ids & new_ids) / len(old_ids | new_ids) if old_ids | new_ids else 1.0,
        "null_guard_before_dereference": null_guard,
        "failure_explained": failure_explained,
        "condition_consistent": relation("condition_consistent", "loop_condition_feasible"),
        "guard_preserves_valid_paths": relation("guard_preserves_valid_paths"),
        # No matching literal is NOT evidence that the repair generalizes.
        "not_trigger_specific": relation("not_trigger_specific"),
        "external_behavior_preserved": relation("external_behavior_preserved"),
        "semantic_role_preserved": relation("semantic_role_preserved", "receiver_dependency_preserved"),
        "required_computation_preserved": relation("required_computation_preserved"),
        "side_effects_preserved": relation("side_effects_preserved"),
        "operands_preserved": relation("operands_preserved", "operand_preserved"),
        "repair_path_feasible": relation("repair_path_feasible", "loop_condition_feasible"),
        "return_contract_aligned": relation("return_contract_aligned"),
        "receiver_delta": int(old_calls != new_calls),
        "variable_to_constant": int(bool(old_ids - new_ids) and bool(set(diff.get("added_literals", [])) - set(diff.get("removed_literals", [])))),
        "side_effect_deleted": int("deletion" in types and bool(old_calls - new_calls)),
        "condition_contradiction": int(relation("condition_consistent", "loop_condition_feasible") == 0),
        "controlled_region_has_early_exit": int("early_return_or_default_return" in types),
        "controlled_region_has_later_dereference": int(null_guard == 1),
        "changed_call_matches_trigger": int(bool((new_calls - old_calls) & _identifiers(patch.observed_failure()))),
        "trigger_literal_overlap": int(trigger_specific),
        "method_public_or_protected": int(bool(re.search(r"\b(?:public|protected)\b", patch.buggy_method))),
    }
    facts["schema_support"] = _schema_support(diff, facts, patch)
    facts["semantic_schema_supports"] = [
        item["schema"] for item in facts["schema_support"] if item["condition_value"] == 1
    ]
    return facts


def _schema_support(
    diff: dict[str, Any], facts: dict[str, Any], patch: PatchInput,
) -> list[dict[str, Any]]:
    types = set(diff.get("patch_types", []))
    added = "\n".join(diff.get("added_lines", []))
    activation = {
        "null_or_guard_repair": bool(types & {"guard_insertion", "null_check_change"}),
        "guarded_exception_contract_repair": "exception_handling_change" in types,
        "early_return_contract_repair": "early_return_or_default_return" in types,
        "loop_bound_role_repair": "condition_change" in types,
        "api_result_role_repair": "api_call_change" in types,
        "constructor_context_repair": bool(re.search(r"\b(?:new\s+\w+|super|this)\s*\(", added)),
        "expression_role_repair": bool(types & {"assignment_or_value_change", "expression_change", "condition_change"}),
        "side_effect_free_deletion_repair": "deletion" in types,
    }
    output = []
    for order, schema in enumerate(SCHEMA_ORDER):
        if not activation[schema]:
            continue
        names = SCHEMA_PREDICATES[schema]
        values = [facts.get(name) for name in names]
        output.append({
            "schema": schema, "predicate_names": list(names),
            "condition_value": _conjunction(values),
            "support_count": sum(value == 1 for value in values),
            "scope_level": 2, "tie_order": order,
        })
    return sorted(output, key=lambda item: (-item["support_count"], item["tie_order"]))


def _predicate(
    name: str, value: Any, source: str, scope: str, grounding: list[str],
    measurement: Any = None, group: str = "relation",
) -> dict[str, Any]:
    item = {
        "id": "P:" + name, "name": name, "value": value, "source": source,
        "scope": scope, "group": group, "grounding": [span for span in grounding if span],
    }
    if measurement is not None:
        item["measurement"] = measurement
    return item


def _predicates(
    patch: PatchInput, diff: dict[str, Any], cpg: dict[str, Any],
    failure: dict[str, Any], semantic: dict[str, Any],
) -> list[dict[str, Any]]:
    spans = diff.get("removed_lines", []) + diff.get("added_lines", [])
    changed_lines = {
        line for item in diff.get("files", []) for line in item.get("old_changed_line_numbers", [])
        if not patch.buggy_file or item["file"] == patch.buggy_file
    }
    touches_line = int(patch.buggy_line in changed_lines) if patch.buggy_line and changed_lines else None
    method_match = bool(patch.buggy_method and any(
        line.strip() and line.strip() in patch.buggy_method for line in diff.get("removed_lines", [])
    ))
    local_source = patch.buggy_method or "\n".join(old for _, old, _ in patch.source_pairs())
    values = [
        _predicate("triggering_test_available", int(bool(patch.triggering_test)), "failure", "subclaim", [patch.triggering_test], group="failure"),
        _predicate("failure_observed", 1 if patch.observed_failure() else None, "failure", "subclaim", [patch.observed_failure()], group="failure"),
        _predicate("modified_method_in_stack", int(bool(
            set(failure.get("stack_methods", [])) & set(re.findall(r"\b[A-Za-z_]\w*(?=\s*\()", patch.buggy_method))
        )) if failure.get("stack_methods") else None, "failure", "subclaim", [patch.observed_failure()], group="failure"),
        _predicate("touches_buggy_line", touches_line, "ast" if diff.get("gumtree_success") else "diff", "expression", spans, group="edit"),
        _predicate("touches_buggy_method", 1 if method_match else None, "diff", "subclaim", spans, group="edit"),
        _predicate("modified_file_count", int(diff.get("modified_file_count", 0) > 0), "diff", "claim", spans, diff.get("modified_file_count", 0), "edit"),
        _predicate("patch_size", int(diff.get("patch_size", 0) > 0), "diff", "claim", spans, diff.get("patch_size", 0), "edit"),
        _predicate("local_boundary_available", 1 if local_source else None, "context", "subclaim", [local_source]),
    ]
    external_predicates = {item["name"]: item for item in cpg.get("predicates", [])}
    aliases = {
        "operands_preserved": "operand_preserved",
        "semantic_role_preserved": "receiver_dependency_preserved",
        "condition_consistent": "loop_condition_feasible", "repair_path_feasible": "loop_condition_feasible",
    }
    for name, value in semantic.items():
        if name in {"schema_support", "semantic_schema_supports", "identifier_overlap_ratio"}:
            continue
        external = external_predicates.get(name, external_predicates.get(aliases.get(name, "")))
        if external and external.get("value") == value:
            item = deepcopy(external)
            item.update({"id": "P:" + name, "name": name, "group": "relation"})
        else:
            item = _predicate(name, value, "schema" if name == "failure_explained" else "diff",
                              "expression", [patch.observed_failure()] + spans)
        if name == "failure_explained" and value == 1 and semantic.get("null_guard_before_dereference") == 1:
            item["independent_predicate_names"] = ["failure_observed", "null_guard_before_dereference"]
        values.append(item)
    known = {item["name"] for item in values}
    for item in cpg.get("predicates", []):
        if item.get("name") not in known and item.get("name") not in set(aliases.values()):
            copied = deepcopy(item)
            copied.update({"id": "P:" + item["name"], "group": "relation"})
            values.append(copied)
    for schema in semantic["schema_support"]:
        names = schema["predicate_names"]
        values.append(_predicate(
            "schema:" + schema["schema"], schema["condition_value"], "schema", "expression",
            [span for item in values if item["name"] in names for span in item.get("grounding", [])], group="schema",
        ))
    values.extend([
        _predicate("identifier_overlap_ratio", 1, "diff", "expression", spans, semantic["identifier_overlap_ratio"], "edit"),
        _predicate("candidate_pool_size", 1, "pool", "claim", [], patch.pool_size, "pool"),
        _predicate("raw_patch_rank", 1, "pool", "claim", [], patch.patch_rank, "pool"),
    ])
    return values


def _source_level(source: str, independent_facts: int = 1) -> int:
    if source == "failure":
        return 3
    if source == "schema" and independent_facts >= 2:
        return 2
    return 1


def _argument(
    uid: str,
    kind: str,
    target: str,
    value: Any,
    strength: str,
    source: str,
    scope: str,
    predicate_names: list[str],
    grounding: list[str],
    **extra: Any,
) -> dict[str, Any]:
    item = {
        "id": uid,
        "kind": kind,
        "polarity": "+" if kind == "support" else "-",
        "target_id": target,
        "condition_value": value,
        "strength": ARGUMENT_STRENGTH[strength],
        "strength_label": strength,
        "source": source,
        "source_level": _source_level(source, len(predicate_names)),
        "scope": scope,
        "scope_level": SCOPE_LEVEL[scope],
        "predicate_names": predicate_names,
        "grounding": [value for value in grounding if value],
        "status": "unresolved",
        "checking_condition": {"predicates": predicate_names, "operator": "all" if kind == "support" else "not_all"},
    }
    item.update(extra)
    return item


def _supports(predicates: dict[str, dict[str, Any]], semantic: dict[str, Any]) -> list[dict[str, Any]]:
    rules = (
        ("S1", "F", ("failure_observed",), "high"),
        ("S2", "T", ("touches_buggy_line",), "high"),
        ("S3", "T", ("touches_buggy_method",), "medium"),
        ("S4", "M", ("failure_explained",), "high"),
        ("S5", "M", ("failure_observed", "null_guard_before_dereference", "repair_path_feasible"), "high"),
        ("S6", "P", ("operands_preserved", "return_contract_aligned"), "medium"),
        ("S7", "P", ("required_computation_preserved", "side_effects_preserved"), "medium"),
        ("S8", "B", ("local_boundary_available",), "medium"),
    )
    output = []
    for uid, target, names, strength in rules:
        items = [predicates[name] for name in names]
        source = items[0]["source"] if len(items) == 1 else "schema"
        independent = list(dict.fromkeys(
            name for item in items for name in item.get("independent_predicate_names", [item["name"]])
        ))
        output.append(_argument(
            uid, "support", target, _conjunction([item["value"] for item in items]), strength,
            source, "subclaim" if target in {"F", "B"} else "expression",
            list(names), [span for item in items for span in item.get("grounding", [])],
        ))
        output[-1]["source_level"] = _source_level(source, len(independent))
        output[-1]["independent_predicate_names"] = independent
    for index, schema in enumerate(semantic["schema_support"], 1):
        names = schema["predicate_names"]
        output.append(_argument(
            f"SS{index}", "support", "M", schema["condition_value"], "medium", "schema", "expression",
            names, [span for name in names for span in predicates[name].get("grounding", [])],
            schema=schema["schema"],
        ))
    return output


def _rebuttals(patch_types: set[str], predicates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for operator, activation, targets, severity, name, test, hypothesis in REBUTTAL_LIBRARY:
        if not patch_types & activation:
            continue
        predicate = predicates[test]
        safe_value = predicate["value"]
        risk_value = None if safe_value is None else 1 - int(safe_value)
        for target in targets:
            output.append(_argument(
                f"{operator}:{target}", "rebuttal", target, risk_value, severity,
                predicate["source"], predicate["scope"], [test], predicate.get("grounding", []),
                operator=operator, operator_name=name, severity=severity, hypothesis=hypothesis,
                activation_types=sorted(patch_types & activation), safe_predicate_names=[test],
            ))
    return output


def _missing(predicates: list[dict[str, Any]], patch: PatchInput) -> list[dict[str, Any]]:
    output = []
    for item in predicates:
        if item.get("value") is not None:
            continue
        output.append(
            {
                "id": f"M{len(output) + 1}",
                "kind": "missing",
                "predicate_name": item["name"],
                "new_status": "unresolved",
                "criticality": "high" if item["name"] in {
                    "repair_path_feasible", "return_contract_aligned", "semantic_role_preserved",
                } else "medium",
                "reason": f"{item['name']} cannot be determined from the available artifacts.",
                "grounding": item.get("grounding", []),
            }
        )
    if re.search(r"\b(?:public|protected)\b", patch.buggy_method) and not any(
        item["name"] == "external_behavior_preserved" for item in predicates
    ):
        output.append(
            {
                "id": f"M{len(output) + 1}",
                "kind": "missing",
                "predicate_name": "external_caller_behavior",
                "new_status": "unresolved",
                "criticality": "medium",
                "reason": "Caller-visible behavior outside the observed tests is unavailable.",
                "grounding": [],
            }
        )
    return output


def _undercutters(
    supports: list[dict[str, Any]],
    diff: dict[str, Any],
    cpg: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for support in supports:
        unavailable = (
            support["source"] == "ast" and not diff.get("gumtree_success")
        ) or (
            support["source"] == "schema" and not (
                cpg.get("analysis_success") or any(
                    item.get("source") in {"cpg", "pdg"} and item.get("value") in {0, 1}
                    for item in cpg.get("predicates", [])
                )
            )
        )
        if not unavailable:
            continue
        output.append(
            {
                "id": f"U{len(output) + 1}",
                "kind": "undercutter",
                "target_id": support["id"],
                "condition_value": 1,
                "severity": "medium",
                "status": "supported",
                "reason": "The analysis source required by this inference is unavailable.",
                "grounding": [],
            }
        )
    return output


def _priority(item: dict[str, Any]) -> tuple[int, int, float]:
    return (
        int(item.get("source_level", 1)),
        int(item.get("scope_level", 0)),
        float(item.get("strength", 0.0)),
    )


def resolve_defeaters(graph: dict[str, Any]) -> None:

    arguments = graph["supporting_evidence"] + graph["rebuttals"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in arguments:
        item.pop("resolution", None)
        value = item["condition_value"]
        override = item.get("semantic_status")
        if override == "contradicted":
            value = 0
        elif override == "supported":
            value = 1
        elif override in {"unresolved", "undercut"}:
            value = None
        item["evaluated_condition"] = value
        item["status"] = "contradicted" if value == 0 else "unresolved" if value is None else "supported"
        if value == 1:
            grouped[item["target_id"]].append(item)
    states = {}
    for target in [*SUBCLAIMS, "C0"]:
        applicable = grouped.get(target, [])
        if not applicable:
            states[target] = "unresolved"
            continue
        maximum = max(_priority(item) for item in applicable)
        highest = [item for item in applicable if _priority(item) == maximum]
        polarities = {item["polarity"] for item in highest}
        if len(polarities) != 1:
            states[target] = "unresolved"
            for item in applicable:
                item["status"] = "unresolved"
            continue
        winner = next(iter(polarities))
        states[target] = "supported" if winner == "+" else "refuted"
        for item in applicable:
            if item["polarity"] == winner:
                item["status"] = "supported"
            elif _priority(item) < maximum:
                item["status"] = "defeated"
                item["resolution"] = "deterministic_priority"
    for undercutter in graph["undercutters"]:
        value = undercutter["condition_value"]
        undercutter["status"] = "supported" if value == 1 else "contradicted" if value == 0 else "unresolved"
    for item in arguments:
        factor = math.prod(
            1.0 - UNDERCUT_FACTOR[u["severity"]] for u in graph["undercutters"]
            if u["target_id"] in {item["id"], "I:" + item["id"]} and u["status"] == "supported"
        )
        item["effective_strength"] = float(item["strength"]) * factor
    graph["subclaim_states"] = {target: states[target] for target in SUBCLAIMS}
    core = [states[target] for target in SUBCLAIMS]
    graph["claim_state"] = (
        "refuted" if "refuted" in core or states["C0"] == "refuted"
        else "supported" if all(state == "supported" for state in core) else "unresolved"
    )
    graph["argument_states"] = {item["id"]: item["status"] for item in arguments + graph["undercutters"]}
    graph["cae_layers"] = {
        "claims": ["C0", *SUBCLAIMS],
        "arguments": [item["id"] for item in arguments + graph["undercutters"]],
        "evidence": [item["id"] for item in graph["immutable_facts"]["predicates"]],
    }
    graph["relations"] = [item for item in graph["relations"] if not item.get("generated_by_resolution")]
    for items in grouped.values():
        for defeated in items:
            if defeated["status"] != "defeated":
                continue
            for winner in items:
                if winner["status"] == "supported" and winner["polarity"] != defeated["polarity"] and _priority(winner) > _priority(defeated):
                    graph["relations"].append({
                        "id": f"REF:{winner['id']}:{defeated['id']}", "type": "refute",
                        "source": winner["id"], "target": defeated["id"], "generated_by_resolution": True,
                    })


def score_graph(graph: dict[str, Any]) -> None:
    support = sum(
        20.0 * float(item.get("effective_strength", item.get("strength", 0.0)))
        for item in graph.get("supporting_evidence", [])
        if item.get("status") == "supported"
    )
    rebuttal = 0.0
    for item in graph.get("rebuttals", []):
        weight = SEVERITY_WEIGHT[item["severity"]]
        status = item.get("status")
        if item.get("semantic_status") == "undercut":
            rebuttal += weight * 0.15
        elif status == "supported":
            rebuttal += weight
        elif status == "unresolved":
            rebuttal += weight * 0.35
        elif status == "undercut":
            rebuttal += weight * 0.15
    missing = sum(
        SEVERITY_WEIGHT[item["criticality"]] * 0.35
        for item in graph.get("missing_evidence", [])
        if item.get("new_status") != "resolved"
    )
    graph["score"] = {
        "support_score": support,
        "rebuttal_penalty": rebuttal,
        "missing_evidence_penalty": missing,
        "assurance_score": max(0.0, min(100.0, 50.0 + support - rebuttal - missing)),
    }


def construct_evidence_graph(patch: PatchInput, static: dict[str, Any]) -> dict[str, Any]:
    diff = static["diff_ast_facts"]
    cpg = static["cpg_predicates"]
    failure = static["failure_predicates"]
    semantic = _semantic_facts(patch, diff, cpg, failure)
    predicate_list = _predicates(patch, diff, cpg, failure, semantic)
    predicate_map = {item["name"]: item for item in predicate_list}
    best_schema = next(
        (item["schema"] for item in semantic["schema_support"] if item["condition_value"] == 1),
        _fallback_mechanism(diff),
    )
    claim = {
        "id": "C0",
        "statement": (
            "The patch addresses the observed failure by applying the identified mechanism "
            "to the modification target while preserving required behavior within the assessed boundary."
        ),
        "failure_model": _failure_model(failure),
        "change_target": diff.get("change_target", {}),
        "fix_mechanism": best_schema,
        "preservation_obligations": [
            "required_operands", "return_contract", "nonfailing_paths",
            "data_dependencies", "observable_side_effects",
        ],
        "assessment_boundary": {
            "scope": "observed_failure_and_local_dependency_slice",
            "buggy_method": patch.buggy_method,
            "stack_methods": failure.get("stack_methods", []),
            "modified_files": [item["file"] for item in diff.get("files", [])],
            "method_visibility": "public_or_protected" if semantic["method_public_or_protected"] else "local_or_unknown",
            "data_control_dependencies": {
                "buggy_slice": cpg.get("buggy_slice", {}),
                "patched_slice": cpg.get("patched_slice", {}),
            },
            "external_callers_verified": semantic["external_behavior_preserved"] == 1,
        },
        "subclaims": [
            {"id": uid, "statement": statement} for uid, statement in SUBCLAIMS.items()
        ],
    }
    claim["components"] = {name: claim[key] for name, key in (
        ("F", "failure_model"), ("T", "change_target"), ("M", "fix_mechanism"),
        ("P", "preservation_obligations"), ("B", "assessment_boundary"),
    )}
    claim["statement"] = (
        f"The patch addresses failure {claim['failure_model']} by applying mechanism {best_schema} "
        f"to target {claim['change_target']}, while preserving {claim['preservation_obligations']} "
        f"within boundary {claim['assessment_boundary']['scope']}."
    )
    supports = _supports(predicate_map, semantic)
    rebuttals = _rebuttals(set(diff.get("patch_types", [])), predicate_map)
    undercutters = _undercutters(supports, diff, cpg)
    graph = {
        "patch_uid": patch.patch_uid,
        "claim": claim,
        "relations": [
            {"id": "I:" + item["id"], "type": "support", "source": item["id"], "target": item["target_id"]}
            for item in supports
        ] + [
            {"id": "I:" + item["id"], "type": "attack", "source": item["id"], "target": item["target_id"]}
            for item in rebuttals
        ] + [
            {"id": "I:" + item["id"], "type": "undercut", "source": item["id"], "target": "I:" + item["target_id"]}
            for item in undercutters
        ],
        "immutable_facts": {
            "diff_ast_facts": diff,
            "cpg_predicates": cpg,
            "failure_predicates": failure,
            "predicates": predicate_list,
            "predicate_groups": {
                group: [item["id"] for item in predicate_list if item.get("group") == group]
                for group in ("failure", "edit", "schema", "relation", "pool")
            },
        },
        "semantic_facts": semantic,
        "supporting_evidence": supports,
        "rebuttals": rebuttals,
        "undercutters": undercutters,
        "missing_evidence": _missing(predicate_list, patch),
    }
    resolve_defeaters(graph)
    score_graph(graph)
    graph["feature_vector"] = extract_features(patch, graph)
    return graph


def _failure_model(failure: dict[str, Any]) -> str:
    if not failure.get("observed_failure"):
        return "failure_unavailable"
    if failure.get("has_null_failure"):
        return "null_dereference_exception"
    if failure.get("has_assertion_mismatch"):
        return "assertion_mismatch"
    if failure.get("exception_type"):
        return "exceptional_failure"
    return "observed_test_failure"


def _fallback_mechanism(diff: dict[str, Any]) -> str:
    types = set(diff.get("patch_types", []))
    order = (
        ("null_check_change", "null_handling_change"),
        ("guard_insertion", "guarded_control_flow"),
        ("condition_change", "predicate_change"),
        ("exception_handling_change", "exception_handling_change"),
        ("api_call_change", "api_behavior_change"),
        ("constant_change", "literal_value_change"),
        ("deletion", "behavior_deletion"),
    )
    return next((value for key, value in order if key in types), "general_code_change")


def integrate_revision(graph: dict[str, Any], revision: dict[str, Any]) -> dict[str, Any]:

    updated = deepcopy(graph)
    if not revision.get("validated"):
        updated["semantic_revision"] = deepcopy(revision)
        return updated
    rebuttals = {item["id"]: item for item in updated["rebuttals"]}
    for proposed in revision["rebuttal_refinements"]:
        target = rebuttals[proposed["rebuttal_id"]]
        target["semantic_status"] = proposed["new_status"]
        target["revision_reason"] = proposed["reason"]
        target["revision_grounding"] = proposed["grounding"]
        if proposed["new_status"] == "undercut":
            undercutter = {
                "id": "LU:" + target["id"], "kind": "undercutter",
                "target_id": "I:" + target["id"], "condition_value": 1,
                "severity": target["severity"], "status": "supported",
                "reason": proposed["reason"], "grounding": proposed["grounding"],
            }
            updated["undercutters"].append(undercutter)
            updated["relations"].append({
                "id": "I:" + undercutter["id"], "type": "undercut",
                "source": undercutter["id"], "target": undercutter["target_id"],
            })
        elif proposed["new_status"] == "contradicted":
            updated["relations"].append({
                "id": "SEMREF:" + target["id"], "type": "refute",
                "source": proposed["predicate_ids"], "target": target["id"],
            })
    existing_ids = {item["id"] for item in updated["supporting_evidence"] + updated["rebuttals"] + updated["undercutters"]}
    for proposed in revision["new_rebuttals"]:
        index = 1
        while f"RN{index}" in existing_ids:
            index += 1
        uid = f"RN{index}"
        item = _argument(
            uid, "rebuttal", proposed["target_id"], None, proposed["severity"],
            "context", "claim" if proposed["target_id"] == "C0" else "subclaim",
            proposed["predicate_names"], proposed["grounding"],
            operator=proposed["operator"], operator_name=proposed["operator"],
            severity=proposed["severity"], hypothesis=proposed["hypothesis"],
            initial_status="unresolved",
        )
        # True cited facts do not establish a newly proposed failure hypothesis.
        updated["rebuttals"].append(item)
        updated["relations"].append({
            "id": "I:" + uid, "type": "attack", "source": uid, "target": proposed["target_id"],
        })
        existing_ids.add(uid)
    missing = {item["id"]: item for item in updated["missing_evidence"]}
    for proposed in revision["missing_evidence_updates"]:
        target = missing.get(proposed["missing_id"])
        if target is None:
            target = {"id": proposed["missing_id"], "kind": "missing", "predicate_name": proposed["missing_id"]}
            updated["missing_evidence"].append(target)
            missing[target["id"]] = target
        target.update({key: proposed[key] for key in ("new_status", "criticality", "reason", "grounding")})
    for proposed in revision["semantic_schema_candidates"]:
        if proposed["status"] != "supported":
            continue
        existing = next((
            item for item in updated["supporting_evidence"]
            if item.get("schema") == proposed["schema"] and item["condition_value"] == 1
        ), None)
        if existing:
            existing["semantic_confirmation"] = deepcopy(proposed)
            continue
        index = 1
        while f"LS{index}" in existing_ids:
            index += 1
        uid = f"LS{index}"
        item = _argument(
            uid, "support", "M", 1, "medium", "schema", "expression",
            list(SCHEMA_PREDICATES[proposed["schema"]]), proposed["grounding"],
            schema=proposed["schema"],
        )
        updated["supporting_evidence"].append(item)
        updated["relations"].append({"id": "I:" + uid, "type": "support", "source": uid, "target": "M"})
        existing_ids.add(uid)
        if proposed["schema"] not in updated["semantic_facts"]["semantic_schema_supports"]:
            updated["semantic_facts"]["semantic_schema_supports"].append(proposed["schema"])
    updated["semantic_schema_candidates"] = deepcopy(revision["semantic_schema_candidates"])
    updated["semantic_revision"] = deepcopy(revision)
    # Missing evidence is deliberately absent from the state-resolution input.
    resolve_defeaters(updated)
    score_graph(updated)
    return updated


def extract_features(patch: PatchInput, graph: dict[str, Any]) -> dict[str, float]:
    facts = graph["immutable_facts"]["diff_ast_facts"]
    semantic = graph["semantic_facts"]
    supports = graph["supporting_evidence"]
    rebuttals = graph["rebuttals"]
    missing = graph["missing_evidence"]
    score = graph["score"]
    patch_types = set(facts.get("patch_types", []))
    resolved = {"contradicted", "defeated"}
    semantic_support = sum(
        20.0 * float(item.get("effective_strength", item.get("strength", 0.0)))
        for item in supports if item.get("schema") and item["status"] == "supported"
    )
    pool_rank = 1.0
    if patch.pool_size > 1:
        pool_rank = 1.0 - min(max(int(patch.metadata.get("pool_position", max(patch.patch_rank - 1, 0))), 0), patch.pool_size - 1) / (patch.pool_size - 1)
    values = {
        "prior_score": score["assurance_score"] / 100.0,
        "raw_rank_feature": pool_rank,
        "support_score": score["support_score"] / 100.0,
        "base_support_score": (score["support_score"] - semantic_support) / 100.0,
        "semantic_support_score": semantic_support / 100.0,
        "strong_schema_count": float(len(semantic.get("semantic_schema_supports", []))),
        "resolved_rebuttal_count": float(sum(item["status"] in resolved for item in rebuttals)),
        "resolved_high_medium_count": float(sum(
            item["status"] in resolved and item["severity"] in {"high", "medium"}
            for item in rebuttals
        )),
        "identifier_overlap": float(semantic["identifier_overlap_ratio"]),
        "operands_preserved": float(semantic.get("operands_preserved") or 0),
        "null_guard_before_deref": float(semantic["null_guard_before_dereference"] == 1),
        "condition_flip_explains_symptom": float(
            semantic["condition_consistent"] == 1 and semantic["failure_explained"] == 1
        ),
        "return_contract_alignment": float(semantic.get("return_contract_aligned") or 0),
        "def_use_role_support": float(semantic.get("semantic_role_preserved") or 0),
        "controlled_region_guard": float(
            semantic["controlled_region_has_early_exit"]
            and semantic["controlled_region_has_later_dereference"]
        ),
        "trigger_call_alignment": float(semantic["changed_call_matches_trigger"]),
        "supported_high_count": float(sum(
            item["status"] == "supported" and item["severity"] == "high" for item in rebuttals
        )),
        "supported_medium_count": float(sum(
            item["status"] == "supported" and item["severity"] == "medium" for item in rebuttals
        )),
        "unresolved_high_medium_count": float(sum(
            item["status"] == "unresolved" and item["severity"] in {"high", "medium"}
            for item in rebuttals
        )),
        "rebuttal_penalty": score["rebuttal_penalty"] / 100.0,
        "missing_penalty": score["missing_evidence_penalty"] / 100.0,
        "missing_count": float(sum(item.get("new_status") != "resolved" for item in missing)),
        "public_regression_missing": float(any(
            item.get("predicate_name") in {"external_caller_behavior", "external_behavior_preserved"}
            and item.get("new_status") != "resolved" for item in missing
        ) and semantic["method_public_or_protected"]),
        "side_effect_deleted": float(semantic["side_effect_deleted"]),
        "condition_contradiction": float(semantic["condition_contradiction"]),
        "return_contract_risk": float(semantic.get("return_contract_aligned") == 0),
        "receiver_delta": float(semantic["receiver_delta"]),
        "variable_to_constant": float(semantic["variable_to_constant"]),
        "expression_operand_loss": float(semantic.get("operands_preserved") == 0),
        "trigger_literal_overlap": float(semantic["trigger_literal_overlap"]),
        "patch_size": min(float(facts.get("patch_size", 0)), 50.0) / 50.0,
        "touches_buggy_method": float(next((p["value"] == 1 for p in graph["immutable_facts"]["predicates"] if p["name"] == "touches_buggy_method"), False)),
        "touches_buggy_line": float(next((p["value"] == 1 for p in graph["immutable_facts"]["predicates"] if p["name"] == "touches_buggy_line"), False)),
        "method_public_or_protected": float(semantic["method_public_or_protected"]),
        "type_condition_change": float("condition_change" in patch_types),
        "type_guard_insertion": float("guard_insertion" in patch_types),
        "type_null_check_change": float("null_check_change" in patch_types),
        "type_constant_change": float("constant_change" in patch_types),
        "type_api_call_change": float("api_call_change" in patch_types),
        "type_deletion": float("deletion" in patch_types),
        "type_value_change": float("assignment_or_value_change" in patch_types),
    }
    return {
        key: float(value) if math.isfinite(float(value)) else 0.0
        for key, value in values.items()
    }
