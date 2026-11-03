from __future__ import annotations

from collections import Counter, defaultdict, deque
import difflib
import html
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any

from .config import ADALConfig
from .types import PatchInput


CALL = re.compile(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(")
COND = re.compile(r"\b(?:if|while|for)\s*\((.*?)\)", re.DOTALL)
IDENT = re.compile(r"\b[A-Za-z_]\w*\b")
LITERAL = re.compile(r"\b(?:0x[0-9A-Fa-f]+|\d+(?:\.\d+)?|true|false|null)\b")
METHOD = re.compile(
    r"(?:\b(?:public|protected|private|static|final|synchronized|abstract|native|default)\s+)*"
    r"(?:[\w.$<>\[\],?]+\s+)+([A-Za-z_$][\w$]*)\s*\([^;{}]*\)\s*(?:throws\s+[^{}]+)?\{"
)


def _run(command: list[str], timeout: int, cwd: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout,
            check=False,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "duration_seconds": time.perf_counter() - started,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(exc),
            "returncode": -1,
            "duration_seconds": time.perf_counter() - started,
        }


def _patch_types(removed: list[str], added: list[str]) -> list[str]:
    old = "\n".join(removed)
    new = "\n".join(added)
    values: set[str] = set()
    if COND.findall(old) != COND.findall(new):
        values.add("condition_change")
    if re.search(r"\bif\s*\(", new) and not re.search(r"\bif\s*\(", old):
        values.add("guard_insertion")
    if "null" in old or "null" in new:
        values.add("null_check_change")
    if set(LITERAL.findall(old)) != set(LITERAL.findall(new)):
        values.add("constant_change")
    if set(CALL.findall(old)) != set(CALL.findall(new)):
        values.add("api_call_change")
    if METHOD.findall(old) != METHOD.findall(new):
        values.add("method_signature_change")
    if len(removed) > len(added):
        values.add("deletion")
    if re.search(r"\breturn\b", new) and not re.search(r"\breturn\b", old):
        values.add("early_return_or_default_return")
    if re.search(r"\breturn\b", old) and re.search(r"\breturn\b", new) and old != new:
        values.add("return_change")
    if re.search(r"\b(?:try|catch|throw)\b", old + "\n" + new):
        values.add("exception_handling_change")
    if re.search(r"(?<![=!<>])=(?!=)", old + "\n" + new):
        values.add("assignment_or_value_change")
    if old != new and (IDENT.findall(old) != IDENT.findall(new) or LITERAL.findall(old) != LITERAL.findall(new)):
        values.add("assignment_or_value_change")
    if old != new and (COND.findall(old) or COND.findall(new)):
        values.add("expression_change")
    return sorted(values or {"general_code_change"})


def _diff_facts(patch: PatchInput) -> dict[str, Any]:
    file_diffs = []
    removed: list[str] = []
    added: list[str] = []
    changed_identifiers: set[str] = set()
    for name, buggy, patched in patch.source_pairs():
        old_lines = buggy.splitlines()
        new_lines = patched.splitlines()
        if buggy == patched:
            continue
        matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
        old_changed_lines = []
        new_changed_lines = []
        for operation, old_start, old_stop, new_start, new_stop in matcher.get_opcodes():
            if operation != "equal":
                old_changed_lines.extend(range(old_start + 1, old_stop + 1))
                new_changed_lines.extend(range(new_start + 1, new_stop + 1))
        diff = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{name}",
                tofile=f"b/{name}",
                lineterm="",
            )
        )
        file_removed = [
            line[1:] for line in diff if line.startswith("-") and not line.startswith("---")
        ]
        file_added = [
            line[1:] for line in diff if line.startswith("+") and not line.startswith("+++")
        ]
        removed.extend(file_removed)
        added.extend(file_added)
        changed_identifiers.update(IDENT.findall("\n".join(file_removed + file_added)))
        file_diffs.append(
            {
                "file": name,
                "unified_diff": "\n".join(diff),
                "removed_lines": file_removed,
                "added_lines": file_added,
                "old_changed_line_numbers": old_changed_lines,
                "new_changed_line_numbers": new_changed_lines,
            }
        )
    old_ids = set(IDENT.findall("\n".join(removed)))
    new_ids = set(IDENT.findall("\n".join(added)))
    union = old_ids | new_ids
    return {
        "files": file_diffs,
        "unified_diff": "\n".join(item["unified_diff"] for item in file_diffs),
        "removed_lines": removed,
        "added_lines": added,
        "old_conditions": [value.strip() for value in COND.findall("\n".join(removed))],
        "new_conditions": [value.strip() for value in COND.findall("\n".join(added))],
        "removed_calls": sorted(set(CALL.findall("\n".join(removed)))),
        "added_calls": sorted(set(CALL.findall("\n".join(added)))),
        "removed_literals": sorted(set(LITERAL.findall("\n".join(removed)))),
        "added_literals": sorted(set(LITERAL.findall("\n".join(added)))),
        "changed_identifiers": sorted(changed_identifiers),
        "identifier_overlap_ratio": len(old_ids & new_ids) / len(union) if union else 1.0,
        "patch_size": len(removed) + len(added),
        "modified_file_count": len(file_diffs),
        "patch_types": _patch_types(removed, added),
    }


def _failure_predicates(patch: PatchInput) -> dict[str, Any]:
    text = patch.observed_failure()
    exception = ""
    for line in text.splitlines():
        match = re.search(r"\b([\w.$]*(?:Exception|Error|Failure))\b", line)
        if match:
            exception = match.group(1)
            break
    expected = ""
    actual = ""
    match = re.search(r"expected:?\s*<([^>]*)>\s*but\s*(?:was|got):?\s*<([^>]*)>", text, re.I)
    if match:
        expected, actual = match.groups()
    frames = []
    for value in re.finditer(r"\bat\s+([\w.$]+)\.([\w$<>]+)\(([^:()]+)(?::(\d+))?\)", text):
        frames.append(
            {
                "class": value.group(1),
                "method": value.group(2),
                "file": value.group(3),
                "line": int(value.group(4)) if value.group(4) else None,
            }
        )
    return {
        "exception_type": exception,
        "expected": expected,
        "actual": actual,
        "stack_frames": frames,
        "stack_methods": sorted({item["method"] for item in frames}),
        "triggering_test_available": bool(patch.triggering_test),
        "has_assertion_mismatch": bool(expected or actual or "assert" in text.lower()),
        "has_null_failure": "NullPointerException" in text,
        "observed_failure": text,
    }


def _parse_gumtree(text: str, source: str) -> dict[str, Any]:
    actions = []
    current: dict[str, Any] | None = None
    counts = Counter()
    for raw in text.splitlines():
        line = raw.strip()
        match = re.match(
            r"^(insert-node|insert-tree|delete-node|delete-tree|move-tree|update-node)\b",
            line,
        )
        if match:
            if current:
                actions.append(current)
            current = {"action": match.group(1), "source": source, "raw": line}
            counts[match.group(1)] += 1
            continue
        if current:
            node = re.search(r"^(?:---\s*)?([A-Za-z_]\w*)(?::\s*(.*?))?\s*\[(\d+),(\d+)\]$", line)
            if node:
                current["node_type"] = node.group(1)
                current["label"] = (node.group(2) or "").strip()
            location = re.search(r"\[(\d+),(\d+)\]", line)
            if location:
                current["start"] = int(location.group(1))
                current["end"] = int(location.group(2))
                current["length"] = max(0, int(location.group(2)) - int(location.group(1)))
    if current:
        actions.append(current)
    lowered = text.lower()
    return {
        "actions": actions,
        "action_counts": dict(counts),
        "guard_inserted": bool(
            ("insert-node" in lowered or "insert-tree" in lowered)
            and ("ifstatement" in lowered or "conditionalexpression" in lowered)
        ),
        "return_inserted": "returnstatement" in lowered and "insert" in lowered,
        "condition_updated": "infixexpression" in lowered and "update-node" in lowered,
        "invocation_changed": "methodinvocation" in lowered,
        "tree_deleted": "delete-tree" in lowered,
    }


def _line_number(source: str, offset: int | None) -> int | None:
    if offset is None or offset < 0 or offset > len(source):
        return None
    return source.count("\n", 0, offset) + 1


def _enclosing_method(source: str, offset: int | None) -> str:
    if offset is None:
        return ""
    matches = list(METHOD.finditer(source[:offset]))
    return matches[-1].group(1) if matches else ""


def _write_sources(root: Path, files: dict[str, str]) -> None:
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _run_gumtree(patch: PatchInput, config: ADALConfig, root: Path) -> dict[str, Any]:
    if not config.analysis.run_gumtree or not config.paths.gumtree:
        return {"success": False, "reason": "disabled_or_unconfigured", "features": {}}
    results = []
    all_actions = []
    total = 0.0
    for index, (name, buggy, patched) in enumerate(patch.source_pairs()):
        old_path = root / "gumtree" / str(index) / "Buggy.java"
        new_path = root / "gumtree" / str(index) / "Patched.java"
        old_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.write_text(buggy, encoding="utf-8")
        new_path.write_text(patched, encoding="utf-8")
        result = _run(
            [config.paths.gumtree, "textdiff", str(old_path), str(new_path)],
            config.analysis.static_timeout_seconds,
        )
        parsed = _parse_gumtree(result.get("stdout", ""), name) if result["success"] else {}
        for action in parsed.get("actions", []):
            source = patched if action["action"].startswith("insert") else buggy
            action["line"] = _line_number(source, action.get("start"))
            action["enclosing_method"] = _enclosing_method(source, action.get("start"))
            node_type = action.get("node_type", "").lower()
            action["region_kind"] = (
                "condition" if "if" in node_type or "infix" in node_type
                else "method_invocation" if "invocation" in node_type
                else "expression" if "expression" in node_type
                else "statement"
            )
        all_actions.extend(parsed.get("actions", []))
        total += result["duration_seconds"]
        results.append({"file": name, "success": result["success"], "features": parsed})
    return {
        "success": bool(results) and all(item["success"] for item in results),
        "duration_seconds": total,
        "files": results,
        "features": {"actions": all_actions},
    }


def _parse_dot(directory: Path) -> dict[str, Any]:
    nodes: dict[str, dict[str, str]] = {}
    edges: list[dict[str, str]] = []
    for path in directory.rglob("*.dot"):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            label_match = re.search(r'\blabel\s*=\s*(?:"((?:\\.|[^"\\])*)"|<(.*)>)', line)
            label = ""
            if label_match:
                label = html.unescape(label_match.group(1) or label_match.group(2) or "")
                label = re.sub(r"</?(?:FONT|B|I|TABLE|TR|TD|BR)\b[^>]*>", " ", label, flags=re.I)
            edge = re.search(r'^\s*"?([\w:.-]+)"?\s*->\s*"?([\w:.-]+)"?', line)
            if edge:
                edges.append(
                    {
                        "source": edge.group(1),
                        "target": edge.group(2),
                        "label": label,
                    }
                )
                continue
            node = re.search(r'^\s*"?([\w:.-]+)"?\s*\[', line)
            if node:
                nodes[node.group(1)] = {"label": label, "file": path.name}
    return {"nodes": nodes, "edges": edges}


def _run_joern_source(
    files: dict[str, str],
    name: str,
    config: ADALConfig,
    root: Path,
) -> dict[str, Any]:
    source_root = root / name / "src"
    source_root.mkdir(parents=True, exist_ok=True)
    _write_sources(source_root, files)
    cpg_path = root / name / "cpg.bin"
    cpg_dir = root / name / "cpg"
    pdg_dir = root / name / "pdg"
    parsed = _run(
        [
            config.paths.joern_parse,
            "--language",
            "javasrc",
            "-o",
            str(cpg_path),
            str(source_root),
        ],
        config.analysis.static_timeout_seconds,
    )
    if not parsed["success"]:
        return {"success": False, "parse": parsed}
    cpg_export = _run(
        [
            config.paths.joern_export,
            "--repr",
            "all",
            "--format",
            "dot",
            "-o",
            str(cpg_dir),
            str(cpg_path),
        ],
        config.analysis.static_timeout_seconds,
    )
    pdg_export = _run(
        [
            config.paths.joern_export,
            "--repr",
            "pdg",
            "--format",
            "dot",
            "-o",
            str(pdg_dir),
            str(cpg_path),
        ],
        config.analysis.static_timeout_seconds,
    )
    return {
        "success": cpg_export["success"] and pdg_export["success"],
        "parse": parsed,
        "cpg_export": cpg_export,
        "pdg_export": pdg_export,
        "cpg": _parse_dot(cpg_dir) if cpg_export["success"] else {},
        "pdg": _parse_dot(pdg_dir) if pdg_export["success"] else {},
    }


def _slice(graph: dict[str, Any], identifiers: set[str], depth: int = 2) -> dict[str, Any]:
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])
    roots = {
        uid
        for uid, node in nodes.items()
        if any(re.search(rf"\b{re.escape(name)}\b", node.get("label", "")) for name in identifiers)
    }
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        adjacency[edge["source"]].add(edge["target"])
        adjacency[edge["target"]].add(edge["source"])
    selected = set(roots)
    queue = deque((uid, 0) for uid in roots)
    while queue:
        uid, distance = queue.popleft()
        if distance >= depth:
            continue
        for neighbor in adjacency.get(uid, set()):
            if neighbor not in selected:
                selected.add(neighbor)
                queue.append((neighbor, distance + 1))
    return {
        "root_ids": sorted(roots),
        "node_ids": sorted(selected),
        "nodes": {uid: nodes[uid] for uid in selected if uid in nodes},
        "edges": [
            edge for edge in edges
            if edge["source"] in selected and edge["target"] in selected
        ],
    }


def _predicate(
    name: str,
    value: bool | int | float | str | None,
    source: str,
    scope: str,
    grounding: list[str],
) -> dict[str, Any]:
    normalized = int(value) if isinstance(value, bool) else value
    return {
        "name": name,
        "value": normalized,
        "source": source,
        "scope": scope,
        "grounding": [item for item in grounding if item],
    }


def _typed_path(
    graph: dict[str, Any], sources: set[str], targets: set[str], labels: set[str],
) -> list[dict[str, str]]:
    """Produce a concrete typed relation path; absence is not a semantic disproof."""
    adjacency: dict[str, list[dict[str, str]]] = defaultdict(list)
    for edge in graph.get("edges", []):
        if any(edge.get("label", "").upper().startswith(label) for label in labels):
            adjacency[edge["source"]].append(edge)
    queue = deque((uid, []) for uid in sources)
    seen = set(sources)
    while queue:
        uid, path = queue.popleft()
        if uid in targets and path:
            return path
        for edge in adjacency.get(uid, []):
            if edge["target"] not in seen:
                seen.add(edge["target"])
                queue.append((edge["target"], path + [edge]))
    return []


def _graph_signature(graph: dict[str, Any], prefixes: tuple[str, ...]) -> set[tuple[str, str, str]]:
    nodes = graph.get("nodes", {})
    def label(uid: str) -> str:
        return re.sub(r"\s+", "", nodes.get(uid, {}).get("label", ""))
    return {
        (label(edge["source"]), label(edge["target"]), edge.get("label", "").split(":", 1)[0])
        for edge in graph.get("edges", [])
        if edge.get("label", "").upper().startswith(prefixes)
        and label(edge["source"]) and label(edge["target"])
    }


def _cfg_dominators(graph: dict[str, Any]) -> dict[str, set[str]]:
    predecessors: dict[str, set[str]] = defaultdict(set)
    vertices = set()
    for edge in graph.get("edges", []):
        if edge.get("label", "").upper().startswith("CFG"):
            predecessors[edge["target"]].add(edge["source"])
            vertices.update((edge["source"], edge["target"]))
    roots = {uid for uid in vertices if not predecessors[uid]}
    if not roots:
        return {}
    reachable = set(roots)
    while True:
        expanded = reachable | {uid for uid in vertices if predecessors[uid] & reachable}
        if expanded == reachable:
            break
        reachable = expanded
    dominators = {uid: {uid} if uid in roots else set(reachable) for uid in reachable}
    changed = True
    while changed:
        changed = False
        for uid in reachable - roots:
            parents = predecessors[uid] & reachable
            current = {uid} | (set.intersection(*(dominators[parent] for parent in parents)) if parents else set())
            if current != dominators[uid]:
                dominators[uid] = current
                changed = True
    return dominators


def _relation_predicates(
    patch: PatchInput, diff: dict[str, Any],
    buggy: dict[str, Any], patched: dict[str, Any],
) -> dict[str, Any]:
    removed, added = "\n".join(diff["removed_lines"]), "\n".join(diff["added_lines"])
    changed = set(diff["changed_identifiers"])
    def combined(value: dict[str, Any]) -> dict[str, Any]:
        return {
            "nodes": {**value.get("cpg", {}).get("nodes", {}), **value.get("pdg", {}).get("nodes", {})},
            "edges": value.get("cpg", {}).get("edges", []) + value.get("pdg", {}).get("edges", []),
        }
    old_graph, new_graph = combined(buggy), combined(patched)
    old_slice, new_slice = _slice(old_graph, changed), _slice(new_graph, changed)
    nodes = new_graph["nodes"]
    dominators = _cfg_dominators(new_graph)
    guards = {uid for uid, node in nodes.items() if re.search(r"!=\s*null", node.get("label", ""))}
    dereferences = {uid for uid, node in nodes.items() if re.search(r"fieldAccess|indirectFieldAccess|arrayAccess", node.get("label", ""), re.I)}
    domination = []
    for guard in guards:
        match = re.search(r"([\w.$\[\]]+)\s*!=\s*null", nodes[guard]["label"])
        if not match:
            continue
        operand = match.group(1)
        for target in dereferences:
            if guard in dominators.get(target, set()) and operand in nodes[target]["label"]:
                domination.append({"guard": guard, "dereference": target, "operand": operand,
                                   "guard_label": nodes[guard]["label"], "target_label": nodes[target]["label"]})
    assignments = {uid for uid, node in nodes.items() if "assignment" in node.get("label", "").lower()}
    returns = {uid for uid, node in nodes.items() if re.search(r"\bRETURN\b", node.get("label", ""), re.I)}
    calls = {uid for uid, node in nodes.items() if re.search(r"\bCALL\b", node.get("label", ""))}
    data_path = _typed_path(new_graph, assignments, returns, {"DDG", "REACHING_DEF"})
    effect_path = _typed_path(new_graph, calls | assignments, returns, {"CFG"})
    old_ddg, new_ddg = _graph_signature(old_slice, ("DDG", "REACHING_DEF")), _graph_signature(new_slice, ("DDG", "REACHING_DEF"))
    old_cfg, new_cfg = _graph_signature(old_slice, ("CFG", "CDG")), _graph_signature(new_slice, ("CFG", "CDG"))
    dependencies_equal = bool(old_ddg and new_ddg and old_ddg == new_ddg)
    control_equal = bool(old_cfg and new_cfg and old_cfg == new_cfg)
    old_calls, new_calls = set(diff["removed_calls"]), set(diff["added_calls"])
    # Local equivalence is accepted only with matching data AND control witnesses.
    role = 1 if dependencies_equal and control_equal and old_calls == new_calls else None
    computation = 1 if dependencies_equal and control_equal and data_path else None
    side_effects = 1 if dependencies_equal and control_equal and old_calls <= new_calls else None
    non_values = {"if", "else", "for", "while", "return", "new", "null", "true", "false",
                  "int", "long", "double", "float", "boolean", "char", "byte", "short",
                  "public", "protected", "private", "final", "static", "this", "super"}
    old_operands = set(IDENT.findall(removed)) - non_values - {call.rsplit(".", 1)[-1] for call in old_calls}
    new_operands = set(IDENT.findall(added)) - non_values - {call.rsplit(".", 1)[-1] for call in new_calls}
    # This predicate reports syntactic operand retention, not semantic equivalence.
    operands = int(old_operands <= new_operands) if old_operands else None
    impossible = next((
        condition for condition in diff["new_conditions"]
        if re.fullmatch(r"\s*(?:false|([A-Za-z_]\w*)\s*(?:!=|<|>)\s*\1)\s*", condition)
    ), None)
    tautology = next((condition for condition in diff["new_conditions"] if condition.strip() == "true"), None)
    feasible = 0 if impossible is not None else 1 if tautology is not None else None
    flow_grounding = [json.dumps(edge, ensure_ascii=False) for edge in data_path + effect_path]
    unchanged_returns = (
        bool(re.findall(r"\breturn\b[^;]*;", removed))
        and re.findall(r"\breturn\b[^;]*;", removed) == re.findall(r"\breturn\b[^;]*;", added)
    )
    return_contract = 1 if unchanged_returns and dependencies_equal and control_equal else None
    predicates = [
        _predicate("guard_dominates_dereference", 1 if domination else None, "cpg", "execution_path",
                   [json.dumps(item, ensure_ascii=False) for item in domination]),
        _predicate("assignment_reaches_return", 1 if data_path else None, "pdg", "execution_path", flow_grounding),
        _predicate("loop_condition_feasible", feasible, "ast", "expression",
                   [impossible or tautology] if impossible is not None or tautology is not None else []),
        _predicate("receiver_dependency_preserved", role, "schema", "expression", flow_grounding + diff["removed_lines"] + diff["added_lines"]),
        _predicate("operand_preserved", operands, "diff", "expression", diff["removed_lines"] + diff["added_lines"]),
        _predicate("side_effect_reachable", 1 if effect_path else None, "cpg", "execution_path", flow_grounding),
        _predicate("required_computation_preserved", computation, "schema", "execution_path", flow_grounding),
        _predicate("side_effects_preserved", side_effects, "schema", "execution_path", flow_grounding),
        _predicate("guard_preserves_valid_paths", 1 if control_equal else None, "cpg", "execution_path", flow_grounding),
        _predicate("return_contract_aligned", return_contract, "schema", "subclaim", flow_grounding + diff["added_lines"]),
        _predicate("external_behavior_preserved", None, "context", "claim", []),
        _predicate("not_trigger_specific", None, "context", "claim", []),
    ]
    return {
        "analysis_success": bool(buggy.get("success") and patched.get("success")),
        "predicates": predicates, "buggy_slice": old_slice, "patched_slice": new_slice,
        "data_dependency_edges": [edge for edge in new_slice["edges"] if edge.get("label", "").upper().startswith(("DDG", "REACHING_DEF"))],
        "positive_predicates": sorted(item["name"] for item in predicates if item["value"] == 1),
        "negative_predicates": sorted(item["name"] for item in predicates if item["value"] == 0),
    }


def _run_joern(patch: PatchInput, diff: dict[str, Any], config: ADALConfig, root: Path) -> dict[str, Any]:
    if (
        not config.analysis.run_joern
        or not config.paths.joern_parse
        or not config.paths.joern_export
    ):
        return {
            "success": False, "reason": "disabled_or_unconfigured",
            "predicates": _relation_predicates(patch, diff, {}, {}),
        }
    pairs = patch.source_pairs()
    old_files = {name: buggy for name, buggy, _ in pairs}
    new_files = {name: patched for name, _, patched in pairs}
    buggy = _run_joern_source(old_files, "buggy", config, root)
    patched = _run_joern_source(new_files, "patched", config, root)
    success = buggy.get("success", False) and patched.get("success", False)
    return {
        "success": success,
        "buggy": buggy,
        "patched": patched,
        "predicates": _relation_predicates(patch, diff, buggy, patched),
    }


def _target(diff: dict[str, Any], gumtree: dict[str, Any], patch: PatchInput, failure: dict[str, Any]) -> dict[str, Any]:
    stack_methods = set(failure.get("stack_methods", []))
    candidates = []
    for index, action in enumerate(gumtree.get("features", {}).get("actions", [])):
        label = action.get("label", "")
        score = 0
        if patch.buggy_line and action.get("line") == patch.buggy_line:
            score = 3
        elif action.get("enclosing_method") in stack_methods:
            score = 2
        elif patch.buggy_method and action.get("enclosing_method") in METHOD.findall(patch.buggy_method):
            score = 1
        candidates.append(
            (
                score,
                action,
            )
        )
    if candidates:
        highest = max(score for score, _ in candidates)
        selected_actions = [action for score, action in candidates if score == highest]
        files = {item.get("source", "") for item in selected_actions}
        methods = {item.get("enclosing_method", "") for item in selected_actions}
        if len(selected_actions) > 1 and len(files) == 1 and len(methods) == 1 and "" not in methods:
            method = next(iter(methods))
            lines = [item["line"] for item in selected_actions if item.get("line") is not None]
            starts = [item["start"] for item in selected_actions if item.get("start") is not None]
            return {
                "file": next(iter(files)),
                "node_type": "MethodDeclaration",
                "region_kind": "method",
                "label": method,
                "line": min(lines) if lines else None,
                "enclosing_method": method,
                "source_order": min(starts) if starts else None,
            }
        selected = min(
            selected_actions,
            key=lambda action: (
                int(action.get("length", 10**9)),
                int(action.get("start", 10**9)),
            ),
        )
        return {
            "file": selected.get("source", ""),
            "node_type": selected.get("node_type", ""),
            "region_kind": selected.get("region_kind", ""),
            "label": selected.get("label", ""),
            "line": selected.get("line"),
            "enclosing_method": selected.get("enclosing_method", ""),
            "source_order": selected.get("start"),
        }
    return {
        "file": patch.buggy_file or (diff["files"][0]["file"] if diff["files"] else ""),
        "node_type": "changed_region",
        "label": "\n".join(diff["added_lines"][:3] or diff["removed_lines"][:3]),
        "source_order": patch.buggy_line or None,
    }


def analyze_static_facts(patch: PatchInput, config: ADALConfig) -> dict[str, Any]:
    started = time.perf_counter()
    diff = _diff_facts(patch)
    failure = _failure_predicates(patch)
    with tempfile.TemporaryDirectory(prefix="adal-") as temp:
        root = Path(temp)
        gumtree = _run_gumtree(patch, config, root)
        joern = _run_joern(patch, diff, config, root)
    diff["gumtree"] = gumtree.get("features", {})
    diff["gumtree_success"] = bool(gumtree.get("success"))
    diff["change_target"] = _target(diff, gumtree, patch, failure)
    types = set(diff["patch_types"])
    for action in diff["gumtree"].get("actions", []):
        node_type = action.get("node_type", "").lower()
        operation = action["action"]
        if action.get("enclosing_method"):
            types.add("method_change")
        if "methoddeclaration" in node_type:
            types.add("method_signature_change")
        if "methodinvocation" in node_type:
            types.add("api_call_change")
        if "returnstatement" in node_type:
            types.add("return_change")
        if "ifstatement" in node_type and operation.startswith("insert"):
            types.add("guard_insertion")
        if operation in {"delete-node", "delete-tree"}:
            types.add("deletion")
    if patch.buggy_method and any(line.strip() and line.strip() in patch.buggy_method for line in diff["removed_lines"]):
        types.add("method_change")
    diff["patch_types"] = sorted(types)
    cpg = joern.get("predicates", {})
    if patch.static_facts:
        supplied = json.loads(json.dumps(patch.static_facts))
        diff.update(supplied.get("diff_ast_facts", {}))
        failure.update(supplied.get("failure_predicates", {}))
        cpg.update(supplied.get("cpg_predicates", {}))
    return {
        "diff_ast_facts": diff,
        "cpg_predicates": cpg,
        "failure_predicates": failure,
        "analysis_status": {
            "gumtree_success": bool(gumtree.get("success")),
            "joern_success": bool(joern.get("success")),
            "gumtree_reason": gumtree.get("reason", ""),
            "joern_reason": joern.get("reason", ""),
            "precomputed_facts_supplied": bool(patch.static_facts),
        },
        "timing": {
            "deterministic_seconds": time.perf_counter() - started,
            "gumtree_seconds": gumtree.get("duration_seconds", 0.0),
            "buggy_cpg_seconds": joern.get("buggy", {}).get("parse", {}).get("duration_seconds", 0.0),
            "patched_cpg_seconds": joern.get("patched", {}).get("parse", {}).get("duration_seconds", 0.0),
            "buggy_pdg_seconds": joern.get("buggy", {}).get("pdg_export", {}).get("duration_seconds", 0.0),
            "patched_pdg_seconds": joern.get("patched", {}).get("pdg_export", {}).get("duration_seconds", 0.0),
        },
    }
