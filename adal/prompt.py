from __future__ import annotations

import json
import re
from typing import Any


ALLOWED_REPAIR_SCHEMAS = (
    "expression_role_repair", "loop_bound_role_repair", "api_result_role_repair",
    "early_return_contract_repair", "guarded_exception_contract_repair",
    "constructor_context_repair", "side_effect_free_deletion_repair",
    "null_or_guard_repair",
)
ALLOWED_REBUTTAL_OPERATORS = tuple(f"RO{index}" for index in range(1, 12))

SYSTEM_PROMPT = """You are an expert in Java program analysis and defeasible reasoning. Given a candidate patch, its correctness claims, and the supplied program evidence, examine the supporting arguments, rebuttals, and evidence gaps. Propose structured revisions to the evidence graph, grounding each revision in the supplied evidence. Keep IMMUTABLE FACTS unchanged. Do not issue a correct/incorrect label or assign defeated states. Ground each update in an input predicate and an exact code or failure context span. Keep unsupported conclusions unresolved."""

USER_PROMPT_TEMPLATE = """Task: revise the defeasible evidence for one candidate Java patch.

INPUTS
Patch metadata and claim: {{PATCH_METADATA_AND_CLAIM}}
Allowed schemas and operators: {{ALLOWLISTS}}
Immutable diff, AST, CPG, and failure facts: {{IMMUTABLE_FACTS}}
Current support, rebuttals, and missing evidence: {{EVIDENCE}}
Patch diff, buggy method, and failure context: {{CODE_AND_FAILURE}}

REVISION RULES
1. Use undercut when evidence weakens an inference; use
   contradicted only when an immutable fact disproves it.
2. New rebuttals target existing claims or inference relations
   and are initially unresolved.
3. Resolve missing evidence only when the required fact appears
   in IMMUTABLE FACTS.
4. ranking_boost requires a grounded medium/high rebuttal to be
   undercut or contradicted, or a program-supported repair schema.
5. promote_to_candidate additionally requires a medium/high
   rebuttal to be contradicted and no new high-severity rebuttal.
6. Otherwise, use no_change.

OUTPUT JSON
Return exactly one JSON object with:
patch_uid;
semantic_schema_candidates:
  [{schema, status, confidence, grounding}];
rebuttal_refinements:
  [{rebuttal_id, original_status, new_status, severity,
    reason, confidence, grounding}];
new_rebuttals:
  [{operator, target_id, severity, hypothesis, grounding}];
missing_evidence_updates:
  [{missing_id, new_status, criticality, reason, grounding}];
suggested_action; action_confidence.

Allowed values:
Schema status: supported | unsupported | uncertain
Rebuttal status: supported | undercut | contradicted | unresolved
Missing-evidence status: resolved | unresolved
Severity/confidence: low | medium | high
Action: no_change | ranking_boost | promote_to_candidate"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def build_user_prompt(
    patch_uid: str,
    metadata: dict[str, Any],
    claim: dict[str, Any],
    diff_ast_facts: dict[str, Any],
    cpg_predicates: dict[str, Any],
    failure_predicates: dict[str, Any],
    failure_context: str,
    supporting_evidence: list[dict[str, Any]],
    rebuttals: list[dict[str, Any]],
    missing_evidence: list[dict[str, Any]],
    patch_diff_context: str,
    buggy_method_context: str,
    predicates: list[dict[str, Any]] | None = None,
    undercutters: list[dict[str, Any]] | None = None,
    relations: list[dict[str, Any]] | None = None,
) -> str:
    from .evidence import REBUTTAL_LIBRARY, SCHEMA_PREDICATES
    # Do not forward arbitrary metadata (labels and old learned scores in particular).
    public_metadata = {
        name: metadata[name] for name in
        ("bug_uid", "project", "tool", "patch_rank", "pool_size") if name in metadata
    }
    public_metadata["patch_uid"] = patch_uid
    blocks = {
        "PATCH_METADATA_AND_CLAIM": {"metadata": public_metadata, "claim": claim},
        "ALLOWLISTS": {
            "repair_schemas": ALLOWED_REPAIR_SCHEMAS,
            "rebuttal_operators": ALLOWED_REBUTTAL_OPERATORS,
            "schema_required_predicates": SCHEMA_PREDICATES,
            "rebuttal_operator_definitions": [
                {"operator": uid, "activation_conditions": sorted(activation), "targets": targets,
                 "severity": severity, "name": name, "checking_predicate": predicate,
                 "hypothesis": hypothesis}
                for uid, activation, targets, severity, name, predicate, hypothesis in REBUTTAL_LIBRARY
            ],
        },
        "IMMUTABLE_FACTS": {
            "diff_ast_facts": diff_ast_facts, "cpg_predicates": cpg_predicates,
            "failure_predicates": failure_predicates, "predicates": predicates or [],
        },
        "EVIDENCE": {
            "supporting_evidence": supporting_evidence, "rebuttals": rebuttals,
            "undercutters": undercutters or [], "relations": relations or [],
            "missing_evidence": missing_evidence,
        },
        "CODE_AND_FAILURE": {
            "patch_diff": patch_diff_context, "buggy_method": buggy_method_context,
            "failure_context": failure_context,
        },
    }
    # One substitution pass; placeholders inside source code remain literal data.
    return re.sub(r"\{\{([A-Z_]+)\}\}", lambda match: _json(blocks[match.group(1)]), USER_PROMPT_TEMPLATE)
