from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from .config import ADALConfig, METHOD_VERSION
from .evidence import construct_evidence_graph, extract_features, integrate_revision
from .llm import LanguageModel, no_change, should_revise
from .model import ADALAggregator, lopo_predictions, semantic_signal
from .prompt import build_user_prompt
from .static_analysis import analyze_static_facts
from .types import PatchInput


class ADAL:
    def __init__(
        self,
        config: ADALConfig | None = None,
        aggregator: ADALAggregator | None = None,
    ):
        self.config = config or ADALConfig()
        self.aggregator = aggregator
        self.language_model = LanguageModel(self.config.models)

    def analyze_patch(self, value: PatchInput | dict[str, Any]) -> dict[str, Any]:
        return self.analyze_pool([value])[0]

    def _analyze_evidence(self, value: PatchInput | dict[str, Any]) -> dict[str, Any]:
        patch = value if isinstance(value, PatchInput) else PatchInput.from_dict(value)
        started = time.perf_counter()
        static = analyze_static_facts(patch, self.config)
        graph = construct_evidence_graph(patch, static)
        initial_graph = deepcopy(graph)
        request_uid = f"patch-{uuid4()}"
        prompt = build_user_prompt(
            patch_uid=request_uid,
            metadata={
                **patch.metadata,
                "patch_uid": patch.patch_uid,
                "bug_uid": patch.bug_uid,
                "project": patch.project,
                "tool": patch.tool,
                "patch_rank": patch.patch_rank,
                "pool_size": patch.pool_size,
            },
            claim=graph["claim"],
            diff_ast_facts=graph["immutable_facts"]["diff_ast_facts"],
            cpg_predicates=graph["immutable_facts"]["cpg_predicates"],
            failure_predicates=graph["immutable_facts"]["failure_predicates"],
            failure_context=patch.observed_failure(),
            supporting_evidence=graph["supporting_evidence"],
            rebuttals=[
                item
                for item in graph["rebuttals"]
                if item["status"] in {"supported", "unresolved"}
            ],
            missing_evidence=graph["missing_evidence"],
            patch_diff_context=graph["immutable_facts"]["diff_ast_facts"].get("unified_diff", ""),
            buggy_method_context=patch.buggy_method,
            predicates=graph["immutable_facts"]["predicates"],
            undercutters=graph["undercutters"],
            relations=graph["relations"],
        )
        revision = no_change(request_uid, "revision_disabled_or_not_required")
        revision_started = time.perf_counter()
        if self.config.analysis.run_llm and should_revise(graph):
            revision = self.language_model.revise(request_uid, prompt, graph)
            revision["request_patch_uid"] = request_uid
            revision["patch_uid"] = patch.patch_uid
            graph = integrate_revision(graph, revision)
            graph["feature_vector"] = extract_features(patch, graph)
        revision["request_patch_uid"] = request_uid
        revision["patch_uid"] = patch.patch_uid
        result = {
            "method": " ",
            "method_version": METHOD_VERSION,
            "patch_uid": patch.patch_uid,
            "bug_uid": patch.bug_uid,
            "truth_label": patch.truth_label,
            "project": patch.project,
            "tool": patch.tool,
            "patch_rank": patch.patch_rank,
            "pool_size": patch.pool_size,
            "feature_vector": graph["feature_vector"],
            "evidence_graph": graph,
            "initial_evidence_graph": initial_graph,
            "static_analysis_status": static.get("analysis_status", {}),
            "semantic_revision": revision,
            "initial_evidence_retained": not revision.get("validated", False),
            "timing": {
                **static.get("timing", {}),
                "semantic_revision_seconds": time.perf_counter() - revision_started,
                "total_seconds": time.perf_counter() - started,
            },
        }
        return result

    def analyze_pool(self, values: list[PatchInput | dict[str, Any]]) -> list[dict[str, Any]]:
        if not values:
            return []
        patches = [value if isinstance(value, PatchInput) else PatchInput.from_dict(value) for value in values]
        if len({(patch.project, patch.tool, patch.bug_uid) for patch in patches}) != 1:
            raise ValueError("analyze_pool expects one project/bug/APR-tool candidate pool.")
        patches.sort(key=lambda patch: (patch.patch_rank, patch.patch_uid))
        rows = []
        for position, patch in enumerate(patches):
            patch.pool_size = len(patches)
            patch.metadata = {**patch.metadata, "pool_position": position}
            rows.append(self._analyze_evidence(patch))
        predicted = self._predict(rows)
        if self.aggregator:
            predicted.sort(key=lambda row: (
                -row["ranking_score"], -row["filter_probability"],
                self._surviving_rebuttals(row, "high"), self._surviving_rebuttals(row, "medium"),
                row["patch_rank"], row["patch_uid"],
            ))
        retained_position = 0
        for rank, row in enumerate(predicted, 1):
            row["rank_all_candidates"] = rank if self.aggregator else None
            if row.get("predicted_label") == "correct":
                retained_position += 1
                row["rank_after_filtering"] = retained_position
            else:
                row["rank_after_filtering"] = None
            self._save_artifact(row)
        return predicted

    def prepare_training_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in rows:
            if row.get("truth_label") not in {"correct", "incorrect"}:
                raise ValueError("Training needs a correct/incorrect truth_label.")
            key = str(row.get("project", "")), str(row.get("tool", "")), str(row["bug_uid"])
            grouped.setdefault(key, []).append(row)
        output = []
        for pool in grouped.values():
            ordered = sorted(pool, key=lambda row: (int(row.get("patch_rank", 0)), str(row["patch_uid"])))
            for position, row in enumerate(ordered):
                if "feature_vector" in row:
                    if row.get("method_version") != METHOD_VERSION:
                        raise ValueError("Cached features must carry the current method_version; regenerate historical evidence first.")
                    output.append(row)
                else:
                    record = dict(row)
                    record["pool_size"] = len(pool)
                    record["metadata"] = {**row.get("metadata", {}), "pool_position": position}
                    output.append(self._analyze_evidence(record))
        return output

    def train(self, rows: list[dict[str, Any]]) -> ADALAggregator:
        prepared = self.prepare_training_rows(rows)
        self.aggregator = ADALAggregator(self.config.analysis.random_seed, self.config.aggregation).fit(prepared)
        return self.aggregator

    def train_lopo(
        self, rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, ADALAggregator]]:

        prepared = self.prepare_training_rows(rows)
        return lopo_predictions(prepared, self.config.analysis.random_seed, self.config.aggregation)

    def _predict(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.aggregator:
            return self.aggregator.predict(rows)

        output = []
        for row in rows:
            item = dict(row)
            item.update({
                "assessment_status": "evidence_only", "filter_probability": None,
                "ranking_score": None, "predicted_label": None,
                "semantic_revision_signal": semantic_signal(row),
            })
            output.append(item)
        return output

    @staticmethod
    def _surviving_rebuttals(row: dict[str, Any], severity: str) -> int:
        return sum(
            item.get("severity") == severity
            and item.get("status") in {"supported", "unresolved"}
            for item in row.get("evidence_graph", {}).get("rebuttals", [])
        )

    def _save_artifact(self, result: dict[str, Any]) -> None:
        root = self.config.paths.artifact_root
        if not root:
            return
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in str(result["patch_uid"])
        )
        (directory / f"{safe_name}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
