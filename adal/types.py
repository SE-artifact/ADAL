from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PatchInput:
    patch_uid: str
    bug_uid: str
    buggy_source: str = ""
    patched_source: str = ""
    failure_context: str = ""
    triggering_test: str = ""
    failure_message: str = ""
    stack_trace: str = ""
    buggy_files: dict[str, str] = field(default_factory=dict)
    patched_files: dict[str, str] = field(default_factory=dict)
    buggy_file: str = ""
    buggy_line: int = 0
    buggy_method: str = ""
    truth_label: str = ""
    project: str = ""
    tool: str = ""
    patch_rank: int = 0
    pool_size: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    static_facts: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PatchInput":
        fields = set(cls.__dataclass_fields__)
        return cls(**{key: value[key] for key in fields if key in value})

    def source_pairs(self) -> list[tuple[str, str, str]]:
        old_files = dict(self.buggy_files)
        new_files = dict(self.patched_files)
        if not old_files and not new_files and (self.buggy_source or self.patched_source):
            name = self.buggy_file or "Input.java"
            old_files[name] = self.buggy_source
            new_files[name] = self.patched_source
        names = sorted(set(old_files) | set(new_files))
        return [(name, old_files.get(name, ""), new_files.get(name, "")) for name in names]

    def observed_failure(self) -> str:
        values = [
            self.triggering_test,
            self.failure_message,
            self.stack_trace,
            self.failure_context,
        ]
        return "\n".join(dict.fromkeys(value for value in values if value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
