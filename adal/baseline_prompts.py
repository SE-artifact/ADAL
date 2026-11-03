from __future__ import annotations

import re


PATCH_SYSTEM_PROMPT = """You are an experienced software reviewer. Assess whether the candidate patch correctly resolves the observed failure using only the supplied artifacts. Do not assume any unavailable specification, test, or developer patch."""

PATCH_USER_PROMPT_TEMPLATE = """[Task]
Determine whether the candidate patch fixes the observed failure while preserving the behavior implied by the available code context.
[Triggering Test] {triggering_test}
[Failure Message] {failure_message}
[Stack Trace] {stack_trace}
[Buggy Code Context] {buggy_context}
[Candidate Patch] {patch_diff}
[Assessment Requirements]
1. Identify the behavior required by the triggering test and failure.
2. Check whether the edit addresses the corresponding failure cause.
3. Check for behavioral or interface violations visible in the context.
4. Do not use information outside the supplied artifacts, and identify any unresolved evidence in the assessment.
[Response]
Return exactly one JSON object:
{
  "reason": "<concise assessment>",
  "label": "<correct or incorrect>"
}
The label must be exactly "correct" or "incorrect"."""

SPECROVER_SYSTEM_PROMPT = """You are a software patch reviewer. Decide whether the candidate patch resolves the observed failure. Base every conclusion on the supplied artifacts only."""

SPECROVER_USER_PROMPT_TEMPLATE = """An observed failure and a candidate patch are provided. The triggering test or the patch may encode an incomplete understanding of the defect.
[Observed Failure]
Failure message: {failure_message}
Stack trace: {stack_trace}
[Triggering Test] {triggering_test}
[Buggy Code Context] {buggy_context}
[Candidate Patch] {patch_diff}
[Review Procedure]
1. Determine whether the triggering test and failure information consistently identify the defective behavior.
2. Determine whether the candidate patch addresses that behavior.
3. Check whether the patch introduces a visible regression, excessive change, or unsupported behavior.
4. Do not use information outside the supplied artifacts, and identify any unresolved evidence in the assessment.
[Response]
Return exactly one JSON object:
{
  "test_assessment": "<concise assessment of the failure message>",
  "patch_assessment": "<concise assessment of the candidate patch>",
  "label": "<correct or incorrect>"
}
The label must be exactly "correct" or "incorrect"."""

PROMPTS = {
    "patch": (PATCH_SYSTEM_PROMPT, PATCH_USER_PROMPT_TEMPLATE),
    "specrover": (SPECROVER_SYSTEM_PROMPT, SPECROVER_USER_PROMPT_TEMPLATE),
}
ARTIFACT_FIELDS = ("triggering_test", "failure_message", "stack_trace", "buggy_context", "patch_diff")


def build_baseline_prompt(method: str, artifacts: dict[str, str]) -> tuple[str, str]:
    system, template = PROMPTS[method.lower()]
    # Only these five artifact fields are substituted. Replacement is single-pass,
    # so braces in Java/JSON source or supplied artifacts remain unchanged.
    values = {name: artifacts.get(name) or "[not provided]" for name in ARTIFACT_FIELDS}
    user = re.sub(
        r"\{(triggering_test|failure_message|stack_trace|buggy_context|patch_diff)\}",
        lambda match: values[match.group(1)], template,
    )
    return system, user
