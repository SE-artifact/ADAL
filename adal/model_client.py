from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

import requests

from .config import ModelConfig


@dataclass
class ModelResponse:
    content: str
    provider: str
    model: str
    finish_reason: str
    reasoning: str
    usage: dict[str, Any]
    raw_response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelClient:


    def __init__(self, config: ModelConfig):
        self.config = config

    def settings(self) -> dict[str, Any]:
        provider = self.config.provider.lower()
        models = {
            "qwen3-14b": self.config.qwen_model,
            "deepseek-v3": self.config.deepseek_model,
            "gpt-4.1": self.config.openai_model,
        }
        return {
            "provider": provider, "requested_model": models.get(provider, ""),
            "backend": "vllm" if provider == "qwen3-14b" else "api",
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
            "timeout_seconds": self.config.timeout_seconds,
            "enable_thinking": self.config.qwen_enable_thinking if provider == "qwen3-14b" else None,
        }

    def complete(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        provider = self.config.provider.lower()
        if provider == "qwen3-14b":
            if not self.config.vllm_base_url:
                raise ValueError("vllm_base_url is empty.")
            endpoint = self.config.vllm_base_url.rstrip("/") + "/chat/completions"
            return self._chat_completion(
                endpoint, self.config.qwen_model, self.config.vllm_api_key,
                system_prompt, user_prompt,
                {"chat_template_kwargs": {"enable_thinking": self.config.qwen_enable_thinking}},
            )
        if provider == "deepseek-v3":
            if not self.config.deepseek_api_key:
                raise ValueError("DeepSeek API key is empty.")
            return self._chat_completion(
                self.config.deepseek_endpoint, self.config.deepseek_model,
                self.config.deepseek_api_key, system_prompt, user_prompt,
            )
        if provider == "gpt-4.1":
            return self._openai_response(system_prompt, user_prompt)
        raise ValueError("Unsupported model provider: " + self.config.provider)

    def _chat_completion(
        self, endpoint: str, model: str, api_key: str, system_prompt: str,
        user_prompt: str, extra_body: dict[str, Any] | None = None,
    ) -> ModelResponse:
        if not model or not endpoint:
            raise ValueError("Model name and service endpoint are required.")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "response_format": {"type": "json_object"}, "stream": False,
        }
        body.update(extra_body or {})
        response = requests.post(endpoint, headers=headers, json=body, timeout=self.config.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        return ModelResponse(
            content=message.get("content") or "", provider=self.config.provider,
            model=data.get("model", model), finish_reason=choice.get("finish_reason") or ("" if choices else "failed"),
            reasoning=message.get("reasoning") or message.get("reasoning_content") or "",
            usage=data.get("usage") or {}, raw_response=data,
        )

    def _openai_response(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key is empty.")
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": "Bearer " + self.config.openai_api_key, "Content-Type": "application/json"},
            json={
                "model": self.config.openai_model, "instructions": system_prompt,
                "input": user_prompt, "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_output_tokens,
                "text": {"format": {"type": "json_object"}},
            }, timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        parts = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    parts.append(content.get("text", ""))
        return ModelResponse(
            content=data.get("output_text") or "".join(parts), provider=self.config.provider,
            model=data.get("model", self.config.openai_model), finish_reason=data.get("status", ""),
            reasoning="", usage=data.get("usage") or {}, raw_response=data,
        )


def parse_json_object(response: ModelResponse) -> dict[str, Any]:

    if response.finish_reason in {"length", "incomplete", "content_filter", "failed", "cancelled"}:
        raise ValueError("The model did not complete its answer: " + response.finish_reason)
    if not response.content.strip():
        raise ValueError("The model returned no final JSON content.")
    value = json.loads(response.content)
    if not isinstance(value, dict):
        raise ValueError("The model must return exactly one JSON object.")
    return value
