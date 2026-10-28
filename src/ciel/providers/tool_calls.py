"""Validate native tool metadata without turning arguments into visible reply text."""

import json
import math
from typing import Any

from ciel.providers.base import MAX_TOOL_CALLS, ToolCall, ToolCallDelta


def response_object(value: Any, label: str) -> dict[str, Any]:
    """Require an API object while keeping malformed provider values out of errors."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def reply_content(value: Any) -> str:
    """Accept nullable text, never stringify structured content or tool arguments."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("Provider reply content must be text or null")
    return value


def openai_choice(value: Any) -> dict[str, Any] | None:
    """Read the first OpenAI-compatible choice, allowing usage-only stream events."""
    data = response_object(value, "Provider response")
    choices = data.get("choices", [])
    if not isinstance(choices, list):
        raise ValueError("Provider choices must be a list")
    if not choices:
        return None
    return response_object(choices[0], "Provider choice")


def _call_objects(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_TOOL_CALLS:
        raise ValueError("Provider tool calls must be a list of at most 128 calls")
    calls = []
    for item in value:
        call = response_object(item, "Provider tool call")
        if "type" in call and call["type"] != "function":
            raise ValueError("Provider tool call type must be function")
        calls.append(call)
    return calls


def openai_tool_calls(value: Any) -> tuple[ToolCall, ...]:
    """Parse complete calls while retaining the exact JSON argument string."""
    calls = []
    for call in _call_objects(value):
        function = response_object(call.get("function"), "Provider tool function")
        calls.append(ToolCall(
            id=call.get("id", ""),
            name=function.get("name", ""),
            arguments=function.get("arguments", ""),
        ))
    if len({call.id for call in calls}) != len(calls):
        raise ValueError("Native tool call IDs must be unique within a response")
    return tuple(calls)


def openai_tool_deltas(value: Any) -> tuple[ToolCallDelta, ...]:
    """Keep indexed function/name/argument fragments separate from reply text."""
    deltas = []
    for call in _call_objects(value):
        function = response_object(call.get("function", {}), "Provider tool function")
        deltas.append(ToolCallDelta(
            index=call.get("index"),
            id=call.get("id", ""),
            name=function.get("name", ""),
            arguments=function.get("arguments", ""),
        ))
    return tuple(deltas)


def _json_value(value: Any, depth: int = 0) -> None:
    """Reject non-JSON Ollama arguments rather than coercing Python objects."""
    if depth > 64:
        raise ValueError("Ollama tool arguments are nested too deeply")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item, depth + 1)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_value(item, depth + 1)
        return
    raise ValueError("Ollama tool arguments must contain JSON values")


def ollama_tool_deltas(value: Any, start_index: int = 0) -> tuple[ToolCallDelta, ...]:
    """Normalize Ollama's complete native calls to the shared indexed delta form."""
    deltas = []
    for offset, call in enumerate(_call_objects(value)):
        function = response_object(call.get("function"), "Ollama tool function")
        index = call.get("index", function.get("index", start_index + offset))
        arguments = function.get("arguments", "")
        if isinstance(arguments, dict):
            _json_value(arguments)
            arguments = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
        elif not isinstance(arguments, str):
            raise ValueError("Ollama tool arguments must be an object or JSON text")
        # Native Ollama calls usually omit IDs. Their indexes identify calls for
        # this response and remain stable until the matching result is appended.
        delta = ToolCallDelta(
            index=index,
            id=call.get("id", f"ollama_call_{index}"),
            name=function.get("name", ""),
            arguments=arguments,
        )
        # Unlike OpenAI fragments, Ollama emits complete function metadata.
        ToolCall(id=delta.id, name=delta.name, arguments=delta.arguments)
        deltas.append(delta)
    return tuple(deltas)


def ollama_tool_calls(value: Any) -> tuple[ToolCall, ...]:
    """Normalize complete native Ollama calls to immutable response calls."""
    calls = tuple(
        ToolCall(id=delta.id, name=delta.name, arguments=delta.arguments)
        for delta in ollama_tool_deltas(value)
    )
    if len({call.id for call in calls}) != len(calls):
        raise ValueError("Native tool call IDs must be unique within a response")
    return calls
