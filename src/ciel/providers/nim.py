"""NVIDIA NIM LLM provider implementation."""

import json
import time
from collections.abc import Iterator
from typing import Any

import httpx

from ciel.config import DEFAULT_NIM_FALLBACK_MODEL, ProviderConfig, get_settings
from ciel.latency import http_extensions, mark
from ciel.logging import get_logger
from ciel.providers.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    ToolCall,
    ToolCallAccumulator,
    require_text_messages,
)
from ciel.providers.streaming import streaming_client
from ciel.providers.tool_calls import (
    openai_choice,
    openai_tool_calls,
    openai_tool_deltas,
    reply_content,
    response_object,
)

logger = get_logger("providers.nim")
STREAM_READ_TIMEOUT_SECONDS = 12.0
MIN_REASONING_COMPLETION_TOKENS = 8192


class _FinalAnswerStream:
    """Discard exterior thinking tags even when a server splits them across events."""

    _TAGS = ("think", "thought", "reasoning", "reflection")

    def __init__(self) -> None:
        self._pending = ""
        self._closing = ""
        self._visible = False

    def feed(self, text: str) -> str:
        """Return visible text while retaining only a possible boundary tag prefix."""
        if self._visible:
            return text
        self._pending += text
        while True:
            if self._closing:
                end = self._pending.lower().find(self._closing)
                if end < 0:
                    self._pending = self._pending[-(len(self._closing) - 1):]
                    return ""
                self._pending = self._pending[end + len(self._closing):].lstrip()
                self._closing = ""
            candidate = self._pending.lstrip().lower()
            for tag in self._TAGS:
                opening = f"<{tag}>"
                if candidate.startswith(opening):
                    self._pending = self._pending.lstrip()[len(opening):]
                    self._closing = f"</{tag}>"
                    break
            else:
                if not candidate or any(f"<{tag}>".startswith(candidate) for tag in self._TAGS):
                    return ""
                # Once a final answer or JSON starts, preserve it verbatim. A JSON
                # string may legitimately contain literal thinking tags.
                self._visible = True
                visible, self._pending = self._pending, ""
                return visible


class NimProvider(LLMProvider):
    """Client for NVIDIA NIM (Inference Microservice) Cloud API."""

    name: str = "nim"
    default_model: str = ProviderConfig().nim_model
    base_url: str = "https://integrate.api.nvidia.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        *,
        config: ProviderConfig | None = None,
    ) -> None:
        providers = config if config is not None else get_settings().providers
        if api_key:
            self.api_key = api_key
        elif providers.nim_api_key:
            self.api_key = providers.nim_api_key.get_secret_value()
        else:
            self.api_key = None

        self.default_model = model or providers.nim_model or self.default_model
        self.complex_model = providers.nim_complex_model
        self.fallback_model = providers.nim_fallback_model or DEFAULT_NIM_FALLBACK_MODEL
        self.timeout = timeout

    def is_configured(self) -> bool:
        """Return True if NIM API key is configured."""
        return bool(self.api_key and len(self.api_key.strip()) > 0)

    def _get_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ValueError("NVIDIA NIM API key is not configured.")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def health_check(self) -> bool:
        """Verify API connectivity with a lightweight model list request."""
        if not self.is_configured():
            return False
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.base_url}/models", headers=self._get_headers())
                return res.status_code == 200
        except Exception as err:
            logger.debug("NIM health check failed: %s", err)
            return False

    RELIABLE_BACKUP_MODEL: str = DEFAULT_NIM_FALLBACK_MODEL

    @staticmethod
    def _model_payload(payload: dict[str, Any], model: str) -> dict[str, Any]:
        """Translate the routing snapshot into supported Nemotron template options."""
        prepared = {**payload, "model": model}
        reasoning_enabled = prepared.pop("reasoning_enabled", None)
        if model.startswith(("nvidia/nemotron-3-", "nvidia/nemotron-3.5-")):
            template = prepared.get("chat_template_kwargs") or {}
            prepared["chat_template_kwargs"] = {"enable_thinking": False, **template}
            if reasoning_enabled is not None:
                prepared["chat_template_kwargs"]["enable_thinking"] = reasoning_enabled
            if prepared["chat_template_kwargs"]["enable_thinking"]:
                # NIM counts hidden reasoning against the completion limit. A short
                # voice-answer budget must still leave space for the final answer.
                prepared["max_tokens"] = max(
                    prepared.get("max_tokens", 0), MIN_REASONING_COMPLETION_TOKENS,
                )
        else:
            # Nemotron-specific flags must not break another model or its fallback.
            prepared.pop("chat_template_kwargs", None)
        return prepared

    @classmethod
    def _reply_text(cls, data: dict[str, Any]) -> str:
        choice = openai_choice(data)
        if choice is None:
            return ""
        message = response_object(choice.get("message", {}), "NIM message")
        # Never promote reasoning_content/reasoning into the spoken answer.
        return cls.clean_reasoning(message.get("content"))

    @staticmethod
    def _reply_tool_calls(data: dict[str, Any]) -> tuple[ToolCall, ...]:
        choice = openai_choice(data)
        if choice is None:
            return ()
        message = response_object(choice.get("message", {}), "NIM message")
        return openai_tool_calls(message.get("tool_calls"))

    def send(
        self,
        messages: list[ChatMessage] | str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Request a visible answer, retry transient errors, and try one backup model."""
        msgs = self.normalize_messages(messages)
        require_text_messages(msgs)
        target_model = model or self.default_model
        allow_model_fallback = kwargs.pop("allow_model_fallback", True)
        max_attempts = kwargs.pop("max_attempts", 3)
        if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be an integer from 1 to 3")
        fallback_model = (
            self.fallback_model or self.RELIABLE_BACKUP_MODEL
            if allow_model_fallback else target_model
        )
        payload = {
            "model": target_model,
            "messages": [m.to_dict() for m in msgs],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            **kwargs,
        }
        retryable = (429, 500, 502, 503, 504)
        start_t = time.monotonic()
        with httpx.Client(timeout=self.timeout) as client:

            def complete(active_model: str, attempts: int) -> dict[str, Any]:
                request = self._model_payload(payload, active_model)
                for attempt in range(attempts):
                    try:
                        mark("provider_request_started", provider=self.name, model=active_model)
                        response = client.post(
                            f"{self.base_url}/chat/completions",
                            headers=self._get_headers(),
                            json=request,
                            extensions=http_extensions(),
                        )
                        mark("provider_http_response", status=response.status_code)
                        response.raise_for_status()
                        return response.json()
                    except httpx.HTTPStatusError as err:
                        if err.response.status_code not in retryable or attempt == attempts - 1:
                            raise
                        wait = 0.5 * (2**attempt)
                        logger.warning(
                            "NIM '%s' HTTP %d; retrying in %.1fs",
                            active_model,
                            err.response.status_code,
                            wait,
                        )
                        time.sleep(wait)
                raise RuntimeError("NIM request attempts exhausted")

            try:
                data = complete(target_model, max_attempts)
            except httpx.HTTPStatusError as err:
                if (
                    err.response.status_code not in (*retryable, 404, 410)
                    or target_model == fallback_model
                ):
                    raise
                logger.warning(
                    "NIM '%s' HTTP %d; trying '%s'.",
                    target_model, err.response.status_code, fallback_model,
                )
                target_model = fallback_model
                data = complete(target_model, 1)

            content = self._reply_text(data)
            tool_calls = self._reply_tool_calls(data)
            if not content and not tool_calls and target_model != fallback_model:
                logger.warning(
                    "NIM '%s' returned no visible answer; trying '%s'.",
                    target_model,
                    fallback_model,
                )
                target_model = fallback_model
                data = complete(target_model, 1)
                content = self._reply_text(data)
                tool_calls = self._reply_tool_calls(data)
            if not content and not tool_calls:
                raise RuntimeError(f"NIM '{target_model}' returned no user-visible answer")

        return LLMResponse(
            content=content,
            model=target_model,
            provider=self.name,
            usage=data.get("usage") or {},
            latency=time.monotonic() - start_t,
            tool_calls=tool_calls,
        )

    @staticmethod
    def clean_reasoning(text: str | None) -> str:
        """Strip internal reasoning tags, self-check blocks, and Nemotron CoT monologue."""
        import re

        if text is None:
            return ""
        if not isinstance(text, str):
            raise ValueError("NIM reply content must be text or null")
        cleaned = text.strip()

        def is_structured_reply(candidate: str) -> bool:
            """Protect JSON string values from conversational cleanup heuristics."""
            fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.I | re.S)
            try:
                parsed = json.loads(fenced.group(1) if fenced else candidate)
            except (ValueError, TypeError):
                return False
            return isinstance(parsed, (dict, list))

        # A machine-readable reply can mention 'Final answer:' or contain literal
        # thinking tags in string values. Only remove exterior reasoning blocks
        # before returning the complete JSON, optionally with its original fence.
        structured = cleaned
        while True:
            if is_structured_reply(structured):
                return structured
            exterior = re.match(
                r"^<(think|thought|reasoning|reflection)>[\s\S]*?</\1>",
                structured, re.IGNORECASE,
            )
            if exterior is None:
                break
            structured = structured[exterior.end():].strip()

        # 1. Remove properly closed XML thinking tags
        cleaned = re.sub(
            r"<(think|thought|reasoning|reflection)>[\s\S]*?</\1>",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        # 2. Handle unclosed opening thinking tags — keep prefix before tag, discard rest
        unclosed = re.search(r"<(think|thought|reasoning|reflection)>", cleaned, re.IGNORECASE)
        if unclosed:
            after = cleaned[unclosed.end() :].strip()
            close = re.search(r"</(think|thought|reasoning|reflection)>", after, re.IGNORECASE)
            if close:
                cleaned = after[close.end() :].strip()
            else:
                cleaned = cleaned[: unclosed.start()].strip()

        cleaned = cleaned.strip()

        # 3. Strip *Self-check:*, *Plan:*, [Analysis], etc. preamble blocks
        cleaned = re.sub(
            r"^\s*[*#_\[]*\s*(?:Self[- ]check|Plan|Analysis|Reasoning|Thought|Evaluation|"
            r"Draft|Pre-computation|Check|Internal|Notes?)[*#_\]:]*[\s\S]*?"
            r"(?=\n\s*\n\s*[A-Za-z0-9\"\'\“\‘]|$)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        # 4. Detect Nemotron-style CoT ending with a "Final X:" marker
        final_marker_match = re.search(
            r"(?:Final\s+(?:decision|call|answer|response|statement)|Decision|Direct\s+response)[:\s]+(.+)$",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if final_marker_match:
            final_part = final_marker_match.group(1).strip()
            quoted_match = re.match(r"^\"(.+)\"$", final_part, re.DOTALL)
            if quoted_match:
                return quoted_match.group(1).strip()
            return final_part

        # 5. If response is still starting with bulleted meta lines (- Banned phrases avoided? etc.)
        if cleaned.startswith("- ") or cleaned.startswith("* ") or cleaned.startswith("• "):
            non_bullet = [
                line.strip()
                for line in cleaned.splitlines()
                if line.strip() and not line.strip().startswith(("-", "*", "•"))
            ]
            if non_bullet:
                cleaned = " ".join(non_bullet).strip()
            else:
                # If everything was just bullet self-checks with no real response
                cleaned = ""

        # 6. Detect raw thinking monologue at start — return last paragraph as reply
        if re.match(
            r"^\s*(Okay[,.]?\s|Hmm[,.]?\s|Alright[,.]?\s|Let me\s|Looking at|"
            r"The user|I need to|I should|I'll analyze)",
            cleaned,
            re.IGNORECASE,
        ):
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", cleaned) if p.strip()]
            if paragraphs:
                return paragraphs[-1]

        return cleaned.strip()

    def stream(
        self,
        messages: list[ChatMessage] | str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Iterator[LLMStreamChunk]:
        """Retry transient failures; switch retired models before any text is emitted."""
        cancel_event = kwargs.pop("cancel_event", None)
        allow_model_fallback = kwargs.pop("allow_model_fallback", True)
        if cancel_event is not None and cancel_event.is_set():
            return
        msgs = self.normalize_messages(messages)
        require_text_messages(msgs)
        target_model = model or self.default_model
        fallback_model = (
            self.fallback_model or self.RELIABLE_BACKUP_MODEL
            if allow_model_fallback else target_model
        )

        payload = {
            "model": target_model,
            "messages": [m.to_dict() for m in msgs],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs,
        }

        RETRYABLE = (429, 500, 502, 503, 504)
        MAX_RETRIES = 3

        emitted = False

        def _iter_stream(client: httpx.Client, active_model: str) -> Iterator[LLMStreamChunk]:
            nonlocal emitted
            calls = ToolCallAccumulator()
            final_answer = _FinalAnswerStream()
            mark("provider_request_started", provider=self.name, model=active_model)
            with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=self._model_payload(payload, active_model),
                extensions=http_extensions(),
            ) as response:
                mark("provider_http_response", status=response.status_code)
                response.raise_for_status()
                for line in response.iter_lines():
                    mark("provider_first_line")
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:") :].strip()
                    mark("provider_first_sse")
                    if data_str == "[DONE]":
                        calls.finish()
                        if emitted:
                            yield LLMStreamChunk(
                                delta="", model=active_model, provider=self.name, is_final=True
                            )
                        break
                    try:
                        chunk_json = json.loads(data_str)
                        choice = openai_choice(chunk_json)
                        if choice is None:
                            continue  # Usage-only SSE event.
                        raw_delta = response_object(choice.get("delta", {}), "NIM delta")
                        if raw_delta.get("reasoning_content") or raw_delta.get("reasoning"):
                            mark("provider_first_model_token", kind="reasoning")
                            mark("provider_first_reasoning")
                        delta = final_answer.feed(reply_content(raw_delta.get("content")))
                        tool_calls = openai_tool_deltas(raw_delta.get("tool_calls"))
                        calls.add(tool_calls)
                        if delta:
                            mark("provider_first_model_token", kind="content")
                            mark("provider_first_content")
                        if tool_calls:
                            mark("provider_first_model_token", kind="tool")
                        if delta or tool_calls:
                            emitted = True
                            yield LLMStreamChunk(
                                delta=delta, model=active_model, provider=self.name, is_final=False,
                                tool_calls=tool_calls,
                            )
                        if choice.get("finish_reason") is not None:
                            calls.finish()
                            if emitted:
                                yield LLMStreamChunk(
                                    delta="", model=active_model, provider=self.name, is_final=True
                                )
                            return
                    except json.JSONDecodeError:
                        continue
                calls.finish()

        stream_timeout = httpx.Timeout(
            self.timeout,
            connect=min(self.timeout, 10.0),
            read=min(self.timeout, STREAM_READ_TIMEOUT_SECONDS),
        )
        with streaming_client(stream_timeout, cancel_event) as client:
            last_err: httpx.HTTPStatusError | None = None
            for attempt in range(MAX_RETRIES):
                if cancel_event is not None and cancel_event.is_set():
                    return
                try:
                    yield from _iter_stream(client, target_model)
                except httpx.HTTPStatusError as err:
                    if emitted:
                        raise
                    last_err = err
                    if err.response.status_code in (404, 410):
                        break  # Retrying a retired or missing model cannot help.
                    if err.response.status_code not in RETRYABLE:
                        raise
                    if attempt == MAX_RETRIES - 1:
                        break
                    wait = 0.5 * (2**attempt)
                    logger.warning(
                        "NIM stream '%s' HTTP %d (attempt %d/%d) — retrying in %.1fs...",
                        target_model,
                        err.response.status_code,
                        attempt + 1,
                        MAX_RETRIES,
                        wait,
                    )
                    if cancel_event is not None:
                        if cancel_event.wait(wait):
                            return
                    else:
                        time.sleep(wait)
                else:
                    if emitted:
                        return
                    break

            if cancel_event is not None and cancel_event.is_set():
                return
            # Exactly one backup attempt, outside the primary retry loop.
            if target_model != fallback_model:
                logger.warning(
                    "NIM stream '%s' unavailable or empty; trying '%s'.",
                    target_model, fallback_model,
                )
                yield from _iter_stream(client, fallback_model)
            elif last_err is not None:
                raise last_err
            if not emitted and (cancel_event is None or not cancel_event.is_set()):
                raise RuntimeError("NIM stream returned no reply text")
