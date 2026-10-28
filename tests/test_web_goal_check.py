"""Independent claim relevance checks preserve the actual task and fail closed."""

import json
import time
from threading import BoundedSemaphore, Event, Lock, Thread
from unittest.mock import Mock

import pytest

from ciel.internet import InternetResult, InternetSource
from ciel.providers.base import ChatMessage, LLMResponse
from ciel.web_answers import relevant_web_claims, verified_web_answer, web_answer_messages


def _case():
    quote = "AuroraClient supports Linux desktop systems."
    unrelated = "AuroraClient version 7 was released on Friday."
    lookup = InternetResult(query="AuroraClient Linux ARM64 support", sources=[
        InternetSource("AuroraClient reference", "https://example.org/reference",
                       content=quote + " " + unrelated),
    ])
    claims = [
        {"text": quote, "quote": quote, "source": 1},
        {"text": unrelated, "quote": unrelated, "source": 1},
    ]
    messages = [
        ChatMessage("system", "Helpful"),
        ChatMessage("user", "Does the client support Linux and ARM64?"),
        ChatMessage("assistant", "Which client?"),
        ChatMessage("user", "AuroraClient."),
    ]
    return json.dumps({"claims": claims}), lookup, messages, claims


def _router(raw='{"relevant_claims":[1]}'):
    router = Mock()
    router.send.return_value = LLMResponse(raw, "model", "provider")
    return router


def test_goal_check_retains_relevant_partial_answer_without_rewriting_claims():
    content, lookup, messages, claims = _case()
    router = _router()

    result = relevant_web_claims(content, lookup, messages, router)

    assert json.loads(result) == {"claims": [claims[0]]}
    request = router.send.call_args.args[0]
    payload = json.loads(request[-1].content)
    assert "WEB GOAL CHECK" in request[0].content
    assert "untrusted" in request[0].content.casefold()
    assert payload["latest_user_request"] == "AuroraClient."
    assert "Linux and ARM64" in str(payload["recent_user_requests"])
    assert payload["resolved_goal"] == lookup.query
    assert payload["claims"][0]["index"] == 1
    assert payload["claims"][0]["quote"] == claims[0]["quote"]
    assert "Which client?" not in request[-1].content
    assert router.send.call_args.kwargs["purpose"] == "controller"
    router.send.assert_called_once()


def test_goal_check_can_reject_genuine_quotes_that_answer_a_different_question():
    content, lookup, messages, _claims = _case()
    router = _router('{"relevant_claims":[]}')
    assert json.loads(relevant_web_claims(content, lookup, messages, router)) == {"claims": []}


@pytest.mark.parametrize("raw", [
    "not JSON", "[]", '{"relevant_claims":"1"}', '{"relevant_claims":[true]}',
    '{"relevant_claims":["1"]}', '{"relevant_claims":[0]}',
    '{"relevant_claims":[3]}', '{"relevant_claims":[1,1]}',
    '{"relevant_claims":[1],"claims":[{"text":"rewritten"}]}',
    '{"relevant_claims":[1],"confidence":0.99}',
])
def test_invalid_judgment_cannot_approve_or_rewrite_claims(raw):
    content, lookup, messages, _claims = _case()
    router = _router(raw)
    assert json.loads(relevant_web_claims(content, lookup, messages, router)) == {"claims": []}
    router.send.assert_called_once()


@pytest.mark.parametrize("raw", [
    '{"claims":[]}', "invalid", '{"claims":[{"text":"made up","source":1,"quote":"no"}]}',
])
def test_abstention_or_invalid_evidence_does_not_make_another_model_call(raw):
    _content, lookup, messages, _claims = _case()
    router = _router()
    assert json.loads(relevant_web_claims(raw, lookup, messages, router)) == {"claims": []}
    router.send.assert_not_called()


def test_full_history_privacy_is_computed_before_recent_user_projection():
    content, lookup, messages, _claims = _case()
    messages = [ChatMessage("system", "Old protected context", local_only=True)] + [
        ChatMessage("assistant", "Old dialogue") for _ in range(20)
    ] + messages
    router = _router()
    relevant_web_claims(content, lookup, messages, router)
    assert all(message.local_only for message in router.send.call_args.args[0])
    assert "Old protected context" not in router.send.call_args.args[0][-1].content


def test_local_prefix_in_old_user_turn_remains_local_after_compaction():
    content, lookup, messages, _claims = _case()
    messages = [ChatMessage("user", "/local protected request")] + [
        ChatMessage("user", "Old dialogue") for _ in range(20)
    ] + messages
    router = _router()
    relevant_web_claims(content, lookup, messages, router)
    assert all(message.local_only for message in router.send.call_args.args[0])


@pytest.mark.parametrize("explicit_flag", [True, False])
def test_report_projection_keeps_privacy_from_older_unprojected_user_turn(explicit_flag):
    _content, lookup, _messages, _claims = _case()
    protected = ("Old protected request" if explicit_flag else "/local Old protected request")
    messages = [ChatMessage("system", "Retain these runtime rules."),
                ChatMessage("user", protected, local_only=explicit_flag)] + [
        ChatMessage("user", f"Recent ordinary request {index}") for index in range(20)
    ] + [ChatMessage("user", "Explain the previously observed project.")]

    request = web_answer_messages(messages, lookup)

    assert all(message.local_only for message in request)
    assert "Old protected request" not in "\n".join(message.content for message in request)
    assert "Retain these runtime rules." in request[0].content
    assert request[-1].content == messages[-1].content


def test_canceled_request_does_not_contact_judge():
    content, lookup, messages, _claims = _case()
    router, canceled = _router(), Event()
    canceled.set()
    assert json.loads(relevant_web_claims(
        content, lookup, messages, router, cancel_event=canceled,
    )) == {"claims": []}
    router.send.assert_not_called()


def test_cancellation_during_judgment_propagates_and_discards_approval():
    content, lookup, messages, _claims = _case()
    router, canceled, events = _router(), Event(), []

    def judge(_messages, **options):
        events.append(options["cancel_event"])
        canceled.set()
        assert events[0].is_set()
        return LLMResponse('{"relevant_claims":[1]}', "model", "provider")

    router.send.side_effect = judge
    assert json.loads(relevant_web_claims(
        content, lookup, messages, router, cancel_event=canceled,
    )) == {"claims": []}
    assert events[0].is_set()


def test_one_deadline_covers_judgment_and_parsing(monkeypatch):
    from ciel import web_answers

    content, lookup, messages, _claims = _case()
    now = [100.0]
    router = _router()
    monkeypatch.setattr(web_answers.time, "monotonic", lambda: now[0])

    def bounded(call, deadline, canceled, *, workers=None):
        assert deadline == 108.0
        assert workers is web_answers._GOAL_WORKERS
        value = call()
        now[0] = deadline + 0.1
        return value

    monkeypatch.setattr(web_answers, "_bounded", bounded)
    assert json.loads(relevant_web_claims(
        content, lookup, messages, router, timeout_seconds=30,
    )) == {"claims": []}
    assert router.send.call_args.kwargs["cancel_event"].is_set()


def test_goal_check_completes_when_normal_retrieval_workers_are_full(monkeypatch):
    from ciel import internet

    content, lookup, messages, claims = _case()
    router = _router()
    workers = BoundedSemaphore(2)
    monkeypatch.setattr(internet, "_WORKERS", workers)
    assert workers.acquire(blocking=False)
    assert workers.acquire(blocking=False)
    try:
        with pytest.raises(TimeoutError, match="still closing"):
            internet._bounded(lambda: None, time.monotonic() + 1, None)
        assert json.loads(relevant_web_claims(content, lookup, messages, router)) == {
            "claims": [claims[0]],
        }
        router.send.assert_called_once()
    finally:
        workers.release()
        workers.release()


def test_goal_check_workers_remain_bounded_and_recover_after_requests_finish(monkeypatch):
    from ciel import web_answers

    content, lookup, messages, claims = _case()
    monkeypatch.setattr(web_answers, "_GOAL_WORKERS", BoundedSemaphore(2))
    entered = [Event(), Event()]
    release, lock = Event(), Lock()
    count = 0
    results = []
    router = _router()

    def judge(*_args, **_kwargs):
        nonlocal count
        with lock:
            index = count
            count += 1
        entered[index].set()
        assert release.wait(2), "The test must release its blocked provider requests"
        return LLMResponse('{"relevant_claims":[1]}', "model", "provider")

    router.send.side_effect = judge

    def check():
        results.append(json.loads(relevant_web_claims(
            content, lookup, messages, router, timeout_seconds=2,
        )))

    requests = [Thread(target=check, daemon=True) for _ in range(2)]
    try:
        for request in requests:
            request.start()
        assert all(started.wait(1) for started in entered)
        assert json.loads(relevant_web_claims(content, lookup, messages, router)) == {
            "claims": [],
        }
        assert router.send.call_count == 2
    finally:
        release.set()
        for request in requests:
            request.join(2)
    assert not any(request.is_alive() for request in requests)
    assert results == [{"claims": [claims[0]]}] * 2
    router.send.side_effect = None
    assert json.loads(relevant_web_claims(content, lookup, messages, router)) == {
        "claims": [claims[0]],
    }
    assert router.send.call_count == 3


def test_provider_error_never_exposes_task_or_provider_payload(caplog):
    content, lookup, messages, _claims = _case()
    router = _router()
    marker = "PRIVATE_TASK_AND_PROVIDER_RESPONSE"
    router.send.side_effect = RuntimeError(marker)
    assert json.loads(relevant_web_claims(content, lookup, messages, router)) == {"claims": []}
    assert marker not in caplog.text


def test_verification_deadline_reports_timeout_without_approving_claims(monkeypatch):
    from ciel import web_answers

    content, lookup, messages, _claims = _case()
    now, failures = [100.0], []
    monkeypatch.setattr(web_answers.time, "monotonic", lambda: now[0])

    def timed_out(_call, deadline, _canceled, *, workers=None):
        now[0] = deadline
        raise TimeoutError("PRIVATE_PROVIDER_PAYLOAD")

    monkeypatch.setattr(web_answers, "_bounded", timed_out)
    result = relevant_web_claims(content, lookup, messages, _router(), on_failure=failures.append)
    assert json.loads(result) == {"claims": []}
    assert failures == ["timeout"]


def test_verification_provider_failure_reports_only_fixed_status():
    content, lookup, messages, _claims = _case()
    router, failures = _router(), []
    router.send.side_effect = RuntimeError("PRIVATE_PROVIDER_PAYLOAD")
    result = relevant_web_claims(content, lookup, messages, router, on_failure=failures.append)
    assert json.loads(result) == {"claims": []}
    assert failures == ["provider_error"]


@pytest.mark.parametrize("cancel", [False, True])
def test_rejected_or_canceled_claims_do_not_report_provider_failure(cancel):
    content, lookup, messages, _claims = _case()
    canceled, failures = Event(), []
    if cancel:
        canceled.set()
    result = relevant_web_claims(
        content, lookup, messages, _router('{"relevant_claims":[]}'),
        cancel_event=canceled, on_failure=failures.append,
    )
    assert json.loads(result) == {"claims": []}
    assert not failures


def test_failure_callback_cannot_break_fail_closed_evidence_handling():
    content, lookup, messages, _claims = _case()
    router = _router()
    router.send.side_effect = RuntimeError("PRIVATE_PROVIDER_PAYLOAD")

    def failed_callback(_reason):
        raise RuntimeError("CALLBACK_ERROR")

    result = relevant_web_claims(content, lookup, messages, router, on_failure=failed_callback)
    assert json.loads(result) == {"claims": []}


@pytest.mark.parametrize("medium", ["lightnovel", "light-novel", "LN", "manga"])
@pytest.mark.parametrize("reversed_order", [False, True])
def test_reading_point_spellings_still_require_an_explicit_volume_or_chapter(
    medium, reversed_order,
):
    quote = "Amber Wings anime episode 12 aired on Saturday."
    query = (f"Amber Wings {medium} after anime episode 12" if reversed_order
             else f"Amber Wings anime episode 12 to {medium}")
    lookup = InternetResult(query=query, sources=[
        InternetSource("Reference", "https://example.org", content=quote),
    ])
    answer = json.dumps({"claims": [{"text": quote, "quote": quote, "source": 1}]})
    assert "couldn't verify" in verified_web_answer(answer, lookup)
