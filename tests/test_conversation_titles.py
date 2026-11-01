"""Ciel titles conversations, reads old ones, and hiding only removes them from the list."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from ciel.memory import ConversationManager, MemoryStore
from ciel.memory.conversation_index import ConversationIndex, clean_title, conversation_id
from ciel.memory.titles import RETITLE_EVERY_TURNS, ConversationTitler
from ciel.providers.base import LLMResponse


@pytest.fixture
def store(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "memory.db"))
    yield store
    store.close()


def chat(store, session_id, *pairs):
    manager = ConversationManager(store=store, session_id=session_id)
    for user, reply in pairs:
        manager.add_turn("user", user)
        manager.add_turn("assistant", reply)
    return manager


def test_titles_are_cleaned_short_and_masked():
    assert clean_title('  "Planning the weekend trip!"  ') == "Planning the weekend trip"
    assert len(clean_title("word " * 40)) <= 60
    with pytest.raises(ValueError):
        clean_title("  ''  ")


def loader(turns, local_only=False):
    return lambda: (turns, local_only)


def test_titler_names_the_first_exchange_then_reconsiders_later(store):
    index = ConversationIndex(store)
    router = Mock()
    router.send.return_value = LLMResponse("Weekend trip to the coast.", "nim", "fast")
    titler = ConversationTitler(router, index)
    load = loader([("user", "Help me plan a weekend at the coast"), ("assistant", "Sure!")])

    titler.after_reply("s1", user_turns=1, load=load, background=False)

    assert index.title("s1") == "Weekend trip to the coast"
    assert router.send.call_args.kwargs["purpose"] == "summary"
    titler.after_reply("s1", user_turns=2, load=load, background=False)
    assert router.send.call_count == 1, "Not due again until several more messages"

    router.send.return_value = LLMResponse("KEEP", "nim", "fast")
    titler.after_reply("s1", user_turns=1 + RETITLE_EVERY_TURNS, load=load, background=False)
    assert index.title("s1") == "Weekend trip to the coast"
    assert "current title" in router.send.call_args.args[0][0].content

    router.send.return_value = LLMResponse("Packing list and tide times", "nim", "fast")
    titler.after_reply("s1", user_turns=1 + 2 * RETITLE_EVERY_TURNS, load=load,
                       background=False)
    assert index.title("s1") == "Packing list and tide times"


def test_private_conversations_are_titled_by_the_local_route(store):
    router = Mock()
    router.send.return_value = LLMResponse("Private budget", "ollama", "llama")
    titler = ConversationTitler(router, ConversationIndex(store))

    titler.after_reply("s1", user_turns=1, load=loader([("user", "budget")], True),
                       background=False)

    assert router.send.call_args.args[0][1].local_only is True


def test_a_failed_title_request_backs_off_instead_of_retrying_every_reply(store):
    from ciel.memory.titles import RETRY_AFTER_TURNS

    router = Mock()
    router.send.side_effect = RuntimeError("provider down")
    titler = ConversationTitler(router, ConversationIndex(store))
    load = loader([("user", "hi")])

    titler.after_reply("s1", user_turns=1, load=load, background=False)
    titler.after_reply("s1", user_turns=2, load=load, background=False)
    assert router.send.call_count == 1 and ConversationIndex(store).title("s1") is None

    router.send.side_effect, router.send.return_value = None, LLMResponse("Hello", "nim", "x")
    titler.after_reply("s1", user_turns=1 + RETRY_AFTER_TURNS, load=load, background=False)
    assert ConversationIndex(store).title("s1") == "Hello"


def test_titler_never_overwrites_a_rename_made_while_it_was_thinking(store):
    index = ConversationIndex(store)
    router = Mock()

    def slow_reply(*_args, **_kwargs):
        index.set_title("s1", "Owner's own title")
        return LLMResponse("Generated title", "nim", "x")

    router.send.side_effect = slow_reply
    ConversationTitler(router, index).after_reply("s1", user_turns=1, load=loader([]),
                                                  background=False)

    assert index.title("s1") == "Owner's own title"


def test_concurrent_writers_never_lose_titles_or_hidden_flags(store):
    index = ConversationIndex(store)

    def write(number):
        index.set_title(f"s{number}", f"Title {number}")
        index.set_hidden(f"s{number}", number % 2 == 0)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(index.title(f"s{n}") == f"Title {n}" for n in range(20))
    assert index.hidden() == {f"s{n}" for n in range(0, 20, 2)}


def test_saving_a_reply_asks_the_titler_in_the_background(store):
    manager = ConversationManager(store=store, session_id="s1")
    router = Mock(send=Mock(return_value=LLMResponse("Coast trip", "nim", "fast")))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(manager.lifecycle, "start", lambda *_a, **_k: True)
        manager.start_memory_lifecycle(router)
    done = threading.Event()
    original = manager._titler._run
    manager._titler._run = lambda *args: (original(*args), done.set())

    manager.add_turn("user", "Plan a coast trip")
    manager.add_turn("assistant", "Here is a plan.")

    assert done.wait(2)
    assert ConversationIndex(store).title("s1") == "Coast trip"
    manager.close()


def test_hidden_conversations_leave_the_list_but_keep_history_and_stay_readable(store):
    from ciel.webapp.conversations import ConversationCatalog

    chat(store, "s1", ("Plan the coast trip", "Sure."))
    chat(store, "s2", ("Fix my bike", "Check the chain."))
    catalog = ConversationCatalog(store, lambda: ("s1", "s2"))
    handles = {item["title"]: item["handle"] for item in catalog.list()["conversations"]}

    catalog.hide(handles["Fix my bike"])

    titles = [item["title"] for item in catalog.list()["conversations"]]
    assert titles == ["Plan the coast trip"]
    assert len(store.get_recent_turns(session_id="s2", limit=10)) == 2
    index = ConversationIndex(store)
    listed = {item["title"]: item for item in index.summaries(("s1", "s2"), include_private=True)}
    assert listed["Fix my bike"]["hidden_from_list"] is True
    page = index.read(("s1", "s2"), listed["Fix my bike"]["id"], include_private=True)
    assert [m["text"] for m in page["messages"]] == ["Fix my bike", "Check the chain."]


def test_sidebar_and_header_use_ciels_title(store):
    from ciel.webapp.conversations import ConversationCatalog

    chat(store, "s1", ("Help me plan a coast weekend", "Sure."))
    ConversationIndex(store).set_title("s1", "Coast weekend")
    catalog = ConversationCatalog(store, lambda: ("s1",))

    entry = catalog.list()["conversations"][0]

    assert entry["title"] == "Coast weekend"
    assert catalog.title(entry["handle"]) == "Coast weekend"


def test_unknown_conversation_ids_are_refused(store):
    chat(store, "s1", ("Hello", "Hi"))

    with pytest.raises(ValueError):
        ConversationIndex(store).read(("s1",), conversation_id("someone-else"),
                                      include_private=True)


@pytest.fixture
def tools(tmp_path):
    from ciel.agent_tools import MainTools
    from ciel.config import Settings
    from ciel.dialogue import DialogueContext
    from ciel.memory.service import MemoryService
    from ciel.providers.base import ChatMessage

    settings = Settings(_env_file=None, memory_db_path=str(tmp_path / "m.db"),
                        memory_workspace_path=str(tmp_path / "notes"),
                        memory_semantic_enabled=False, memory_consolidation_enabled=False)
    store = MemoryStore(db_path=settings.memory.db_path,
                        workspace_path=settings.memory.workspace_path)
    conversation = chat(store, "test-main", ("What should I cook tonight?", "Try a risotto."))
    context = DialogueContext(settings, store, conversation, MemoryService(store))
    made = MainTools(settings, Mock(), [ChatMessage("user", "What did we talk about?")],
                     "What did we talk about?", context=context, internet=Mock(),
                     current=lambda: True)
    yield made, store
    context.tools.close()
    conversation.close()
    store.close()


def run(tools, name, **arguments):
    from ciel.providers.base import ToolCall

    return tools.execute(ToolCall("call_1", name, json.dumps(arguments)))


def test_ciel_can_list_and_read_its_conversations(tools):
    made, _store = tools

    listed = run(made, "conversation_list").content["conversations"]
    assert listed[0]["current"] is True and listed[0]["messages"] == 2
    page = run(made, "conversation_read", id=listed[0]["id"]).content
    assert [m["role"] for m in page["messages"]] == ["user", "assistant"]
    assert "risotto" in page["messages"][1]["text"]
    assert run(made, "conversation_read", id="cdeadbeef00").is_error


def test_ciel_can_rename_the_current_conversation(tools):
    made, store = tools

    result = run(made, "set_conversation_title", title="  Dinner ideas.  ")

    assert result.content == {"title": "Dinner ideas"}
    assert ConversationIndex(store).title("test-main") == "Dinner ideas"


def test_web_hides_a_conversation_and_reports_the_current_title(store):
    import httpx

    from ciel.web import DesktopWebUI
    from ciel.webapp.conversations import ConversationCatalog

    chat(store, "s1", ("Coast trip", "Sure."))
    chat(store, "s2", ("Bike repair", "Chain."))
    catalog = ConversationCatalog(store, lambda: ("s1", "s2"))
    ConversationIndex(store).set_title("s1", "Coast weekend")
    from ciel.webapp.selection import ActiveSelection

    selection = SimpleNamespace(current=lambda: ActiveSelection(catalog.handle("s1"), 1))
    ui = DesktopWebUI("owner", "hey ciel")
    ui.configure_history(catalog, selection)
    assert ui.snapshot()["title"] == "Coast weekend"
    handle = next(item["handle"] for item in catalog.list()["conversations"]
                  if item["title"] == "Bike repair")
    url = ui.start(MagicMock(return_value=True), port=0)
    try:
        with httpx.Client(base_url=url, trust_env=False, timeout=2) as client:
            assert client.post("/api/conversations/hide", json={"handle": handle}).json() == {
                "hidden": True}
            assert client.post("/api/conversations/hide", json={"handle": "x" * 32}
                               ).status_code == 404
            listed = client.get("/api/conversations").json()["conversations"]
    finally:
        ui.close()
    assert [item["title"] for item in listed] == ["Coast weekend"]
    assert len(store.get_recent_turns(session_id="s2", limit=10)) == 2


def test_header_title_follows_a_conversation_switch_immediately(store):
    from ciel.web import DesktopWebUI
    from ciel.webapp.conversations import ConversationCatalog
    from ciel.webapp.selection import ActiveSelection

    chat(store, "s1", ("Coast trip", "Sure."))
    chat(store, "s2", ("Bike repair", "Chain."))
    index = ConversationIndex(store)
    index.set_title("s1", "Coast weekend")
    index.set_title("s2", "Bike chain fix")
    catalog = ConversationCatalog(store, lambda: ("s1", "s2"))
    active = ["s1"]
    ui = DesktopWebUI("owner", "hey ciel")
    ui.configure_history(catalog, SimpleNamespace(
        current=lambda: ActiveSelection(catalog.handle(active[0]), 1)))

    assert ui.snapshot()["title"] == "Coast weekend"
    active[0] = "s2"
    assert ui.snapshot()["title"] == "Bike chain fix"
