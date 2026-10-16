"""Algorithm and safety tests with fake vectors, not evidence of real model recall quality."""

import json

import httpx
import pytest

from ciel.memory.models import MemoryItem, MemoryType
from ciel.memory.semantic import LocalSemanticRecall
from ciel.memory.store import MemoryStore


@pytest.fixture
def store():
    memory = MemoryStore(":memory:")
    yield memory
    memory.close()


def fake_ollama(monkeypatch, vectors, *, transform=None):
    """Install an isolated HTTP transport and record requests plus client privacy options."""
    requests, options = [], []
    original_client = httpx.Client

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        response = {
            "model": "tiny-embed:latest",
            "embeddings": [vectors[text] for text in payload["input"]],
        }
        if transform is not None:
            response = transform(payload, response)
        return httpx.Response(200, json=response)

    def client(**kwargs):
        options.append(kwargs)
        return original_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("ciel.memory.semantic.httpx.Client", client)
    return requests, options


def test_default_backend_performs_no_requests_or_vector_schema_writes(store, monkeypatch):
    def unexpected_request(**_kwargs):
        raise AssertionError("Disabled embeddings must not contact a provider")

    monkeypatch.setattr("ciel.memory.semantic.httpx.Client", unexpected_request)
    backend = LocalSemanticRecall(store)
    item = MemoryItem(content="My laptop has 32 GB RAM")
    store.save_memory(item)
    lexical = [(1.0, item)]
    assert backend.hybrid_rank("notebook capacity", lexical) == lexical
    assert backend.reindex().error == "No local embedding model is configured"
    assert store._get_connection().execute(
        "SELECT 1 FROM sqlite_master WHERE name='memory_vectors'"
    ).fetchone() is None


@pytest.mark.parametrize("url", [
    "https://127.0.0.1:11434", "http://example.com:11434", "http://192.168.1.2:11434",
    "http://user:pass@127.0.0.1:11434", "http://127.0.0.1:11434/other",
    "http://127.0.0.1:11434?remote=1", "http://127.0.0.1:99999",
])
def test_remote_or_ambiguous_embedding_endpoint_is_rejected(store, url):
    with pytest.raises(ValueError, match="loopback"):
        LocalSemanticRecall(store, model="tiny-embed", base_url=url)


def test_explicit_indexing_is_cpu_only_and_does_not_reembed_unchanged_records(store, monkeypatch):
    item = MemoryItem(content="A telescope at the observatory")
    store.save_memory(item)
    store.save_memory(MemoryItem(content="Archived chat", memory_type=MemoryType.CONVERSATION))
    requests, options = fake_ollama(monkeypatch, {item.content: [3.0, 4.0]})
    backend = LocalSemanticRecall(store, model="tiny-embed")
    assert backend.reindex().indexed == 1
    assert backend.reindex().unchanged == 1
    assert len(requests) == 1
    assert requests[0]["options"] == {"num_gpu": 0}
    assert requests[0]["truncate"] is False
    assert options[0]["trust_env"] is False
    assert options[0]["follow_redirects"] is False


@pytest.mark.parametrize("response", [
    {"model": "wrong-model", "embeddings": [[1.0, 0.0]]},
    {"model": "tiny-embed", "embeddings": [[True, 0.0]]},
    {"model": "tiny-embed", "embeddings": [[0.0, 0.0]]},
    {"model": "tiny-embed", "embeddings": [["bad", 1.0]]},
    {"model": "tiny-embed", "embeddings": []},
])
def test_invalid_embedding_response_does_not_persist_vectors(store, monkeypatch, response):
    item = MemoryItem(content="My notebook")
    store.save_memory(item)
    fake_ollama(monkeypatch, {item.content: [1.0, 0.0]}, transform=lambda *_args: response)
    report = LocalSemanticRecall(store, model="tiny-embed").reindex()
    assert report.failed == 1
    assert report.error is not None
    assert store._get_connection().execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0] == 0


def test_query_cache_and_hybrid_preserve_first_lexical_anchor(store, monkeypatch):
    anchor = MemoryItem(content="My keyboard uses tactile switches")
    paraphrase = MemoryItem(content="My laptop has 32 GB RAM")
    for item in (anchor, paraphrase):
        store.save_memory(item)
    requests, _ = fake_ollama(monkeypatch, {
        anchor.content: [1.0, 0.0], paraphrase.content: [0.0, 1.0], "notebook capacity": [0.0, 1.0],
    })
    backend = LocalSemanticRecall(store, model="tiny-embed")
    backend.reindex()
    assert backend.rank("notebook capacity")[0][1].id == paraphrase.id
    assert backend.rank("notebook capacity")[0][1].id == paraphrase.id
    assert len(requests) == 2  # One document batch and one cached query embedding.
    hybrid = backend.hybrid_rank("notebook capacity", [(2.0, anchor)], limit=2)
    assert [item.id for _, item in hybrid] == [anchor.id, paraphrase.id]


def test_corrected_deleted_and_noncanonical_records_cannot_reuse_vectors(store, monkeypatch):
    item = MemoryItem(content="My current laptop is Amber", metadata={"value": "Amber"})
    store.upsert_fact("user:laptop", item)
    requests, _ = fake_ollama(monkeypatch, {
        item.content: [1.0, 0.0], "My current laptop is Indigo": [0.0, 1.0],
        "notebook": [1.0, 0.0],
    })
    backend = LocalSemanticRecall(store, model="tiny-embed")
    backend.reindex()
    store.upsert_fact("user:laptop", MemoryItem(
        content="My current laptop is Indigo", metadata={"value": "Indigo"},
    ))
    assert backend.rank("notebook") == []
    assert len(requests) == 1
    assert backend.reindex().indexed == 1
    store.forget_memories([store.get_fact("user:laptop")])
    assert backend.rank("notebook") == []
    assert store._get_connection().execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0] == 0


def test_provider_failure_keeps_lexical_without_repeated_requests(store, monkeypatch):
    item = MemoryItem(content="My notebook has RAM")
    store.save_memory(item)
    requests, _ = fake_ollama(monkeypatch, {
        item.content: [1.0, 0.0], "notebook": [1.0, 0.0],
    })
    backend = LocalSemanticRecall(store, model="tiny-embed")
    backend.reindex()

    def failure(**_kwargs):
        raise httpx.ConnectError("Ollama is unavailable")

    monkeypatch.setattr("ciel.memory.semantic.httpx.Client", failure)
    lexical = [(2.0, item)]
    assert backend.hybrid_rank("notebook", lexical) == lexical
    assert backend.hybrid_rank("notebook", lexical) == lexical
    assert len(requests) == 1


def test_forgetting_during_document_or_query_embedding_does_not_restore_fact(store, monkeypatch):
    item = MemoryItem(content="My favorite planet is Saturn", metadata={"value": "Saturn"})
    store.upsert_fact("user:planet", item)
    mode = ["documents"]

    def forget_while_embedding(payload, response):
        if (mode[0] == "documents" or payload["input"] == ["planet"]):
            store.forget_memories([store.get_fact("user:planet")])
        return response

    fake_ollama(monkeypatch, {
        item.content: [1.0, 0.0], "planet": [1.0, 0.0],
    }, transform=forget_while_embedding)
    backend = LocalSemanticRecall(store, model="tiny-embed")
    assert backend.reindex().indexed == 0
    assert store.get_fact("user:planet") is None
    store.upsert_fact("user:planet", item)
    mode[0] = "query"
    assert backend.reindex().indexed == 1
    assert backend.hybrid_rank("planet", [(2.0, item)]) == []
    assert store.get_fact("user:planet") is None


@pytest.mark.parametrize("value", [[float("nan"), 1], [float("inf"), 1], [1] * 8193])
def test_corrupt_persisted_vector_is_ignored_without_network(store, monkeypatch, value):
    item = MemoryItem(content="A remembered telescope")
    store.save_memory(item)
    requests, _ = fake_ollama(monkeypatch, {item.content: [1.0, 0.0]})
    backend = LocalSemanticRecall(store, model="tiny-embed")
    backend.reindex()
    with store._get_connection() as connection:
        connection.execute("UPDATE memory_vectors SET vector_json=?", (json.dumps(value),))
    assert backend.rank("telescope") == []
    assert len(requests) == 1


def test_paused_recall_never_queries_local_model(store, monkeypatch):
    item = MemoryItem(content="A saved telescope")
    store.save_memory(item)
    requests, _ = fake_ollama(monkeypatch, {item.content: [1.0, 0.0]})
    backend = LocalSemanticRecall(store, model="tiny-embed")
    backend.reindex()
    store.set_setting("recall_enabled", False)
    assert backend.rank("telescope") == []
    assert backend.hybrid_rank("telescope", [(2.0, item)]) == []
    assert len(requests) == 1


def test_unrelated_query_below_similarity_threshold_adds_no_semantic_memory(store, monkeypatch):
    item = MemoryItem(content="The observatory houses a telescope")
    store.save_memory(item)
    fake_ollama(monkeypatch, {
        item.content: [1.0, 0.0], "Explain ancient mosaics": [0.1, 0.995],
    })
    backend = LocalSemanticRecall(store, model="tiny-embed")
    backend.reindex()
    assert backend.rank("Explain ancient mosaics") == []
    assert backend.hybrid_rank("Explain ancient mosaics", []) == []


def test_pause_changed_while_query_is_embedded_suppresses_lexical_and_semantic(store, monkeypatch):
    item = MemoryItem(content="My telescope is on the balcony")
    store.save_memory(item)

    def pause_during_query(payload, response):
        if payload["input"] == ["stargazing gear"]:
            store.set_setting("recall_enabled", False)
        return response

    fake_ollama(monkeypatch, {
        item.content: [1.0, 0.0], "stargazing gear": [1.0, 0.0],
    }, transform=pause_during_query)
    backend = LocalSemanticRecall(store, model="tiny-embed")
    backend.reindex()
    assert backend.hybrid_rank("stargazing gear", [(2.0, item)]) == []


def test_batch_with_inconsistent_dimensions_is_not_partially_indexed(store, monkeypatch):
    store.save_memory(MemoryItem(content="First fact"))
    store.save_memory(MemoryItem(content="Second fact"))
    fake_ollama(monkeypatch, {"First fact": [1.0, 0.0], "Second fact": [0.0, 1.0, 0.0]})
    report = LocalSemanticRecall(store, model="tiny-embed").reindex()
    assert report.indexed == 0
    assert report.failed == 2


def test_changed_model_ignores_vectors_indexed_with_previous_model(store, monkeypatch):
    item = MemoryItem(content="My telescope")
    store.save_memory(item)
    requests, _ = fake_ollama(monkeypatch, {item.content: [1.0, 0.0]})
    LocalSemanticRecall(store, model="tiny-embed").reindex()
    assert LocalSemanticRecall(store, model="other-model").rank("telescope") == []
    assert len(requests) == 1
