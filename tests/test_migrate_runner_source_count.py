import pytest

from memanto.cli.migrate.runner import source_count

ALL_PROVIDERS = [
    "mem0", "letta", "supermemory", "okf", "chatgpt", "claude",
    "gemini", "zep", "hindsight", "langgraph", "notion", "obsidian", "chroma",
]


@pytest.mark.parametrize("provider", ALL_PROVIDERS)
def test_empty_dict_returns_zero(provider):
    assert source_count(provider, {}) == 0


def test_letta_counts_passages():
    export = {"passages": ["a", "b", "c"]}
    assert source_count("letta", export) == 3


def test_langgraph_counts_items():
    export = {"items": [1, 2]}
    assert source_count("langgraph", export) == 2


def test_chatgpt_counts_user_nodes():
    export = {
        "memories": [
            {
                "mapping": {
                    "n1": {"message": {"author": {"role": "user"}, "content": "hi"}},
                    "n2": {"message": {"author": {"role": "assistant"}, "content": "hey"}},
                    "n3": {"message": {"author": {"role": "user"}, "content": "ok"}},
                }
            },
            {
                "mapping": {
                    "n4": {"message": {"author": {"role": "user"}, "content": "more"}},
                }
            },
        ]
    }
    assert source_count("chatgpt", export) == 3


def test_claude_counts_human_messages():
    export = {
        "memories": [
            {
                "chat_messages": [
                    {"sender": "human", "text": "hello"},
                    {"sender": "assistant", "text": "hi"},
                    {"sender": "human", "text": "bye"},
                ]
            },
            {
                "chat_messages": [
                    {"sender": "human", "text": "again"},
                ]
            },
        ]
    }
    assert source_count("claude", export) == 3


def test_gemini_counts_user_messages():
    export = {
        "memories": [
            {
                "messages": [
                    {"role": "user", "text": "q1"},
                    {"role": "model", "text": "a1"},
                    {"role": "user", "text": "q2"},
                ]
            }
        ]
    }
    assert source_count("gemini", export) == 2


@pytest.mark.parametrize("provider", ["zep", "hindsight", "notion", "obsidian", "chroma", "mem0", "okf"])
def test_generic_providers_count_memories(provider):
    export = {"memories": ["x", "y", "z"]}
    assert source_count(provider, export) == 3


def test_supermemory_falls_back_to_documents_when_no_memories():
    export = {
        "documents": [
            {"chunks": ["a", "b"]},
            {"chunks": ["c"]},
        ]
    }
    assert source_count("supermemory", export) == 3


def test_supermemory_uses_memories_when_present():
    export = {
        "memories": ["m1", "m2"],
        "documents": [{"chunks": ["a", "b", "c"]}],
    }
    assert source_count("supermemory", export) == 2


@pytest.mark.parametrize("provider", ALL_PROVIDERS)
def test_missing_array_key_does_not_raise(provider):
    assert source_count(provider, {"unrelated": "data"}) == 0
