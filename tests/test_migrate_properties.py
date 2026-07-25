from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st

from memanto.app.constants import VALID_MEMORY_TYPES
from memanto.cli.migrate.mappers import (
    MAPPERS,
    _coerce_type,
    map_chatgpt,
    map_claude,
    map_gemini,
    map_zep,
    map_hindsight,
    map_langgraph,
    map_notion,
    map_obsidian,
    map_chroma,
)
from memanto.cli.migrate.runner import source_count

_text = st.text(min_size=1, max_size=200, alphabet=st.characters(blacklist_categories=("Cs",)))
_opt_text = st.one_of(st.none(), _text)
_tags = st.lists(st.text(min_size=1, max_size=30, alphabet=st.characters(blacklist_categories=("Cs",))), max_size=5)
_float01 = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


def _zep_export(memories):
    return {"memories": memories}


def _hindsight_export(memories):
    return {"memories": memories}


def _langgraph_export(items):
    return {"items": items}


def _markdown_export(memories, source):
    return {"memories": memories}


def _chroma_export(memories):
    return {"memories": memories}


def _chatgpt_conv(content):
    return {
        "id": "conv-1",
        "title": "Test",
        "current_node": "n1",
        "mapping": {
            "n1": {
                "id": "n1",
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": [content]},
                    "create_time": 1700000000.0,
                },
                "parent": None,
                "children": [],
            }
        },
    }


def _claude_conv(text):
    return {
        "uuid": "conv-1",
        "name": "Test",
        "chat_messages": [{"sender": "human", "text": text, "uuid": "msg-1", "created_at": "2024-01-01T00:00:00Z"}],
    }


def _gemini_conv(text):
    return {"id": "conv-1", "messages": [{"role": "user", "text": text}]}


@settings(max_examples=50)
@given(
    st.sampled_from(["chatgpt", "claude", "gemini", "zep", "hindsight", "langgraph", "notion", "obsidian", "chroma"]),
    _text,
)
def test_payload_contract(provider, content):
    if provider == "chatgpt":
        export = {"memories": [_chatgpt_conv(content)]}
    elif provider == "claude":
        export = {"memories": [_claude_conv(content)]}
    elif provider == "gemini":
        export = {"memories": [_gemini_conv(content)]}
    elif provider == "zep":
        export = {"memories": [{"fact": content, "uuid": "u1", "valid_at": "2024-01-01T00:00:00Z"}]}
    elif provider == "hindsight":
        export = {"memories": [{"text": content, "fact_type": "observation", "id": "h1"}]}
    elif provider == "langgraph":
        export = {"items": [{"namespace": ["user"], "key": "k1", "value": {"content": content}}]}
    elif provider in ("notion", "obsidian"):
        export = {"memories": [{"body": content, "title": "Title", "filename_stem": "stem", "tags": []}]}
    else:
        export = {"memories": [{"document": content, "id": "doc-1", "metadata": {}}]}

    rows = MAPPERS[provider](export)
    for row in rows:
        assert isinstance(row["content"], str) and row["content"]
        assert len(row["content"]) <= 10000
        assert row["type"] is None or row["type"] in VALID_MEMORY_TYPES
        assert row["provenance"] == "imported"
        assert isinstance(row["tags"], list)
        assert all(isinstance(t, str) for t in row["tags"])
        assert 0.0 <= row["confidence"] <= 1.0


@settings(max_examples=50)
@given(_text)
def test_mapper_idempotence(content):
    export = {"memories": [{"fact": content, "uuid": "u1"}]}
    r1 = map_zep(export)
    r2 = map_zep(export)
    assert len(r1) == len(r2)
    assert [r["content"] for r in r1] == [r["content"] for r in r2]


@settings(max_examples=50)
@given(_text)
def test_nonempty_export_produces_output_zep(content):
    assume(content.strip())
    rows = map_zep({"memories": [{"fact": content}]})
    assert len(rows) >= 1


@settings(max_examples=50)
@given(_text)
def test_nonempty_export_produces_output_langgraph(content):
    rows = map_langgraph({"items": [{"namespace": [], "key": "k", "value": {"content": content}}]})
    assert len(rows) >= 1


@settings(max_examples=50)
@given(_text)
def test_chatgpt_only_user_messages(user_text):
    assume(user_text.strip())
    export = {
        "memories": [
            {
                "id": "conv-1",
                "current_node": "n2",
                "mapping": {
                    "n1": {
                        "id": "n1",
                        "message": {
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": [user_text]},
                            "create_time": 1.0,
                        },
                        "parent": None,
                        "children": ["n2"],
                    },
                    "n2": {
                        "id": "n2",
                        "message": {
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["assistant reply"]},
                            "create_time": 2.0,
                        },
                        "parent": "n1",
                        "children": [],
                    },
                },
            }
        ]
    }
    rows = map_chatgpt(export)
    for row in rows:
        assert user_text.strip() in row["content"]


@settings(max_examples=50)
@given(_text)
def test_claude_only_human_messages(human_text):
    assume(human_text.strip())
    export = {
        "memories": [
            {
                "uuid": "c1",
                "name": "conv",
                "chat_messages": [
                    {"sender": "human", "text": human_text, "uuid": "m1", "created_at": "2024-01-01T00:00:00Z"},
                    {"sender": "assistant", "text": "reply", "uuid": "m2", "created_at": "2024-01-01T00:00:01Z"},
                ],
            }
        ]
    }
    rows = map_claude(export)
    assert len(rows) == 1
    assert human_text.strip() in rows[0]["content"]


@settings(max_examples=50)
@given(_text)
def test_gemini_only_user_messages(user_text):
    assume(user_text.strip())
    export = {
        "memories": [
            {
                "id": "g1",
                "messages": [
                    {"role": "user", "text": user_text},
                    {"role": "model", "text": "model reply"},
                ],
            }
        ]
    }
    rows = map_gemini(export)
    assert len(rows) == 1
    assert user_text.strip() in rows[0]["content"]


@settings(max_examples=50)
@given(_text, st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_categories=("Cs",))))
def test_markdown_title_fallback(body, stem):
    assume(body.strip())
    assume(stem.strip())
    for mapper in (map_notion, map_obsidian):
        export = {"memories": [{"body": body, "title": "", "filename_stem": stem, "tags": []}]}
        rows = mapper(export)
        assert len(rows) >= 1
        assert rows[0]["title"] == stem.strip() or body.strip()[:80] in rows[0]["title"]


@settings(max_examples=50)
@given(st.lists(st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_categories=("Cs",))), min_size=1, max_size=5), _text)
def test_obsidian_tag_extraction(tags, body):
    assume(body.strip())
    export = {"memories": [{"body": body, "title": "T", "filename_stem": "stem", "tags": tags}]}
    rows = map_obsidian(export)
    assert len(rows) >= 1
    for tag in tags:
        assert tag in rows[0]["tags"]


@settings(max_examples=50)
@given(_text, _text)
def test_chroma_metadata_source_in_footer(doc_content, meta_source):
    assume(doc_content.strip())
    assume(meta_source.strip())
    export = {"memories": [{"document": doc_content, "id": "d1", "metadata": {"source": meta_source}}]}
    rows = map_chroma(export)
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "chroma"
    assert "[Supporting data]" in row["content"]
    assert meta_source[:50] in row["content"]


@settings(max_examples=50)
@given(st.sampled_from(sorted(MAPPERS.keys())))
def test_source_count_missing_key_returns_zero(provider):
    assert source_count(provider, {}) == 0


@settings(max_examples=100)
@given(st.one_of(st.none(), st.text()))
def test_coerce_type_never_invalid(s):
    result = _coerce_type(s)
    assert result is None or result in VALID_MEMORY_TYPES
