import pytest

from examples.migrations.mappers import map_langgraph, map_notion, map_obsidian


def _lg_item(key, value, namespace=None):
    """
    Build a LangGraph export item from a key, value, and optional namespace.
    
    Parameters:
    	key: The item's key.
    	value: The item's value.
    	namespace: An optional namespace associated with the item.
    
    Returns:
    	dict: A dictionary containing the key and value, with the namespace included when provided.
    """
    item = {"key": key, "value": value}
    if namespace is not None:
        item["namespace"] = namespace
    return item


def _md_entry(body="", title="", stem="", tags=None, created_at=None):
    """
    Build a Markdown-style export entry with optional metadata.
    
    Parameters:
    	body (str): Entry content.
    	title (str): Optional entry title.
    	stem (str): Filename stem used to identify the entry.
    	tags (list): Optional entry tags.
    	created_at: Optional creation timestamp.
    
    Returns:
    	dict: An entry containing the body, filename stem, and tags, with title and creation timestamp when provided.
    """
    e = {"body": body, "filename_stem": stem, "tags": tags or []}
    if title:
        e["title"] = title
    if created_at:
        e["created_at"] = created_at
    return e


class TestMapLanggraph:
    def test_value_dict_extracts_content(self):
        export = {"items": [_lg_item("k1", {"content": "hello world"}, ["user", "abc"])]}
        rows = map_langgraph(export)
        assert len(rows) == 1
        assert rows[0]["content"].startswith("hello world")

    def test_value_str_used_directly(self):
        export = {"items": [_lg_item("k2", "plain string value", ["ns"])]}
        rows = map_langgraph(export)
        assert rows[0]["content"].startswith("plain string value")

    def test_value_other_type_str_converted(self):
        export = {"items": [_lg_item("k3", 42, [])]}
        rows = map_langgraph(export)
        assert rows[0]["content"].startswith("42")

    def test_namespace_list_becomes_tag(self):
        export = {"items": [_lg_item("k4", "content", ["user", "thread", "123"])]}
        rows = map_langgraph(export)
        assert "user/thread/123" in rows[0]["tags"]

    def test_key_becomes_source_ref(self):
        export = {"items": [_lg_item("my-key", "some content", [])]}
        rows = map_langgraph(export)
        assert rows[0]["source_ref"] == "my-key"

    def test_source_and_provenance(self):
        export = {"items": [_lg_item("k5", "content", [])]}
        rows = map_langgraph(export)
        assert rows[0]["source"] == "langgraph"
        assert rows[0]["provenance"] == "imported"

    def test_type_is_none(self):
        export = {"items": [_lg_item("k6", "content", [])]}
        rows = map_langgraph(export)
        assert rows[0]["type"] is None

    def test_empty_export(self):
        assert map_langgraph({}) == []
        assert map_langgraph({"items": []}) == []

    def test_empty_content_skipped(self):
        export = {"items": [_lg_item("k7", {"content": "   "}, [])]}
        rows = map_langgraph(export)
        assert rows == []

    def test_empty_namespace_no_tag(self):
        export = {"items": [_lg_item("k8", "content", [])]}
        rows = map_langgraph(export)
        assert rows[0]["tags"] == []

    def test_single_record(self):
        export = {"items": [_lg_item("solo", "solo content", ["a"])]}
        rows = map_langgraph(export)
        assert len(rows) == 1

    def test_missing_namespace_key(self):
        export = {"items": [{"key": "k9", "value": "content"}]}
        rows = map_langgraph(export)
        assert len(rows) == 1
        assert rows[0]["tags"] == []

    @pytest.mark.parametrize("namespace,expected_tag", [
        (["a", "b"], "a/b"),
        (["single"], "single"),
        ("string-ns", "string-ns"),
    ])
    def test_namespace_formats(self, namespace, expected_tag):
        export = {"items": [_lg_item("k", "content", namespace)]}
        rows = map_langgraph(export)
        assert rows[0]["tags"] == [expected_tag]


class TestMapNotion:
    def test_basic_entry(self):
        export = {"memories": [_md_entry(body="My note body", stem="my-note")]}
        rows = map_notion(export)
        assert len(rows) == 1
        assert rows[0]["content"].startswith("My note body")

    def test_source_and_provenance(self):
        export = {"memories": [_md_entry(body="body", stem="stem")]}
        rows = map_notion(export)
        assert rows[0]["source"] == "notion"
        assert rows[0]["provenance"] == "imported"
        assert rows[0]["type"] == "artifact"

    def test_title_from_frontmatter(self):
        export = {"memories": [_md_entry(body="body", title="Explicit Title", stem="stem")]}
        rows = map_notion(export)
        assert rows[0]["title"] == "Explicit Title"

    def test_title_fallback_to_stem(self):
        export = {"memories": [_md_entry(body="body", stem="my-page")]}
        rows = map_notion(export)
        assert rows[0]["title"] == "my-page"

    def test_title_fallback_to_body_excerpt(self):
        body = "This is a long note without explicit title"
        export = {"memories": [_md_entry(body=body, stem="")]}
        rows = map_notion(export)
        assert rows[0]["title"] in body or rows[0]["title"].endswith("...")

    def test_empty_body_with_title_uses_title_as_content(self):
        export = {"memories": [_md_entry(body="", title="Page Title", stem="page-title")]}
        rows = map_notion(export)
        assert len(rows) == 1
        assert rows[0]["content"].startswith("Page Title")

    def test_empty_body_and_no_title_skipped(self):
        export = {"memories": [_md_entry(body="", title="", stem="page")]}
        rows = map_notion(export)
        assert rows == []

    def test_whitespace_only_body_skipped(self):
        export = {"memories": [_md_entry(body="   \n\t  ", title="", stem="page")]}
        rows = map_notion(export)
        assert rows == []

    def test_tags_from_frontmatter(self):
        export = {"memories": [_md_entry(body="body", tags=["python", "notes"], stem="s")]}
        rows = map_notion(export)
        assert "python" in rows[0]["tags"]
        assert "notes" in rows[0]["tags"]

    def test_source_ref_is_stem(self):
        export = {"memories": [_md_entry(body="body", stem="my-note")]}
        rows = map_notion(export)
        assert rows[0]["source_ref"] == "my-note"

    def test_empty_export(self):
        assert map_notion({}) == []
        assert map_notion({"memories": []}) == []

    def test_single_record(self):
        export = {"memories": [_md_entry(body="only entry", stem="only")]}
        rows = map_notion(export)
        assert len(rows) == 1

    def test_missing_optional_fields(self):
        export = {"memories": [{"body": "just a body"}]}
        rows = map_notion(export)
        assert len(rows) == 1
        assert rows[0]["tags"] == []

    def test_created_at_parsed(self):
        export = {"memories": [_md_entry(body="body", stem="s", created_at="2024-01-15T10:00:00Z")]}
        rows = map_notion(export)
        assert rows[0]["created_at"] is not None


class TestMapObsidian:
    def test_basic_entry(self):
        export = {"memories": [_md_entry(body="Vault note content", stem="vault-note")]}
        rows = map_obsidian(export)
        assert len(rows) == 1
        assert rows[0]["content"].startswith("Vault note content")

    def test_source_and_provenance(self):
        export = {"memories": [_md_entry(body="body", stem="stem")]}
        rows = map_obsidian(export)
        assert rows[0]["source"] == "obsidian"
        assert rows[0]["provenance"] == "imported"
        assert rows[0]["type"] == "artifact"

    def test_title_from_frontmatter(self):
        export = {"memories": [_md_entry(body="body", title="Note Title", stem="stem")]}
        rows = map_obsidian(export)
        assert rows[0]["title"] == "Note Title"

    def test_title_fallback_to_stem(self):
        export = {"memories": [_md_entry(body="body", stem="my-vault-note")]}
        rows = map_obsidian(export)
        assert rows[0]["title"] == "my-vault-note"

    def test_empty_body_with_title_not_skipped(self):
        export = {"memories": [_md_entry(body="", title="Has Title", stem="s")]}
        rows = map_obsidian(export)
        assert len(rows) == 1

    def test_empty_body_and_no_title_skipped(self):
        export = {"memories": [_md_entry(body="", title="", stem="note")]}
        rows = map_obsidian(export)
        assert rows == []

    def test_whitespace_only_body_skipped(self):
        export = {"memories": [_md_entry(body="  \n  ", title="", stem="note")]}
        rows = map_obsidian(export)
        assert rows == []

    def test_tags_extracted(self):
        export = {"memories": [_md_entry(body="body", tags=["tag1", "tag2"], stem="s")]}
        rows = map_obsidian(export)
        assert rows[0]["tags"] == ["tag1", "tag2"]

    def test_no_tags(self):
        export = {"memories": [_md_entry(body="body", stem="s")]}
        rows = map_obsidian(export)
        assert rows[0]["tags"] == []

    def test_source_ref_is_stem(self):
        export = {"memories": [_md_entry(body="body", stem="vault-file")]}
        rows = map_obsidian(export)
        assert rows[0]["source_ref"] == "vault-file"

    def test_empty_export(self):
        assert map_obsidian({}) == []
        assert map_obsidian({"memories": []}) == []

    def test_single_record(self):
        export = {"memories": [_md_entry(body="only note", stem="only")]}
        rows = map_obsidian(export)
        assert len(rows) == 1

    def test_missing_optional_fields(self):
        export = {"memories": [{"body": "just a body"}]}
        rows = map_obsidian(export)
        assert len(rows) == 1
        assert rows[0]["tags"] == []

    def test_created_at_parsed(self):
        export = {"memories": [_md_entry(body="body", stem="s", created_at="2023-06-01T08:30:00+00:00")]}
        rows = map_obsidian(export)
        assert rows[0]["created_at"] is not None

    @pytest.mark.parametrize("body,title,stem,expected_title", [
        ("body text", "FM Title", "stem", "FM Title"),
        ("body text", "", "my-stem", "my-stem"),
        ("short body", "", "", "short body"),
    ])
    def test_title_resolution(self, body, title, stem, expected_title):
        entry = _md_entry(body=body, title=title, stem=stem)
        rows = map_obsidian({"memories": [entry]})
        assert rows[0]["title"] == expected_title
