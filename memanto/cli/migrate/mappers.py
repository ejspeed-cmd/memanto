"""
Source -> Memanto schema mappers.

Each mapper takes a provider export dict (the same shape produced by the
``cli/analyze/*_export.py`` modules) and yields memory dicts in the format
accepted by ``SdkClient.batch_remember``:

    {
        "title": str,
        "content": str,         # original text + a [Supporting data] footer
        "type": str | None,     # None lets the parsing service auto-classify
        "tags": list[str],
        "confidence": float,
        "source": str,          # provider name ("mem0", "letta", ...)
        "source_ref": str,      # original record id
        "provenance": "imported",
        "created_at": datetime, # original source timestamp (when present)
        "updated_at": datetime, # migration time = now
    }

Mappers extract every useful field from the source. Anything that maps
naturally onto Memanto's schema (id, created_at, tags) goes into the right
slot. Everything else (provider metadata, scope ids, hashes, scores) gets
packed into a bounded ``[Supporting data]`` markdown block appended to the
content, so it stays searchable and visible without bloating the schema.

Adding a new provider: write a ``map_<provider>`` function returning
``list[dict]``, register it in ``MAPPERS``, and add a per-provider source
count helper in ``runner.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from memanto.app.constants import VALID_MEMORY_TYPES

# Mem0 ships category labels per memory. Map the common ones to Memanto's
# typed primitives; everything else falls through to None (auto-classify).
_MEM0_CATEGORY_TO_TYPE: dict[str, str] = {
    "personal_details": "fact",
    "personal_preferences": "preference",
    "preferences": "preference",
    "professional_info": "fact",
    "work": "fact",
    "skills": "fact",
    "goals_and_plans": "goal",
    "tasks": "commitment",
    "relationships": "relationship",
    "events": "event",
    "decisions": "decision",
    "observations": "observation",
}

_DEFAULT_TITLE_CHARS = 80
_MAX_CONTENT_CHARS = 10000  # MemoryRecord.content max_length
_MAX_FOOTER_CHARS = 800  # cap supporting-data footer so it never dominates


def _title_from(content: str) -> str:
    text = content.strip().replace("\n", " ")
    if len(text) <= _DEFAULT_TITLE_CHARS:
        return text
    return text[: _DEFAULT_TITLE_CHARS - 3].rstrip() + "..."


def _coerce_type(raw: str | None) -> str | None:
    if not raw:
        return None
    t = raw.strip().lower()
    return t if t in VALID_MEMORY_TYPES else None


def _scope_tag(scope: dict[str, Any] | None) -> str | None:
    if not scope:
        return None
    for k, v in scope.items():
        if v:
            return f"{k}={v}"
    return None


def _parse_dt(value: Any) -> datetime | None:
    """Best-effort parse of a timestamp from a source record into UTC datetime.

    Handles ISO 8601 strings (with/without ``Z``), Unix epoch ints/floats,
    and already-parsed ``datetime`` objects. Returns ``None`` when nothing
    sensible can be extracted — the caller falls back to the server default.
    """
    if value in (None, "", 0):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Python <3.11 doesn't accept the trailing 'Z' shorthand.
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def _pick_first_dt(record: dict[str, Any], keys: tuple[str, ...]) -> datetime | None:
    for key in keys:
        dt = _parse_dt(record.get(key))
        if dt is not None:
            return dt
    return None


def _format_supporting_data(items: list[tuple[str, Any]]) -> str:
    """Render the ``[Supporting data]`` footer.

    Filters out empties, truncates over-long values, and caps the total
    footer length so it never overruns ``MemoryRecord.content``.
    """
    lines: list[str] = []
    for label, value in items:
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value if v not in (None, ""))
            if not value:
                continue
        elif isinstance(value, dict):
            # one-line compact dict so the footer doesn't sprawl
            value = "; ".join(
                f"{k}={v}" for k, v in value.items() if v not in (None, "")
            )
            if not value:
                continue
        text = str(value)
        if len(text) > 200:
            text = text[:197] + "..."
        lines.append(f"- {label}: {text}")

    if not lines:
        return ""

    body = "\n".join(lines)
    if len(body) > _MAX_FOOTER_CHARS:
        body = body[: _MAX_FOOTER_CHARS - 4] + "\n..."
    return "\n\n---\n[Supporting data]\n" + body


def _attach_footer(content: str, footer: str) -> str:
    """Append the supporting-data footer, trimming content if it overflows."""
    if not footer:
        return content
    budget = _MAX_CONTENT_CHARS - len(footer)
    if budget < 0:
        # Pathological — footer somehow exceeds content limit on its own.
        return content[:_MAX_CONTENT_CHARS]
    trimmed = content if len(content) <= budget else content[: budget - 4] + "\n..."
    return trimmed + footer


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Mem0
# --------------------------------------------------------------------------


def map_mem0(export: dict[str, Any]) -> list[dict[str, Any]]:
    """Map a Mem0 export to rich Memanto memory payloads."""
    rows: list[dict[str, Any]] = []
    migrated_at = _now_utc()

    for mem in export.get("memories", []) or []:
        content = (mem.get("memory") or mem.get("content") or "").strip()
        if not content:
            continue

        categories = [str(c).lower() for c in (mem.get("categories") or []) if c]
        memory_type: str | None = None
        for cat in categories:
            memory_type = _MEM0_CATEGORY_TO_TYPE.get(cat) or _coerce_type(cat)
            if memory_type:
                break

        tags = list(dict.fromkeys(categories))
        scope = mem.get("export_scope") or {}
        scope_tag = _scope_tag(scope)
        if scope_tag:
            tags.append(scope_tag)

        created_at = _pick_first_dt(mem, ("created_at", "createdAt"))
        expires_at = _pick_first_dt(mem, ("expiration_date", "expires_at"))

        # Anything we couldn't slot directly goes into the footer.
        footer = _format_supporting_data(
            [
                ("Source", f"mem0:{mem.get('id')}" if mem.get("id") else None),
                ("Mem0 scope", scope_tag),
                ("Categories", categories),
                ("Mem0 metadata", mem.get("metadata")),
                ("Mem0 score", mem.get("score")),
                ("Hash", mem.get("hash")),
                ("Immutable", mem.get("immutable")),
                ("Source created_at", created_at.isoformat() if created_at else None),
                ("Expires at", expires_at.isoformat() if expires_at else None),
            ]
        )

        rows.append(
            {
                "title": _title_from(content),
                "content": _attach_footer(content, footer),
                "type": memory_type,
                "tags": tags,
                "confidence": 0.8,
                "source": "mem0",
                "source_ref": str(mem.get("id")) if mem.get("id") else None,
                "provenance": "imported",
                "created_at": created_at,
                "updated_at": migrated_at,
            }
        )
    return rows


# --------------------------------------------------------------------------
# Letta
# --------------------------------------------------------------------------


def map_letta(export: dict[str, Any]) -> list[dict[str, Any]]:
    """Map Letta archival passages to rich Memanto memory payloads."""
    rows: list[dict[str, Any]] = []
    migrated_at = _now_utc()

    for passage in export.get("passages", []) or []:
        content = (passage.get("text") or passage.get("content") or "").strip()
        if not content:
            continue

        tags: list[str] = []
        agent_name = passage.get("export_agent_name")
        agent_id = passage.get("export_agent_id")
        if agent_name:
            tags.append(f"agent={agent_name}")
        elif agent_id:
            tags.append(f"agent_id={agent_id}")

        source_tags = [str(t) for t in (passage.get("tags") or []) if t]
        for t in source_tags:
            if t not in tags:
                tags.append(t)

        created_at = _pick_first_dt(passage, ("created_at", "createdAt"))

        footer = _format_supporting_data(
            [
                ("Source", f"letta:{passage.get('id')}" if passage.get("id") else None),
                ("Letta agent_id", agent_id),
                ("Letta agent_name", agent_name),
                ("Letta tags", source_tags),
                ("Letta metadata", passage.get("metadata")),
                ("Source", passage.get("source")),  # passage may carry its own
                ("Source created_at", created_at.isoformat() if created_at else None),
            ]
        )

        rows.append(
            {
                "title": _title_from(content),
                "content": _attach_footer(content, footer),
                "type": "observation",
                "tags": tags,
                "confidence": 0.8,
                "source": "letta",
                "source_ref": str(passage.get("id")) if passage.get("id") else None,
                "provenance": "imported",
                "created_at": created_at,
                "updated_at": migrated_at,
            }
        )
    return rows


# --------------------------------------------------------------------------
# Supermemory
# --------------------------------------------------------------------------


def map_supermemory(export: dict[str, Any]) -> list[dict[str, Any]]:
    """Map a Supermemory export to rich Memanto memory payloads.

    Primary source is the ``memories[]`` array — Supermemory's AI-extracted
    facts. Falls back to document chunks when no extracted memories exist
    (mostly fresh accounts). Each row keeps its container tag and links
    back to the source via ``source_ref``.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    migrated_at = _now_utc()

    for mem in export.get("memories", []) or []:
        content = (
            mem.get("content") or mem.get("memory") or mem.get("text") or ""
        ).strip()
        if not content:
            continue

        tags: list[str] = []
        tag = mem.get("container_tag")
        if tag:
            tags.append(str(tag))

        created_at = _pick_first_dt(mem, ("createdAt", "created_at"))

        footer = _format_supporting_data(
            [
                (
                    "Source",
                    f"supermemory:{mem.get('id')}" if mem.get("id") else None,
                ),
                ("Container tag", tag),
                ("Document id", mem.get("documentId") or mem.get("document_id")),
                ("Supermemory metadata", mem.get("metadata")),
                ("Score", mem.get("score")),
                ("Source created_at", created_at.isoformat() if created_at else None),
            ]
        )

        rows.append(
            {
                "title": _title_from(content),
                "content": _attach_footer(content, footer),
                "type": None,
                "tags": tags,
                "confidence": 0.8,
                "source": "supermemory",
                "source_ref": str(mem.get("id")) if mem.get("id") else None,
                "provenance": "imported",
                "created_at": created_at,
                "updated_at": migrated_at,
            }
        )
        seen.add(content)

    if rows:
        return rows

    # Fallback: harvest chunk text when extracted memories are empty.
    for doc in export.get("documents", []) or []:
        doc_tags = [str(t) for t in (doc.get("container_tags") or []) if t]
        doc_id = doc.get("id")
        doc_created = _pick_first_dt(
            doc.get("detail") or doc, ("createdAt", "created_at")
        )
        for chunk in doc.get("chunks", []) or []:
            content = (chunk.get("content") or chunk.get("text") or "").strip()
            if not content or content in seen:
                continue
            seen.add(content)
            footer = _format_supporting_data(
                [
                    (
                        "Source",
                        f"supermemory:doc:{doc_id}:chunk:{chunk.get('id')}"
                        if doc_id
                        else None,
                    ),
                    ("Container tags", doc_tags),
                    ("Document id", doc_id),
                    ("Chunk id", chunk.get("id")),
                    (
                        "Source created_at",
                        doc_created.isoformat() if doc_created else None,
                    ),
                ]
            )
            rows.append(
                {
                    "title": _title_from(content),
                    "content": _attach_footer(content, footer),
                    "type": "artifact",
                    "tags": doc_tags,
                    "confidence": 0.7,
                    "source": "supermemory",
                    "source_ref": (f"{doc_id}:{chunk.get('id')}" if doc_id else None),
                    "provenance": "imported",
                    "created_at": doc_created,
                    "updated_at": migrated_at,
                }
            )
    return rows


# --------------------------------------------------------------------------
# OKF (Open Knowledge Format)
# --------------------------------------------------------------------------


def map_okf(export: dict[str, Any]) -> list[dict[str, Any]]:
    """Map OKF bundle entries (from ``okf_loader.load_okf_bundle``) to Memanto
    memory payloads.

    OKF's ``type`` is free-form domain vocabulary, so it can't map onto
    Memanto's fixed types. We use it only when it happens to equal a Memanto
    type (or when a Memanto ``x_memanto.type`` round-trip value is present);
    otherwise we leave ``type=None`` for auto-classification and record the
    original OKF type in the footer. Everything with no schema slot (OKF type,
    resource, links, unknown frontmatter keys) goes into ``[Supporting data]``.
    """
    rows: list[dict[str, Any]] = []
    migrated_at = _now_utc()

    for entry in export.get("memories", []) or []:
        body = (entry.get("body") or "").strip()
        description = (entry.get("description") or "").strip()
        title = (entry.get("title") or "").strip()

        if description and description not in body:
            content = f"{description}\n\n{body}".strip()
        else:
            content = body
        if not content:
            content = title
        if not content:
            continue

        x_memanto = entry.get("x_memanto") or {}
        okf_type = entry.get("type")
        memory_type = _coerce_type(x_memanto.get("type")) or _coerce_type(okf_type)

        tags = [str(t) for t in (entry.get("tags") or []) if t]
        resource = entry.get("resource")

        raw_conf = x_memanto.get("confidence")
        try:
            confidence = float(raw_conf) if raw_conf is not None else 0.8
        except (TypeError, ValueError):
            confidence = 0.8
        confidence = min(1.0, max(0.0, confidence))

        source = x_memanto.get("source") or "okf"
        created_at = _parse_dt(entry.get("timestamp"))

        footer_items: list[tuple[str, Any]] = [
            ("OKF source", entry.get("source_path")),
            # Only surface the OKF type when we couldn't map it to a slot.
            ("OKF type", okf_type if not memory_type else None),
            ("OKF resource", resource),
            ("Links", entry.get("links")),
        ]
        for key, value in (entry.get("extra") or {}).items():
            footer_items.append((f"OKF {key}", value))
        footer = _format_supporting_data(footer_items)

        if footer:
            content = _attach_footer(content, footer)
        elif len(content) > _MAX_CONTENT_CHARS:
            content = content[: _MAX_CONTENT_CHARS - 4] + "\n..."

        rows.append(
            {
                "title": title or _title_from(content),
                "content": content,
                "type": memory_type,
                "tags": tags,
                "confidence": confidence,
                "source": source,
                "source_ref": str(resource) if resource else None,
                "provenance": "imported",
                "created_at": created_at,
                "updated_at": migrated_at,
            }
        )
    return rows


# --------------------------------------------------------------------------
# Claude
# --------------------------------------------------------------------------


def map_claude(export: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    migrated_at = _now_utc()

    for conv in export.get("memories", []) or []:
        if not isinstance(conv, dict):
            continue
        conv_title = (conv.get("name") or "").strip() or None

        for msg in conv.get("chat_messages", []) or []:
            try:
                if msg.get("sender") != "human":
                    continue
                text = msg.get("text")
                content = (text if isinstance(text, str) else "").strip()
                if not content:
                    parts = msg.get("content") or []
                    content = " ".join(
                        p["text"] for p in parts
                        if isinstance(p, dict) and p.get("type") == "text" and p.get("text", "").strip()
                    ).strip()
                if not content:
                    continue

                created_at = _parse_dt(msg.get("created_at"))

                footer = _format_supporting_data(
                    [
                        ("Conversation", conv_title),
                        ("Conversation id", conv.get("uuid")),
                        ("Message id", msg.get("uuid")),
                    ]
                )

                rows.append(
                    {
                        "title": conv_title or _title_from(content),
                        "content": _attach_footer(content, footer),
                        "type": None,
                        "tags": [],
                        "confidence": 0.8,
                        "source": "claude",
                        "source_ref": str(msg.get("uuid")) if msg.get("uuid") else None,
                        "provenance": "imported",
                        "created_at": created_at,
                        "updated_at": migrated_at,
                    }
                )
            except (AttributeError, TypeError):
                continue

    return rows


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------


def map_gemini(export: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    migrated_at = _now_utc()

    for conv in export.get("memories", []) or []:
        try:
            messages = conv.get("messages") or []
            created_at = _parse_dt(conv.get("createdTime"))
        except (AttributeError, TypeError):
            continue

        for msg in messages:
            try:
                if msg.get("role") != "user":
                    continue
                content = (msg.get("text") or "").strip()
                if not content:
                    continue
            except (AttributeError, TypeError):
                continue

            footer = _format_supporting_data(
                [
                    ("Conversation id", conv.get("id")),
                ]
            )

            rows.append(
                {
                    "title": _title_from(content),
                    "content": _attach_footer(content, footer),
                    "type": None,
                    "tags": [],
                    "confidence": 0.8,
                    "source": "gemini",
                    "source_ref": str(conv.get("id")) if conv.get("id") else None,
                    "provenance": "imported",
                    "created_at": created_at,
                    "updated_at": migrated_at,
                }
            )

    return rows


# --------------------------------------------------------------------------
# ChatGPT
# --------------------------------------------------------------------------


def map_chatgpt(export: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    migrated_at = _now_utc()

    for conv in export.get("memories", []) or []:
        if not isinstance(conv, dict):
            continue
        mapping = conv.get("mapping") or {}
        if not isinstance(mapping, dict):
            continue
        current_node = conv.get("current_node")
        if not mapping or not current_node:
            continue

        seen: set[str] = set()
        user_nodes: list[dict[str, Any]] = []
        node_id = current_node

        while node_id and node_id not in seen:
            seen.add(node_id)
            node = mapping.get(node_id)
            if not isinstance(node, dict):
                break
            msg = node.get("message")
            if isinstance(msg, dict):
                author = msg.get("author") or {}
                content_obj = msg.get("content") or {}
                if (
                    author.get("role") == "user"
                    and content_obj.get("content_type", "text") != "user_editable_context"
                ):
                    parts = content_obj.get("parts") or []
                    content = " ".join(p for p in parts if isinstance(p, str) and p.strip())
                    if content:
                        user_nodes.append(node)
            node_id = node.get("parent")

        user_nodes.reverse()

        conv_title = (conv.get("title") or "").strip() or None

        for node in user_nodes:
            msg = node["message"]
            parts = (msg.get("content") or {}).get("parts") or []
            content = " ".join(p for p in parts if isinstance(p, str) and p.strip())

            created_at = _parse_dt(msg.get("create_time")) or _parse_dt(conv.get("create_time"))

            footer = _format_supporting_data(
                [
                    ("Conversation", conv_title),
                    ("Conversation id", conv.get("id")),
                    ("Node id", node.get("id")),
                ]
            )

            rows.append(
                {
                    "title": conv_title or _title_from(content),
                    "content": _attach_footer(content, footer),
                    "type": None,
                    "tags": [],
                    "confidence": 0.8,
                    "source": "chatgpt",
                    "source_ref": str(node.get("id")) if node.get("id") else None,
                    "provenance": "imported",
                    "created_at": created_at,
                    "updated_at": migrated_at,
                }
            )

    return rows


def map_zep(export: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    migrated_at = _now_utc()

    for edge in export.get("memories", []) or []:
        try:
            content = (edge.get("fact") or "").strip()
            if not content:
                continue
            rating = edge.get("score") or edge.get("relevance")
            try:
                confidence = min(1.0, max(0.0, float(rating))) if rating is not None else 0.8
            except (TypeError, ValueError):
                confidence = 0.8
            footer = _format_supporting_data(
                [
                    ("Edge name", edge.get("name")),
                    ("Rating", str(rating) if rating is not None else None),
                    ("UUID", edge.get("uuid")),
                ]
            )
            rows.append(
                {
                    "title": _title_from(content),
                    "content": _attach_footer(content, footer),
                    "type": "fact",
                    "tags": [],
                    "confidence": confidence,
                    "source": "zep",
                    "source_ref": edge.get("uuid"),
                    "provenance": "imported",
                    "created_at": _parse_dt(edge.get("valid_at") or edge.get("created_at")),
                    "updated_at": migrated_at,
                }
            )
        except (AttributeError, TypeError):
            continue

    return rows


_HINDSIGHT_TYPE_MAP: dict[str, str] = {
    "world": "fact",
    "experience": "event",
}


def _hindsight_type(fact_type: str | None) -> str | None:
    if not fact_type:
        return None
    t = fact_type.strip().lower()
    return _HINDSIGHT_TYPE_MAP.get(t) or _coerce_type(t)


def map_hindsight(export: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    migrated_at = _now_utc()

    for item in export.get("memories", []) or []:
        try:
            content = (item.get("text") or item.get("content") or "").strip()
            if not content:
                continue
            metadata = item.get("metadata") or {}
            footer = _format_supporting_data(
                [
                    ("Bank", item.get("bank_id")),
                    ("Fact type", item.get("fact_type")),
                    ("Context", item.get("context") or None),
                    ("Entities", item.get("entities") or None),
                    *((k, str(v)) for k, v in metadata.items() if v),
                ]
            )
            rows.append(
                {
                    "title": _title_from(content),
                    "content": _attach_footer(content, footer),
                    "type": _hindsight_type(item.get("fact_type")),
                    "tags": list(item.get("tags") or []),
                    "confidence": 0.8,
                    "source": "hindsight",
                    "source_ref": item.get("id"),
                    "provenance": "imported",
                    "created_at": _parse_dt(item.get("date") or item.get("mentioned_at")),
                    "updated_at": migrated_at,
                }
            )
        except (AttributeError, TypeError):
            continue

    return rows


def map_langgraph(export: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    migrated_at = _now_utc()

    for item in export.get("items", []) or []:
        try:
            value = item.get("value")
            if isinstance(value, dict):
                raw_content = value.get("content")
                content = (str(raw_content) if raw_content is not None else "").strip()
            elif isinstance(value, str):
                content = value.strip()
            else:
                content = str(value).strip() if value is not None else ""
            if not content:
                continue

            namespace = item.get("namespace") or []
            if isinstance(namespace, (list, tuple)):
                ns_tag = "/".join(str(p) for p in namespace) if namespace else None
            else:
                ns_tag = str(namespace) if namespace else None

            key = item.get("key")
            footer_pairs: list[tuple[str, Any]] = [("Namespace", ns_tag), ("Key", key)]
            if isinstance(value, dict):
                for k, v in value.items():
                    if k != "content" and v:
                        footer_pairs.append((k.capitalize(), str(v)))
            footer = _format_supporting_data(footer_pairs)

            rows.append(
                {
                    "title": _title_from(content),
                    "content": _attach_footer(content, footer),
                    "type": None,
                    "tags": [ns_tag] if ns_tag else [],
                    "confidence": 0.8,
                    "source": "langgraph",
                    "source_ref": key,
                    "provenance": "imported",
                    "created_at": _parse_dt(item.get("created_at")),
                    "updated_at": migrated_at,
                }
            )
        except (AttributeError, TypeError):
            continue

    return rows


def _map_markdown_entry(
    entry: dict[str, Any],
    *,
    source: str,
    memory_type: str,
    migrated_at: Any,
) -> dict[str, Any] | None:
    body = (entry.get("body") or "").strip()
    explicit_title = (entry.get("title") or "").strip()
    stem = (entry.get("filename_stem") or "").strip()

    if not explicit_title and not body:
        return None

    title = explicit_title or stem or _title_from(body)
    content = body if body else explicit_title
    if not content:
        return None

    tags = [str(t) for t in (entry.get("tags") or []) if t]
    footer = _format_supporting_data([("Filename", stem or None)])
    return {
        "title": title,
        "content": _attach_footer(content, footer),
        "type": memory_type,
        "tags": tags,
        "confidence": 0.8,
        "source": source,
        "source_ref": stem or None,
        "provenance": "imported",
        "created_at": _parse_dt(entry.get("created_at")),
        "updated_at": migrated_at,
    }


def map_notion(export: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    migrated_at = _now_utc()

    for entry in export.get("memories", []) or []:
        try:
            row = _map_markdown_entry(entry, source="notion", memory_type="artifact", migrated_at=migrated_at)
            if row:
                rows.append(row)
        except (AttributeError, TypeError):
            continue

    return rows


def map_obsidian(export: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    migrated_at = _now_utc()

    for entry in export.get("memories", []) or []:
        try:
            row = _map_markdown_entry(entry, source="obsidian", memory_type="artifact", migrated_at=migrated_at)
            if row:
                rows.append(row)
        except (AttributeError, TypeError):
            continue

    return rows


def map_chroma(export: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    migrated_at = _now_utc()

    for item in export.get("memories", []) or []:
        try:
            content = (item.get("document") or "").strip()
            if not content:
                continue
            metadata = item.get("metadata") or {}
            meta_source = metadata.get("source")
            footer_pairs: list[tuple[str, Any]] = []
            if meta_source:
                footer_pairs.append(("source", meta_source))
            for k, v in metadata.items():
                if k != "source" and v is not None:
                    footer_pairs.append((k, str(v)))
            footer = _format_supporting_data(footer_pairs)
            rows.append(
                {
                    "title": _title_from(content),
                    "content": _attach_footer(content, footer),
                    "type": None,
                    "tags": [],
                    "confidence": 0.8,
                    "source": "chroma",
                    "source_ref": item.get("id"),
                    "provenance": "imported",
                    "created_at": None,
                    "updated_at": migrated_at,
                }
            )
        except (AttributeError, TypeError):
            continue

    return rows


MAPPERS: dict[str, Callable[[dict[str, Any]], list[dict[str, Any]]]] = {
    "mem0": map_mem0,
    "letta": map_letta,
    "supermemory": map_supermemory,
    "okf": map_okf,
    "chatgpt": map_chatgpt,
    "claude": map_claude,
    "gemini": map_gemini,
    "zep": map_zep,
    "hindsight": map_hindsight,
    "langgraph": map_langgraph,
    "notion": map_notion,
    "obsidian": map_obsidian,
    "chroma": map_chroma,
}


def type_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count mapped rows by resolved (or unclassified) type — for previews."""
    counts: dict[str, int] = {}
    for row in rows:
        key = row.get("type") or "auto"
        counts[key] = counts.get(key, 0) + 1
    return counts
