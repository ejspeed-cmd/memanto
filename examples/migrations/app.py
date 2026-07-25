"""
Memanto Migration UI

Streamlit app for migrating AI conversation exports into Memanto.
Covers the three conversation ZIP providers: ChatGPT, Claude, Gemini.

Run:
    streamlit run examples/migrations/ai-conversations/app.py
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import streamlit as st

st.set_page_config(
    page_title="Memanto Migration",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy imports — memanto must be installed (pip install -e .)
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_memanto():
    try:
        from memanto.cli.migrate.mappers import MAPPERS
        from memanto.cli.migrate.runner import run_migration
        from memanto.cli.client.sdk_client import SdkClient
        return MAPPERS, run_migration, SdkClient
    except ImportError as exc:
        st.error(f"memanto package not found. Run `pip install -e .` from the repo root.\n\n{exc}")
        st.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_gemini_archive(tmp_path: Path) -> dict[str, Any]:
    """Normalize a Google Takeout Gemini ZIP into {memories: [...]}."""
    import html
    import html.parser
    import re

    json_hits = list(tmp_path.rglob("My Activity.json"))
    html_hits = list(tmp_path.rglob("My Activity.html"))

    if json_hits:
        entries = json.loads(json_hits[0].read_text(encoding="utf-8"))
        memories = []
        for entry in entries or []:
            title = entry.get("title") or ""
            if not title.startswith("Prompted "):
                continue
            prompt = title[len("Prompted "):]
            if not prompt.strip():
                continue
            memories.append({"createdTime": entry.get("time"), "messages": [{"role": "user", "text": prompt}]})
        return {"memories": memories}

    if html_hits:
        # basic extraction from HTML activity
        raw = html_hits[0].read_text(encoding="utf-8", errors="replace")
        entries = re.findall(r'Prompted\s+(.*?)(?=Prompted\s|$)', raw, re.DOTALL)
        memories = []
        for e in entries:
            text = re.sub(r'<[^>]+>', '', e).strip()
            if text:
                memories.append({"messages": [{"role": "user", "text": text[:500]}]})
        return {"memories": memories}

    candidates = [f for f in tmp_path.rglob("*.json") if f.name != "My Activity.json"]
    memories = []
    for f in candidates:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "messages" in item:
                        memories.append(item)
            elif isinstance(data, dict) and "messages" in data:
                memories.append(data)
        except Exception:
            continue
    return {"memories": memories}


def _load_export_from_bytes(file_bytes: bytes, source: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                zf.extractall(tmp)
        except zipfile.BadZipFile:
            st.error("Could not read the ZIP file. Make sure you uploaded a valid export archive.")
            st.stop()

        if source in ("chatgpt", "claude"):
            json_file = tmp_path / "conversations.json"
            if not json_file.exists():
                candidates = list(tmp_path.rglob("conversations.json"))
                if not candidates:
                    st.error("conversations.json not found in the ZIP. Make sure you exported the right file.")
                    st.stop()
                json_file = candidates[0]
            raw = json.loads(json_file.read_text(encoding="utf-8"))
            return {"memories": raw} if isinstance(raw, list) else raw

        return _parse_gemini_archive(tmp_path)


def _run_dry_run(source: str, export: dict[str, Any]) -> tuple[list[dict], dict]:
    MAPPERS, run_migration, _ = _load_memanto()
    summary, rows = run_migration(
        provider=source,
        export=export,
        client=None,
        agent_id="",
        dry_run=True,
        on_progress=lambda msg: None,
    )
    return rows, summary.as_dict()


def _do_migrate(source: str, export: dict[str, Any], agent_id: str, api_key: str) -> dict:
    _, run_migration, SdkClient = _load_memanto()
    client = SdkClient(api_key=api_key)
    summary, _ = run_migration(
        provider=source,
        export=export,
        client=client,
        agent_id=agent_id,
        dry_run=False,
        on_progress=lambda msg: None,
    )
    return summary.as_dict()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

PROVIDERS = {
    "chatgpt": "ChatGPT",
    "claude": "Claude",
    "gemini": "Gemini",
}

PROVIDER_LOGOS = {
    "chatgpt": "https://upload.wikimedia.org/wikipedia/commons/e/ef/ChatGPT-Logo.svg",
    "claude": "https://upload.wikimedia.org/wikipedia/commons/b/b0/Claude_AI_symbol.svg",
    "gemini": "https://upload.wikimedia.org/wikipedia/commons/1/1d/Google_Gemini_icon_2025.svg",
}

EXPORT_INSTRUCTIONS = {
    "chatgpt": "**ChatGPT:** Settings → Data controls → Export data → confirm email → download ZIP",
    "claude": "**Claude:** claude.ai → Account settings → Privacy → Export data → download ZIP",
    "gemini": "**Gemini:** [takeout.google.com](https://takeout.google.com) → Deselect all → My Activity → Gemini Apps → JSON → Create export → download ZIP",
}


_MEMANTO_PREFIX = "memanto_agent_"


def _moorcheh_client(api_key: str):
    from moorcheh_sdk import MoorchehClient
    return MoorchehClient(api_key=api_key)


def _fetch_agents(api_key: str) -> list[str]:
    """Return agent_ids (without the memanto_agent_ prefix) from Moorcheh cloud."""
    try:
        client = _moorcheh_client(api_key)
        result = client.namespaces.list()
        namespaces = result.get("namespaces", []) if isinstance(result, dict) else result
        ids = []
        for ns in namespaces:
            name = ns.get("namespace_name", "")
            if name.startswith(_MEMANTO_PREFIX):
                ids.append(name[len(_MEMANTO_PREFIX):])
        return ids
    except Exception:
        return []


def _create_agent(api_key: str, agent_id: str) -> tuple[bool, str]:
    try:
        _, _, SdkClient = _load_memanto()
        client = SdkClient(api_key=api_key)
        client.create_agent(agent_id=agent_id, pattern="tool")
        return True, f"Namespace '{agent_id}' created."
    except Exception as exc:
        if "already exists" in str(exc).lower():
            return True, f"Namespace '{agent_id}' already exists — using it."
        return False, str(exc)


def sidebar():
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/moorcheh-ai/memanto/main/assets/memanto-logo.svg", width=140)
        st.markdown("## Memanto Migration")
        st.markdown("Liberate the memory your AI assistant has built about you.")
        st.divider()
        st.markdown("### Configuration")

        api_key = st.text_input(
            "Moorcheh API Key",
            type="password",
            value=os.environ.get("MOORCHEH_API_KEY", ""),
            help="Get yours at moorcheh.ai",
        )

        agent_id = ""

        if api_key:
            prev_key = st.session_state.get("_loaded_api_key")
            if prev_key != api_key:
                with st.spinner("Loading namespaces..."):
                    st.session_state["agents"] = _fetch_agents(api_key)
                st.session_state["_loaded_api_key"] = api_key
                # a new key means any previously selected namespace no longer applies
                st.session_state.pop("agent_id", None)

            agents: list[str] = st.session_state.get("agents", [])
            CREATE_OPT = "+ Create new namespace"

            if not agents:
                # Brand new key, or a key with no namespaces yet — skip straight
                # to the creation form instead of showing a dropdown with a
                # single "+ Create new namespace" option in it.
                st.caption("No namespaces found for this key yet — create one to get started.")
                choice = CREATE_OPT
            else:
                options = agents + [CREATE_OPT]
                if st.session_state.get("agent_id") in agents:
                    default_idx = options.index(st.session_state["agent_id"])
                else:
                    default_idx = 0
                choice = st.selectbox("Target namespace", options, index=default_idx)

            if choice == CREATE_OPT:
                new_id = st.text_input("New namespace ID", placeholder="e.g. my-memory-namespace")
                create_clicked = st.button(
                    "Create namespace",
                    use_container_width=True,
                    disabled=not new_id.strip(),
                )
                if create_clicked:
                    ok, msg = _create_agent(api_key, new_id.strip())
                    if ok:
                        st.success(msg)
                        # list_agents() on some backends lags right after a create,
                        # so make sure the new namespace shows up regardless.
                        fetched = _fetch_agents(api_key)
                        if new_id.strip() not in fetched:
                            fetched.append(new_id.strip())
                        st.session_state["agents"] = fetched
                        st.session_state["agent_id"] = new_id.strip()
                        st.rerun()
                    else:
                        st.error(msg)
                agent_id = st.session_state.get("agent_id", "")
            else:
                agent_id = choice
                st.session_state["agent_id"] = agent_id

            if agent_id:
                st.caption(f"Selected: `{agent_id}`")
        else:
            st.caption("Enter your API key to load your namespaces.")

        st.divider()
        st.markdown("**Supported providers:**")
        for provider, name in PROVIDERS.items():
            st.markdown(
                f'<img src="{PROVIDER_LOGOS[provider]}" height="16" style="vertical-align:middle;margin-right:6px">{name}',
                unsafe_allow_html=True,
            )
        st.divider()
        st.markdown("[GitHub](https://github.com/moorcheh-ai/memanto) · [Docs](https://docs.memanto.ai)")
    return api_key, agent_id


def main():
    api_key, agent_id = sidebar()

    st.title("🧠 Memanto Migration")
    st.markdown("Upload your AI conversation export and migrate your memories into Memanto.")

    col1, col2, col3 = st.columns(3)
    source = None
    with col1:
        st.markdown(
            f'<div style="text-align:center"><img src="{PROVIDER_LOGOS["chatgpt"]}" height="48" style="margin-bottom:8px"></div>',
            unsafe_allow_html=True,
        )
        if st.button("ChatGPT", use_container_width=True):
            st.session_state["source"] = "chatgpt"
    with col2:
        st.markdown(
            f'<div style="text-align:center"><img src="{PROVIDER_LOGOS["claude"]}" height="48" style="margin-bottom:8px"></div>',
            unsafe_allow_html=True,
        )
        if st.button("Claude", use_container_width=True):
            st.session_state["source"] = "claude"
    with col3:
        st.markdown(
            f'<div style="text-align:center"><img src="{PROVIDER_LOGOS["gemini"]}" height="48" style="margin-bottom:8px"></div>',
            unsafe_allow_html=True,
        )
        if st.button("Gemini", use_container_width=True):
            st.session_state["source"] = "gemini"

    source = st.session_state.get("source")

    if not source:
        st.info("Select a provider above to get started.")
        st.divider()
        st.markdown("### How it works")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**1. Export**\nDownload your conversation history from ChatGPT, Claude or Gemini.")
        with c2:
            st.markdown("**2. Upload**\nDrop the ZIP file here. Nothing leaves your machine until you click Migrate.")
        with c3:
            st.markdown("**3. Own it**\nYour memories land in Memanto and export as portable OKF markdown.")
        return

    st.markdown(f"### {PROVIDERS[source]} Migration")
    st.markdown(
        f'<img src="{PROVIDER_LOGOS[source]}" height="32" style="vertical-align:middle;margin-right:8px">',
        unsafe_allow_html=True,
    )
    st.info(EXPORT_INSTRUCTIONS[source])

    uploaded = st.file_uploader(
        f"Upload your {PROVIDERS[source]} export ZIP",
        type=["zip"],
        key=f"upload_{source}",
    )

    if not uploaded:
        return

    file_bytes = uploaded.read()

    with st.spinner("Parsing export..."):
        export = _load_export_from_bytes(file_bytes, source)

    memory_count = len(export.get("memories", []))
    st.success(f"Loaded {memory_count} conversation records from the export.")

    if st.button("🔍 Preview mapped memories (dry run)", use_container_width=True):
        with st.spinner("Mapping records..."):
            rows, summary = _run_dry_run(source, export)
        st.session_state["preview_rows"] = rows
        st.session_state["preview_summary"] = summary

    if "preview_rows" in st.session_state and st.session_state.get("preview_rows"):
        rows = st.session_state["preview_rows"]
        summary = st.session_state["preview_summary"]

        st.divider()
        st.markdown("### Preview")
        m1, m2, m3 = st.columns(3)
        m1.metric("Source records", summary["source_count"])
        m2.metric("Mapped memories", summary["mapped_count"])
        m3.metric("Skipped (empty)", summary["skipped"])

        if summary["type_counts"]:
            st.markdown("**Type breakdown**")
            st.json(summary["type_counts"])

        st.markdown("**Sample memories**")
        for row in rows[:5]:
            with st.expander(row.get("title", "Memory")[:80], expanded=False):
                st.markdown(f"**Content:**\n\n{row.get('content', '')[:600]}")
                st.markdown(f"**Type:** `{row.get('type') or 'auto'}`  |  **Source:** `{row.get('source')}`  |  **Provenance:** `{row.get('provenance')}`")

        if len(rows) > 5:
            st.caption(f"...and {len(rows) - 5} more memories")

        st.divider()

        if not agent_id:
            st.warning("Select or create a target namespace in the sidebar to migrate.")
        elif not api_key:
            st.warning("Enter your Moorcheh API Key in the sidebar to migrate.")
        else:
            if st.button(f"🚀 Migrate {summary['mapped_count']} memories into {agent_id}", type="primary", use_container_width=True):
                with st.spinner(f"Migrating {summary['mapped_count']} memories..."):
                    result = _do_migrate(source, export, agent_id, api_key)

                if result["failed"] == 0:
                    st.success(f"Migration complete! {result['imported']} memories imported into agent `{agent_id}`.")
                else:
                    st.warning(f"Done with errors. Imported: {result['imported']}, Failed: {result['failed']}")

                st.markdown("**Migration summary**")
                st.json(result)

                st.divider()
                st.markdown("### Export to OKF")
                st.code(f"memanto memory export --okf --output okf_bundle/ --agent {agent_id}", language="bash")
                st.caption("Run the above command in your terminal to export your memories as portable markdown.")


if __name__ == "__main__":
    main()