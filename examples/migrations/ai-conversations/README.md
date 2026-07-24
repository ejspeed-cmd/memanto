# ai-conversations migration showcase

Demonstrates the full **in → owned → portable** loop using Memanto's migration CLI.
Import conversation history and notes from 9 providers, validate recall with an LLM judge,
and export an OKF bundle — all from a single directory.

---

## Setup

```bash
cd examples/migrations/ai-conversations
pip install -r requirements.txt
cp .env.example .env
# fill in MOORCHEH_API_KEY and any provider keys you want to use
```

---

## Quick demo (no API key needed)

```bash
python migrate.py
```

Runs `--dry-run` against the included sample data for ChatGPT, Claude and Gemini.
Prints a per-source summary table. No writes are performed and no Memanto account is required.

---

## Full pipeline (live migration + OKF export)

```bash
# 1. set MOORCHEH_API_KEY in .env, then:
bash scripts/seed_and_export.sh my-agent-id
```

This will:
1. Create and activate the agent
2. Migrate ChatGPT, Claude, Gemini and LangGraph sample data
3. Export an OKF bundle to `okf_bundle/`

---

## Validation (recall parity)

After migrating, run the LLM judge against 10 golden Q&A pairs:

```bash
python validation/validate.py --agent my-agent-id
```

Requires `MOORCHEH_API_KEY` and `OPENROUTER_API_KEY`.
Exits 0 if 8+ of 10 questions pass (score ≥ 10/15 each).

---

## Provider mapping table

| Provider | CLI command | Source field → `content` | `type` | `source` |
|---|---|---|---|---|
| ChatGPT | `migrate conversations --source chatgpt` | `message.content.parts[]` (role=user) | auto | `chatgpt` |
| Claude | `migrate conversations --source claude` | `chat_messages[].text` (sender=human) | auto | `claude` |
| Gemini | `migrate conversations --source gemini` | `messages[].text` (role=user) | auto | `gemini` |
| Zep | `migrate zep` | `fact` | `fact` | `zep` |
| Hindsight | `migrate hindsight` | `text` | via `_coerce_type` | `hindsight` |
| LangGraph | `migrate langgraph --file dump.json` | `value.content` or `value` (str) | auto | `langgraph` |
| Notion | `migrate notion --file export.zip` | markdown body after frontmatter | `artifact` | `notion` |
| Obsidian | `migrate obsidian --file ./vault` | markdown body after frontmatter | `artifact` | `obsidian` |
| Chroma | `migrate chroma --collection <name>` | `document` | auto | `chroma` |

All providers set `provenance = "imported"` and `confidence = 0.8`.

---

## Per-provider guides

- [ChatGPT](docs/chatgpt.md)
- [Claude](docs/claude.md)
- [Gemini](docs/gemini.md)
- [Zep](docs/zep.md)
- [Hindsight](docs/hindsight.md)
- [LangGraph](docs/langgraph.md)
- [Notion](docs/notion.md)
- [Obsidian](docs/obsidian.md)
- [Chroma](docs/chroma.md)

---

## Directory structure

```
ai-conversations/
├── migrate.py              # single-command showcase runner
├── requirements.txt
├── .env.example
├── sample_data/            # de-identified export ZIPs (chatgpt, claude, gemini)
├── scripts/
│   ├── generate_sample_data.py
│   ├── dump_langgraph.py
│   ├── seed_and_export.sh
│   ├── migrate_chatgpt.py
│   ├── migrate_claude.py
│   ├── migrate_gemini.py
│   ├── migrate_notion.py
│   ├── migrate_obsidian.py
│   ├── migrate_zep.py
│   ├── migrate_hindsight.py
│   ├── migrate_langgraph.py
│   └── migrate_chroma.py
├── validation/
│   ├── golden_qa.json      # 10 Q&A pairs for recall validation
│   └── validate.py         # LLM judge runner
├── evidence/               # dry-run output captured for the PR
└── okf_bundle/             # OKF export committed as proof artifact
```
