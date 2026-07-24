# Zep export guide

<!-- VIDEO: https://youtu.be/TODO-zep-export -->

## How to get your data

Zep data is pulled live via the API — no ZIP download needed.

You need a `ZEP_API_KEY` from [app.getzep.com](https://app.getzep.com) → Settings → API Keys.

## CLI command

```bash
export ZEP_API_KEY=your_key_here
memanto migrate zep --agent <id>
```

Or pass the key directly:

```bash
memanto migrate zep --api-key your_key_here --agent <id>
```

Dry-run (exports data but doesn't write to Memanto):

```bash
memanto migrate zep --dry-run
```

Or use the convenience script:

```bash
export ZEP_API_KEY=your_key_here
python scripts/migrate_zep.py [--dry-run] [--agent <id>]
```

## What gets exported

The exporter paginates `GET /api/v2/users-ordered` to list all users, then calls
`POST /api/v2/graph/edge/user/{user_id}` for each user to retrieve graph edge facts.

Each edge fact becomes one Memanto memory.

## Field mapping

| Source field | Memanto field | Notes |
|---|---|---|
| `fact` | `content` | The fact text |
| `valid_at` | `created_at` | Parsed via `_parse_dt` |
| hardcoded | `type` | `"fact"` |
| hardcoded | `source` | `"zep"` |
| hardcoded | `provenance` | `"imported"` |
