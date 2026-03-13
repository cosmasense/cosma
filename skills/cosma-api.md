# CosmaSense API Reference for AI Agents

Use this skill when interacting with the CosmaSense backend API at `http://localhost:60534`.

## Base URL

```
http://localhost:60534
```

## Authentication

None required. All endpoints are unauthenticated (local-only service).

## Trailing Slash Rules

- Endpoints with trailing slash: `/api/status/`, `/api/search/`, `/api/watch/`, `/api/settings/`, `/api/updates/`
- Endpoints WITHOUT trailing slash: everything else (`/api/watch/jobs`, `/api/queue/status`, `/api/filters/config`, etc.)
- Using the wrong convention returns 308 redirect or 404.

---

## Endpoints

### Status

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status/` | Health check. Returns `{"jobs": N}` |

### Watch & Index

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/watch/` | Watch a directory. Body: `{"directory_path": "/abs/path"}` |
| GET | `/api/watch/jobs` | List all watch jobs |
| DELETE | `/api/watch/jobs/{job_id}` | Delete a watch job and its indexed files |
| POST | `/api/index/directory` | One-shot index a directory. Body: `{"directory_path": "/abs/path"}` |
| POST | `/api/index/file` | Index a single file. Body: `{"file_path": "/abs/path/file.pdf"}` |

### Search & Files

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/search/` | Hybrid search. Body: `{"query": "...", "limit": 10, "directory": null}` |
| GET | `/api/files/stats` | File statistics (total_files, file_types, etc.) |
| GET | `/api/files/{file_id}` | Get single file metadata by DB id |

### Queue & Scheduler

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/queue/status` | Queue state (paused, item counts) |
| POST | `/api/queue/pause` | Manually pause the queue |
| POST | `/api/queue/resume` | Resume a manually paused queue |
| GET | `/api/queue/items` | List all queued items |
| DELETE | `/api/queue/items/{item_id}` | Remove a queue item by UUID |
| GET | `/api/queue/scheduler` | Scheduler config and conditions_met |
| PUT | `/api/queue/scheduler` | Update scheduler rules |
| GET | `/api/queue/metrics` | System metrics (battery, CPU, temp, fan) |

### Filters

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/filters/config` | Current filter config (mode, patterns) |
| PUT | `/api/filters/config` | Update filter config |
| POST | `/api/filters/pattern` | Add a single pattern |
| DELETE | `/api/filters/pattern` | Remove a single pattern |
| POST | `/api/filters/test` | Test patterns against file paths |
| POST | `/api/filters/apply` | Apply filter changes (remove excluded files) |
| GET | `/api/filters/defaults` | Default filter patterns |
| POST | `/api/filters/reset` | Reset filters to defaults |

### Settings

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/settings/` | All settings grouped by section |
| PUT | `/api/settings/` | Partial update with dotted paths |
| GET | `/api/settings/defaults` | Default settings values |

### Real-time Updates

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/updates/` | SSE stream (requires `Accept: text/event-stream`) |

---

## Common Workflows

### Watch a new directory and wait for indexing

```bash
# 1. Start watching
curl -X POST http://localhost:60534/api/watch/ \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "/Users/me/Documents"}'

# 2. Monitor progress via SSE
curl -N -H "Accept: text/event-stream" http://localhost:60534/api/updates/
# Look for file_complete and directory_processing_completed events
```

### Search for files

```bash
curl -X POST http://localhost:60534/api/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "tax documents from 2023", "limit": 5}'
```

### Pause indexing during heavy work

```bash
curl -X POST http://localhost:60534/api/queue/pause
# ... do heavy work ...
curl -X POST http://localhost:60534/api/queue/resume
```

### Configure scheduler for battery-aware indexing

```bash
curl -X PUT http://localhost:60534/api/queue/scheduler \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": true,
    "combine_mode": "ALL",
    "rules": [
      {"type": "battery_level", "operator": "gt", "value": 50, "enabled": true},
      {"type": "power_source", "operator": "eq", "value": true, "enabled": true}
    ]
  }'
```

## SSE Event Opcodes

Events arrive as `event: update` with JSON `data` containing `opcode` and `data` fields.

**File processing:** `file_parsing`, `file_parsed`, `file_summarizing`, `file_summarized`, `file_embedding`, `file_embedded`, `file_complete`, `file_failed`, `file_skipped`

**File system:** `file_created`, `file_modified`, `file_deleted`, `file_moved`, `directory_deleted`, `directory_moved`

**Queue:** `queue_item_added`, `queue_item_updated`, `queue_item_processing`, `queue_item_completed`, `queue_item_failed`, `queue_item_removed`, `queue_paused`, `queue_resumed`

**Scheduler:** `scheduler_paused`, `scheduler_resumed`

**Watch:** `watch_started`, `watch_added`, `watch_removed`

**General:** `status_update`, `error`, `info`, `shutting_down`
