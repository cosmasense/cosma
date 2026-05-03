# random100bench report

_Generated 2026-05-01T15:20:32.768622+00:00_

## System info

- Platform: `macOS-26.4.1-arm64-arm-64bit-Mach-O`
- Machine: `arm64`
- Python: `3.13.9`
- cosma_backend: `0.8.8` (api v1)
- CPUs: 8 logical (8 physical)
- RAM: 16.0 GiB total, 5.97 GiB free

## Configuration

- Source: `/Users/ethanpan/Downloads`
- Files: 100/100 requested
- Seed: 42
- Queries: 10
- AI mode: mocked (real parser, mock summarize+embed)

## Selection

- examined: 3284
- unsupported: 2520
- too_large: 94
- unreadable: 0
- supported: 670
- Manifest cache: 100 found / 0 missing

## Timings

- Enqueue all files: 0.022s
- Indexing wall time: **80.623s**
- Theoretical minimum: 36.696s
- Efficiency vs theoretical: **45.5%**

### Per-stage durations

- **parse**: n=100, total=141.1375s, mean=1411.4ms, p50=933.8ms, p99=5932.6ms, max=6571.8ms
- **summarize**: n=100, total=0.0023s, mean=0.0ms, p50=0.0ms, p99=0.0ms, max=0.0ms
- **embed**: n=100, total=0.0204s, mean=0.2ms, p50=0.1ms, p99=0.3ms, max=6.5ms

## Resource usage

- Samples: 788 (100 ms cadence)
- Peak CPU%: **200.1%** (>100% = multi-core utilization)
- Mean CPU% (active samples): 11.8%
- Baseline RSS: 269.7 MiB
- Peak RSS: 890.3 MiB
- RSS growth: 620.6 MiB

## Outcomes (DB-truth)

- COMPLETE: 100

## Search queries fired during indexing

- Total: 10, OK: 10, Failed: 0
- Latency: mean=4.1ms, p50=1.7ms, p99=4.1ms, max=23.9ms

### Query timeline

| # | t+ (s) | query | latency (ms) | results |
|---|---|---|---|---|
| 0 | 0.751 | `summary` | 23.9 | 0 |
| 1 | 2.251 | `log` | 1.8 | 0 |
| 2 | 3.751 | `music` | 1.4 | 0 |
| 3 | 5.251 | `archive` | 1.9 | 0 |
| 4 | 6.751 | `document` | 1.3 | 0 |
| 5 | 8.251 | `data` | 1.6 | 0 |
| 6 | 9.751 | `spreadsheet` | 1.6 | 0 |
| 7 | 11.251 | `presentation` | 1.9 | 0 |
| 8 | 12.751 | `invoice` | 1.5 | 0 |
| 9 | 14.251 | `image` | 4.1 | 0 |