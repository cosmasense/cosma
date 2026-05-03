# random100bench report

_Generated 2026-05-01T15:29:32.506194+00:00_

## System info

- Platform: `macOS-26.4.1-arm64-arm-64bit-Mach-O`
- Machine: `arm64`
- Python: `3.13.9`
- cosma_backend: `0.8.8` (api v1)
- CPUs: 8 logical (8 physical)
- RAM: 16.0 GiB total, 6.12 GiB free

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

- Enqueue all files: 0.018s
- Indexing wall time: **67.137s**
- Theoretical minimum: 42.557s
- Efficiency vs theoretical: **63.4%**

### Per-stage durations

- **parse**: n=100, total=163.6783s, mean=1636.8ms, p50=1333.3ms, p99=6973.9ms, max=7284.9ms
- **summarize**: n=100, total=0.0023s, mean=0.0ms, p50=0.0ms, p99=0.0ms, max=0.1ms
- **embed**: n=100, total=0.0462s, mean=0.5ms, p50=0.1ms, p99=6.4ms, max=16.7ms

## Resource usage

- Samples: 653 (100 ms cadence)
- Peak CPU%: **130.1%** (>100% = multi-core utilization)
- Mean CPU% (active samples): 16.6%
- Baseline RSS: 270.8 MiB
- Peak RSS: 896.6 MiB
- RSS growth: 625.7 MiB

## Outcomes (DB-truth)

- COMPLETE: 100

## Search queries fired during indexing

- Total: 10, OK: 10, Failed: 0
- Latency: mean=12.4ms, p50=4.8ms, p99=5.3ms, max=80.9ms

### Query timeline

| # | t+ (s) | query | latency (ms) | results |
|---|---|---|---|---|
| 0 | 5.985 | `summary` | 5.3 | 9 |
| 1 | 11.648 | `log` | 4.5 | 10 |
| 2 | 19.128 | `music` | 4.9 | 10 |
| 3 | 27.366 | `archive` | 4.6 | 10 |
| 4 | 32.225 | `document` | 5.3 | 10 |
| 5 | 37.779 | `data` | 4.5 | 10 |
| 6 | 42.154 | `spreadsheet` | 4.1 | 10 |
| 7 | 45.414 | `presentation` | 5.2 | 10 |
| 8 | 55.502 | `invoice` | 80.9 | 10 |
| 9 | 60.98 | `image` | 4.7 | 10 |