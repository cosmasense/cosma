# random100bench report

_Generated 2026-05-01T15:25:23.275924+00:00_

## System info

- Platform: `macOS-26.4.1-arm64-arm-64bit-Mach-O`
- Machine: `arm64`
- Python: `3.13.9`
- cosma_backend: `0.8.8` (api v1)
- CPUs: 8 logical (8 physical)
- RAM: 16.0 GiB total, 5.71 GiB free

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
- Indexing wall time: **166.347s**
- Theoretical minimum: 55.31s
- Efficiency vs theoretical: **33.2%**

### Per-stage durations

- **parse**: n=100, total=212.7294s, mean=2127.3ms, p50=1711.2ms, p99=6462.8ms, max=7634.3ms
- **summarize**: n=100, total=0.0024s, mean=0.0ms, p50=0.0ms, p99=0.0ms, max=0.0ms
- **embed**: n=100, total=0.0311s, mean=0.3ms, p50=0.1ms, p99=0.3ms, max=16.2ms

## Resource usage

- Samples: 1635 (100 ms cadence)
- Peak CPU%: **205.1%** (>100% = multi-core utilization)
- Mean CPU% (active samples): 6.0%
- Baseline RSS: 269.8 MiB
- Peak RSS: 921.5 MiB
- RSS growth: 651.7 MiB

## Outcomes (DB-truth)

- COMPLETE: 100

## Search queries fired during indexing

- Total: 10, OK: 10, Failed: 0
- Latency: mean=29.5ms, p50=5.0ms, p99=5.5ms, max=251.0ms

### Query timeline

| # | t+ (s) | query | latency (ms) | results |
|---|---|---|---|---|
| 0 | 6.608 | `summary` | 4.1 | 9 |
| 1 | 22.056 | `log` | 5.0 | 10 |
| 2 | 40.528 | `music` | 5.3 | 10 |
| 3 | 55.177 | `archive` | 4.7 | 10 |
| 4 | 71.171 | `document` | 5.5 | 10 |
| 5 | 86.032 | `data` | 5.0 | 10 |
| 6 | 101.803 | `spreadsheet` | 5.2 | 10 |
| 7 | 115.561 | `presentation` | 5.0 | 10 |
| 8 | 136.642 | `invoice` | 251.0 | 10 |
| 9 | 150.93 | `image` | 4.3 | 10 |