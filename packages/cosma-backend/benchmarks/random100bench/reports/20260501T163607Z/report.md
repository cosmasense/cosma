# random100bench report

_Generated 2026-05-01T16:34:39.435403+00:00_

## System info

- Platform: `macOS-26.4.1-arm64-arm-64bit-Mach-O`
- Machine: `arm64`
- Python: `3.13.9`
- cosma_backend: `0.8.8` (api v1)
- CPUs: 8 logical (8 physical)
- RAM: 16.0 GiB total, 6.02 GiB free

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

- Enqueue all files: 0.013s
- Indexing wall time: **87.337s**
- Theoretical minimum: 60.427s
- Efficiency vs theoretical: **69.2%**

### Per-stage durations

- **parse**: n=100, total=232.4103s, mean=2324.1ms, p50=1896.7ms, p99=7101.6ms, max=7371.4ms
- **summarize**: n=100, total=0.0023s, mean=0.0ms, p50=0.0ms, p99=0.0ms, max=0.1ms
- **embed**: n=100, total=0.0168s, mean=0.2ms, p50=0.1ms, p99=0.4ms, max=2.0ms

## Resource usage

- Samples: 846 (100 ms cadence)
- Peak CPU%: **188.1%** (>100% = multi-core utilization)
- Mean CPU% (active samples): 13.6%
- Baseline RSS: 289.1 MiB
- Peak RSS: 762.5 MiB
- RSS growth: 473.4 MiB

## Outcomes (DB-truth)

- COMPLETE: 100

## Search queries fired during indexing

- Total: 10, OK: 10, Failed: 0
- Latency: mean=9.2ms, p50=3.5ms, p99=6.1ms, max=57.6ms

### Query timeline

| # | t+ (s) | query | latency (ms) | results |
|---|---|---|---|---|
| 0 | 5.304 | `summary` | 4.5 | 9 |
| 1 | 11.815 | `log` | 3.5 | 10 |
| 2 | 19.376 | `music` | 3.4 | 10 |
| 3 | 31.837 | `archive` | 3.1 | 10 |
| 4 | 40.108 | `document` | 4.6 | 10 |
| 5 | 50.287 | `data` | 2.6 | 10 |
| 6 | 58.158 | `spreadsheet` | 2.9 | 10 |
| 7 | 62.93 | `presentation` | 3.3 | 10 |
| 8 | 74.546 | `invoice` | 57.6 | 10 |
| 9 | 81.372 | `image` | 6.1 | 10 |