# process-anomaly-detection

Detecting anomalous process activity on Windows endpoints.

The goal of this project is to surface unusual or suspicious process behaviour —
the kind that indicates malware, living-off-the-land abuse, misconfiguration, or
otherwise out-of-profile activity — by modelling what *normal* process execution
looks like on a host and flagging deviations from it.

## Approach

The pipeline is built in stages:

1. **Collection** — capture every process start on the machine in real time,
   enriched with the context needed to reason about it (identity, signing,
   lineage, timing). This is handled by the
   [`process_stream`](process_stream/README.md) module.
2. **Feature engineering** *(planned)* — turn the raw event stream into model-ready
   features (rarity of image/parent pairs, signing trust, path buckets, temporal
   patterns, ...).
3. **Modelling** *(planned)* — learn a baseline of normal behaviour and score new
   process starts for anomalousness.
4. **Alerting / review** *(planned)* — surface the highest-scoring events for
   triage.

## Components

| Component | Status | Description |
|-----------|--------|-------------|
| [`process_stream`](process_stream/README.md) | Available | Real-time NDJSON stream of Windows process starts, enriched for anomaly detection. The input data source for everything downstream. |
| Feature engineering | Planned | Transform the event stream into model features. |
| Anomaly model | Planned | Baseline normal behaviour and score deviations. |

## Quick start

The data-collection layer is usable today. From an **elevated** terminal on Windows:

```powershell
pip install -r requirements.txt
python -m process_stream > processes.ndjson
```

This writes one enriched JSON object per line, one per process start/stop. See the
[`process_stream` README](process_stream/README.md) for the full schema, options,
and limitations.

## Requirements

- Windows
- Python 3.9+
- **Administrator** privileges (ETW real-time sessions require elevation)
