# process-anomaly-detection

Detecting anomalous process activity on Windows endpoints.

The goal of this project is to surface unusual or suspicious process behaviour —
the kind that indicates malware, living-off-the-land abuse, misconfiguration, or
otherwise out-of-profile activity — by modelling what *normal* process execution
looks like on a host and flagging deviations from it.

## Architecture

The project is split into **two fully independent components** that communicate
only through a line-oriented **NDJSON contract** — one JSON object per process
event, per line:

| Component | Directory | Runtime | Dependencies |
|-----------|-----------|---------|--------------|
| **Collector** — `process_stream` | [`process_stream/`](process_stream/README.md) | Windows endpoint (Administrator) | `pywintrace`, `psutil`, `pywin32` |
| **Model** — `anomaly_detection_model` | [`anomaly_detection_model/`](anomaly_detection_model/README.md) | Any OS, anywhere | `scikit-learn`, `numpy`, `joblib` |

The collector captures and enriches every process start on a Windows host. The
model trains an unsupervised baseline of normal process starts and scores new
ones for anomalousness. They share **no code and no dependencies**: the collector
never imports the model, the model never imports the collector. The model can be
trained and run on a completely different system — it only consumes the recorded
NDJSON stream.

```
Windows host                          Any system
┌──────────────────────┐  NDJSON  ┌───────────────────────┐
│  process_stream      │ ───────► │  anomaly_detection    │
│  (ETW + enrichment)  │  stream  │  (train / score)      │
└──────────────────────┘          └───────────────────────┘
```

## Pipeline stages

1. **Collection** — capture every process start in real time, enriched with the
   context needed to reason about it (identity, signing, lineage, timing).
   Handled by [`process_stream`](process_stream/README.md).
2. **Feature engineering + modelling** — turn the event stream into model-ready
   features (rarity of image/parent pairs, signing trust, path buckets,
   command-line characteristics), learn a baseline of normal behaviour, and score
   new starts (categorical rarity + Isolation Forest). Handled by
   [`anomaly_detection_model`](anomaly_detection_model/README.md).
3. **Alerting / review** *(planned)* — surface the highest-scoring events for
   triage.

## Quick start

### 1. Collect a baseline (Windows, elevated)

```powershell
pip install -r requirements.txt
python -m process_stream > processes.ndjson
```

This writes one enriched JSON object per line. See the
[`process_stream` README](process_stream/README.md) for the full schema, options,
and limitations.

### 2. Train and score

The model is **optional** and runs independently of the collector. You can run it
on the same machine straight from the repo root, or copy `processes.ndjson` to a
different system entirely.

Install its dependencies (only `scikit-learn`, `numpy`, `joblib`):

```bash
pip install -r anomaly_detection_model/requirements.txt
```

Then, from the repo root, train and score using the dotted module path:

```powershell
python -m anomaly_detection_model.model train --input processes.ndjson --out model.joblib
python -m anomaly_detection_model.model score --model model.joblib --input new.ndjson
```

On a Windows host you can also pipe the live collector straight into the scorer:

```powershell
python -m process_stream | python -m anomaly_detection_model.model score --model model.joblib
```

To run the model as a standalone project on a different system instead, copy the
`anomaly_detection_model/` directory across and invoke it as `python -m model`
from inside it. See the
[`anomaly_detection_model` README](anomaly_detection_model/README.md) for the
scoring semantics, output fields, and tuning options.

## Requirements

- **Collector**: Windows, Python 3.9+, Administrator (ETW real-time sessions).
- **Model**: Python 3.9+ on any OS; no Windows-specific dependencies.
