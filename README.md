# process-anomaly-detection

Detecting anomalous process activity on Windows endpoints.

The goal of this project is to surface unusual or suspicious process behaviour —
the kind that indicates malware, living-off-the-land abuse, misconfiguration, or
otherwise out-of-profile activity — by modelling what *normal* process execution
looks like on a host and flagging deviations from it.

## Architecture

The project is split into **three fully independent components** that communicate
only through a line-oriented **NDJSON contract** — one JSON object per process
event, per line:

| Component | Directory | Runtime | Dependencies |
|-----------|-----------|---------|--------------|
| **Collector** — `process_stream` | [`process_stream/`](process_stream/README.md) | Windows endpoint (Administrator) | `pywintrace`, `psutil`, `pywin32` |
| **Model** — `anomaly_detection_model` | [`anomaly_detection_model/`](anomaly_detection_model/README.md) | Any OS, anywhere | `scikit-learn`, `numpy`, `joblib` |
| **Monitor** — `process_monitor` | [`process_monitor/`](process_monitor/README.md) | Any ANSI terminal | none (standard library) |

The collector captures and enriches every process start on a Windows host. The
model trains an unsupervised baseline of normal process starts and scores new
ones for anomalousness. They share **no code and no dependencies**: the collector
never imports the model, the model never imports the collector. The model can be
trained and run on a completely different system — it only consumes the recorded
NDJSON stream.

```
Windows host                  Any system                 Any terminal
┌────────────────────┐ NDJSON ┌────────────────────┐ scored ┌────────────────────┐
│  process_stream    │ ─────► │  anomaly_detection │ ─────► │  process_monitor   │
│  (ETW + enrichment)│ stream │  (train / score)   │ stream │  (live ranked TUI) │
└────────────────────┘        └────────────────────┘        └────────────────────┘
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
3. **Alerting / review** — surface the highest-scoring events for triage. The
   [`process_monitor`](process_monitor/README.md) consumes the scored stream and
   renders a live, non-scrolling terminal table with the most suspicious process
   at the top.

## Repository layout

The repo is a small **workspace** over two self-contained subprojects. The root
`pyproject.toml` builds no code — it only carries shared dev tooling and the
`pytest` / `ruff` configuration so you can work from the root. Each subproject
declares its own runtime dependencies and can be built, installed, and shipped on
its own.

```
process-anomaly-detection/
├── pyproject.toml              # workspace root: dev tooling + shared tool config (no runtime code)
├── process_stream/             # Collector subproject
│   ├── pyproject.toml          #   its own dependencies (single source of truth)
│   └── process_stream/         #   importable package  → python -m process_stream
├── anomaly_detection_model/    # Model subproject
│   ├── pyproject.toml          #   its own dependencies (single source of truth)
│   └── model/                  #   importable package  → python -m model
└── process_monitor/            # Monitor subproject
    ├── pyproject.toml          #   no runtime dependencies (standard library)
    └── process_monitor/        #   importable package  → python -m process_monitor
```

## Quick start

Each subproject is installed separately, so the two modules never share an
environment requirement. To run them with `python -m ...` from the repo root,
install whichever one(s) you need as editable packages.

### 1. Collect a baseline (Windows, elevated)

```powershell
pip install -e process_stream          # installs pywintrace, psutil, pywin32
python -m process_stream > processes.ndjson
```

This writes one enriched JSON object per line. See the
[`process_stream` README](process_stream/README.md) for the full schema, options,
and limitations.

### 2. Train and score

The model is **optional** and runs independently of the collector. You can run it
on the same machine straight from the repo root, or copy `processes.ndjson` to a
different system entirely.

```bash
pip install -e anomaly_detection_model  # installs scikit-learn, numpy, joblib only
```

Then, from the repo root, train and score:

```powershell
python -m model train --input processes.ndjson --out model.joblib
python -m model score --model model.joblib --input new.ndjson
```

On a Windows host with both installed you can pipe the live collector straight
into the scorer:

```powershell
python -m process_stream | python -m model score --model model.joblib
```

### 3. Watch live (optional)

The monitor takes over the terminal and shows a continuously updating, ranked
table of processes — most suspicious first, one colour-coded line each, with the
row count scaling to the terminal height. It has no dependencies:

```powershell
pip install -e process_monitor
python -m process_stream | python -m model score --model model.joblib | python -m process_monitor
```

Press `Ctrl+C` to quit; the terminal scrollback is restored on exit. See the
[`process_monitor` README](process_monitor/README.md) for details.

To run the model as a standalone project on a different system instead, copy the
`anomaly_detection_model/` directory across and invoke it the same way. See the
[`anomaly_detection_model` README](anomaly_detection_model/README.md) for the
scoring semantics, output fields, and tuning options.

## Development

From the repo root, install both subprojects editable plus the shared dev tools,
then run the full test suite in one go:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e process_stream -e anomaly_detection_model -e process_monitor   # the three modules (kept isolated)
pip install -e ".[dev]"                                    # pytest + ruff
pytest                                                     # discovers both subprojects' tests
ruff check .
```

On non-Windows machines the collector's Windows-only dependencies can't install;
install just the model instead (`pip install -e anomaly_detection_model` and
`pip install -e ".[dev]"`) — its tests run anywhere.

## Requirements

- **Collector**: Windows, Python 3.9+, Administrator (ETW real-time sessions).
- **Model**: Python 3.9+ on any OS; no Windows-specific dependencies.
