# anomaly_detection_model

Unsupervised **anomaly detection** for process-start event streams. It learns a
one-class baseline of what *normal* process starts look like, then scores each
new process start for anomalousness — a ranking/triage signal, **not** a
malicious/benign verdict.

This is a **self-contained** project. It consumes the NDJSON records produced by
a collector (such as the [`process_stream`](../process_stream/README.md) Windows
agent) as plain dictionaries and has **no import dependency** on the collector or
on Windows. It can be trained and run on any operating system with only
`scikit-learn`, `numpy`, and `joblib`.

## Install

From the repository root, install this subproject as an editable package:

```bash
pip install -e anomaly_detection_model
```

Or, standalone from this directory (e.g. when copied to another machine):

```bash
pip install -e .
```

The only dependencies are `scikit-learn`, `numpy`, and `joblib` — none of the
collector's Windows-specific packages are required.

## Input contract

The model reads the NDJSON stream emitted by the collector: one JSON object per
line, one per process event. It uses these record fields when present (all
optional — a missing field is treated as an informative `__missing__` value):

- Identity / lineage: `image`, `image_name`, `parent_image`, `user`,
  `path_bucket`.
- Trust: `is_signed`, `signer`, `signer_is_microsoft`, `signature_status`,
  `company_name`, `original_file_name`, `name_mismatch`, `integrity_level`,
  `is_elevated`, `logon_type`.
- Command line: `command_line_normalized`.
- Routing: `event` (only `process_start` is scored), `is_pseudo`,
  `schema_version`.

## Train

Build a versioned artifact from a recorded baseline:

```bash
python -m model train --input processes.ndjson --out model.joblib
```

| Flag | Description |
|------|-------------|
| `--input` | Training NDJSON file (recorded collector output). |
| `--out` | Output artifact path (joblib). |
| `--alpha` | Laplace smoothing for the frequency tables (default `1.0`). |
| `--contamination` | `IsolationForest` contamination (`auto` or a float). |
| `--seed` | Random seed for reproducibility (default `0`). |

Training filters to real `process_start` records (drops `is_pseudo` and
`process_stop`), builds frozen frequency tables, fits a `StandardScaler` +
`IsolationForest`, fits the per-head score normalizer, and stores the rank-hint
thresholds. The stream `schema_version` is recorded in the artifact.

## Score

Score a recorded file or a piped stream. Output is the original record augmented
with anomaly fields, one JSON object per line:

```bash
# from a file
python -m model score --model model.joblib --input new.ndjson

# from stdin
type new.ndjson | python -m model score --model model.joblib
```

> Once installed editable (`pip install -e anomaly_detection_model`), the package
> is importable as `model` from anywhere, including the repo root. To pipe the
> live collector into the scorer on the same Windows host:
> `python -m process_stream | python -m model score --model model.joblib`.

| Flag | Description |
|------|-------------|
| `--model` | Artifact path (joblib). |
| `--input` | Input NDJSON file (default: stdin). |
| `--top-k` | Number of contributing fields to report (default `5`). |
| `--threshold-medium` / `--threshold-high` | Override the rank-hint cutoffs. |
| `--no-schema-guard` | Score even when the record `schema_version` differs. |

### Output

Each eligible record gains:

```json
{
  "...": "(original process_start fields)",
  "anomaly_score": 1.83,
  "anomaly_rank_hint": "high",
  "top_contributing_fields": [
    {"field": "pair(parent=winword.exe,image=powershell.exe)", "contribution_pct": 31.4},
    {"field": "ran_from_temp=true", "contribution_pct": 18.2},
    {"field": "is_signed=false", "contribution_pct": 12.7}
  ],
  "model_version": "1.0.0"
}
```

- `anomaly_score`: continuous, monotonic (**higher = more anomalous**). It is the
  weighted mean of two z-scored heads, so typical events sit near `0` and rare
  events are positive.
- `anomaly_rank_hint`: `low` / `medium` / `high`, derived from training-score
  quantiles (overridable with the `--threshold-*` flags).
- `top_contributing_fields`: the features that pushed the score up, for triage.
  Each entry carries a `contribution_pct` -- the share (in percent) of the
  record's total anomalous deviation attributable to that feature, so you can
  read at a glance by how much each field drove the score; the list is ordered
  by that share.
- Pseudo processes and `process_stop` records pass through with `null` anomaly
  fields.

## How it scores

Two heads are combined:

- **Head A — categorical NLL.** Sum of smoothed per-field "surprise"
  (`-log` of the Laplace-smoothed frequency) over single fields and lineage pairs
  (`parent→image`, `image→path`, `user→image`). Fully interpretable; unseen
  values get the maximum-surprise floor.
- **Head B — Isolation Forest.** Over the full engineered vector (frequency
  features + trust flags + command-line scalars) after standardization. Captures
  interactions the independent NLL head misses.

Each head is z-scored using training statistics stored in the artifact, then
combined (default equal weights).

## Module layout

| File | Responsibility |
|------|----------------|
| [model/featurize.py](model/featurize.py) | Pure feature extraction (categoricals, flags, command-line scalars). |
| [model/paths.py](model/paths.py) | Pure image-path classification (self-contained, no collector dependency). |
| [model/vocab.py](model/vocab.py) | Frequency tables: fit, smoothed-surprise lookup, (de)serialize. |
| [model/train.py](model/train.py) | Two-pass fit pipeline → versioned artifact. |
| [model/score.py](model/score.py) | Frozen-vocabulary streaming scorer + explanations. |
| [model/artifact.py](model/artifact.py) | Load/save, schema-version guard, joblib wrapper. |
| [model/__main__.py](model/__main__.py) | `train` / `score` CLI dispatch. |

## Tests

```bash
pip install pytest
python -m pytest
```

## Limitations

- A single short host capture is far too small a baseline; legitimate-but-rare
  software will read as anomalous. Use days of data, ideally fleet-wide, before
  trusting scores.
- **Cold start**: new-but-benign software always looks anomalous — pair with an
  allowlist / analyst feedback loop.
- **Anomalous ≠ malicious**: the output is a triage ranking, not a verdict.
- Scoring refuses (by default) on a `schema_version` mismatch, since feature
  extraction assumptions may have changed.
