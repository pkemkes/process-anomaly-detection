# Implementation Plan — Process Anomaly Detection

This document describes, in detail, how to implement the anomaly-detection layer
on top of the existing [`process_stream`](process_stream/README.md) NDJSON event
source. The goal: **train an unsupervised baseline of "normal" process starts
from a recorded stream, then score each new process for anomalousness at
classification time.**

The design intentionally mirrors the conventions of the existing
`process_stream` module (pure functions, dataclasses, a `__main__` CLI,
line-oriented streaming).

---

## 1. Problem framing

- The stream is **unlabeled** — there are no ground-truth "malicious" examples.
  This is therefore **one-class / unsupervised anomaly detection**: model what
  *normal* looks like, then flag deviations.
- The schema is dominated by **high-cardinality categorical** fields (images,
  signers, parent/child lineage, paths) plus a handful of numerics. The single
  strongest signal in this domain is **rarity** — of individual category values
  and, more importantly, of their *combinations*.
- Output is a continuous **anomaly score per process start**, plus the
  **fields that contributed most** to that score (for analyst triage). It is a
  *ranking/triage* signal, not a malicious/benign verdict.

### Non-goals (v1)

- No supervised classification (no labels available).
- No fleet-wide aggregation / cross-host modelling (single-host baseline first).
- No automated response — scoring and surfacing only.

---

## 2. Data handling

### 2.1 Record selection

At both train and score time, apply identical filtering:

| Rule | Reason |
|------|--------|
| Drop `is_pseudo == true` (pid 0/4, Registry, Memory Compression, System) | Null images, `1970` create-time sentinels — pure noise. |
| Train only on `event == "process_start"` | At scoring time the start is seen before the stop. |
| Keep `process_stop` separately (optional) | `lifetime_ms` / `exit_code` are a later, second-pass signal. |
| Treat `source` ("existing" backfill vs "new" live) as known | Backfill lacks `process_seq`, `session_id`, sometimes `command_line`. |

**Missing-field handling:** a `null` value is *informative*, not something to
silently impute. Every field gets an explicit `__missing__` sentinel category so
"field absent" is a value the model can learn a frequency for.

### 2.2 Schema versioning

- Persist the stream `schema_version` inside the model artifact.
- At score time, **refuse to score** (or warn loudly) on a `schema_version`
  mismatch, since feature extraction assumptions may have changed.

---

## 3. Feature engineering

Implemented as **pure functions** in a new `process_stream/model/featurize.py`,
taking a parsed record dict and returning a flat `dict[str, float]` plus a
parallel `dict[str, str]` of the raw categorical values (kept for
explanations). Three feature groups:

### 3.1 Categorical rarity (frequency) features

For each selected categorical field, replace the value with a **smoothed
log-frequency** learned from training:

```
score_value(field, value) = -log( (count[field][value] + alpha)
                                   / (total[field] + alpha * (cardinality[field] + 1)) )
```

- Higher = rarer = more surprising. Laplace smoothing (`alpha`, default `1.0`)
  guarantees unseen values at score time get the maximum-surprise floor.
- Single fields: `image_name`, `path_bucket`, `signer`, `company_name`,
  `original_file_name`, `signature_status`, `integrity_level`, `logon_type`,
  `user`.
- **Pair / lineage fields (highest value)** — concatenated keys:
  - `(parent_image_name, image_name)` — the core LOLBin signal.
  - `(image_name, path_bucket)` — known binary from an unusual location.
  - `(user, image_name)` — process unusual for this account.

### 3.2 Boolean / trust flags (0.0 / 1.0, with a third value for unknown)

- `is_signed`, `signer_is_microsoft`, `is_elevated`, `name_mismatch`
- Derived: `is_user_writable_path(image)` (reuse existing
  [`features.is_user_writable_path`](process_stream/features.py)), and a
  `ran_from_temp` flag (`path_bucket in {Temp, Downloads, AppData}`).

### 3.3 Command-line characteristics

Derived from `command_line_normalized` (engineered **scalars**, not raw
bag-of-words — scalars generalize across hosts and resist overfitting):

- length (chars), token count, number of flags/switches.
- Shannon entropy of the string (high entropy → encoded/obfuscated payloads).
- count of suspicious substrings: `-enc`, `-nop`, `-w hidden`, `iex`,
  `frombase64string`, `downloadstring`, `http://`, `https://`, etc.
- ratio of non-alphanumeric characters.

> Temporal features (`hour_of_day`, `day_of_week`) are **deferred** to v2: on a
> single short baseline they overfit. When added, encode cyclically
> (`sin`/`cos`).

---

## 4. Model

A two-headed scorer — one interpretable, one capturing interactions — combined
into a final score.

### 4.1 Head A — categorical negative-log-likelihood (NLL)

- Sum of the per-field smoothed surprise values from §3.1.
- Equivalent to a Naive-Bayes "surprise" score; **fully interpretable** — we can
  report exactly which field/pair was rare. Unseen images / signers /
  parent-child pairs naturally dominate.

### 4.2 Head B — Isolation Forest

- `sklearn.ensemble.IsolationForest` over the full engineered feature vector
  (frequency features + flags + command-line scalars), after a
  `StandardScaler`.
- Robust, fast, little tuning; `score_samples` gives a continuous score and it
  captures *interactions* the independent NLL head misses.

### 4.3 Combination

- Normalize each head to a comparable scale (fit on training: store mean/std or
  empirical quantiles), then combine (default: mean of the two normalized
  scores). The weighting is a config knob.
- Final output `anomaly_score` in a documented, monotonic range
  (higher = more anomalous).

### 4.4 Explanations

- Report the top-k contributing features by their individual surprise
  contribution (from Head A, plus the largest standardized feature deviations
  feeding Head B), e.g.
  `["pair(parent=winword.exe,image=powershell.exe)", "ran_from_temp", "is_signed=false"]`.

### 4.5 Future upgrades (v2+, only if validated need)

- Denoising **autoencoder** on encoded features (reconstruction error as score).
- **LOF** for local-density anomalies.
- Per-pair count-min sketch for memory-bounded fleet-scale vocabularies.

---

## 5. Module layout

New subpackage, mirroring existing style:

```
process_stream/
  model/
    __init__.py
    __main__.py        # CLI dispatch: `train` / `score` subcommands
    featurize.py       # pure feature-extraction functions (§3)
    vocab.py           # frequency tables: fit, smoothed-surprise lookup, (de)serialize
    train.py           # fit pipeline → versioned artifact (§6)
    score.py           # load artifact → stream scorer (§7)
    artifact.py        # load/save, schema_version guard, joblib wrapper
    README.md          # usage, schema, scoring semantics
```

- Dependencies added to [`requirements.txt`](requirements.txt) /
  [`pyproject.toml`](pyproject.toml): `scikit-learn`, `numpy`, `joblib`.
- No changes required to the existing collection modules.

---

## 6. Training pipeline (`train.py`)

```
python -m process_stream.model train \
    --input processes.ndjson \
    --out model.joblib \
    [--alpha 1.0] [--contamination auto] [--seed 0]
```

Steps:

1. Stream-read NDJSON; parse each line; **filter** per §2.1.
2. First pass: build **frequency tables** (`vocab.py`) for every single and
   pair field, recording counts, totals, and cardinalities.
3. Second pass: featurize every record (§3) into a numeric matrix; fit
   `StandardScaler` then `IsolationForest`.
4. Fit the **score-combination normalizer** (head means/stds or quantiles) on
   the training scores.
5. Persist a single **versioned artifact** (joblib) containing: frequency
   tables, scaler, isolation forest, normalizer params, feature column order,
   `alpha`, stream `schema_version`, model version, train-set summary stats.

Determinism: fixed `--seed`; the artifact records it.

---

## 7. Scoring pipeline (`score.py`)

```
python -m process_stream | python -m process_stream.model score --model model.joblib
# or
python -m process_stream.model score --model model.joblib --input new.ndjson
```

Per line (stateless, stream-friendly):

1. Parse; if `is_pseudo` or not a `process_start`, pass through untouched
   (optionally with `anomaly_score: null`).
2. Guard `schema_version` against the artifact.
3. Featurize using the **frozen** training vocabulary — unseen categories map to
   the smoothing floor (max surprise); **never refit at score time**.
4. Compute Head A + Head B, normalize, combine.
5. Emit the **original record augmented** with:
   ```json
   { "anomaly_score": 0.0, "anomaly_rank_hint": "low|medium|high",
     "top_contributing_fields": ["...","..."], "model_version": "..." }
   ```
6. Output one JSON object per line → composes directly with the existing
   pipe-based workflow.

Threshold/labeling (`low/medium/high`) is derived from training-score quantiles
stored in the artifact and overridable via `--threshold`.

---

## 8. Evaluation (no labels)

1. **Synthetic injection** — craft known-suspicious events
   (`powershell.exe -enc <b64>` parented by `winword.exe`, running from `Temp`;
   unsigned binary from `Downloads`) and assert they land in the top scores.
   Lives as unit/integration tests.
2. **Held-out time slice** — split the baseline by time; the held-out score
   distribution should resemble training (drift check).
3. **Top-N review** — manually inspect highest-scoring real events for
   plausibility; tune the threshold to a tolerable alert volume (e.g. top 0.1%).

---

## 9. Testing & quality

- **Unit tests** for every pure featurizer (deterministic, no I/O), the
  smoothed-surprise math (unseen value → floor), and vocab (de)serialization.
- **Golden test**: a tiny fixed NDJSON fixture → train → score → assert stable
  ordering of scores.
- **Round-trip test**: save/load artifact, identical scores before/after.
- Schema-version mismatch test (must refuse / warn).
- Keep functions pure and side-effect-free to match the existing codebase.

---

## 10. Known limitations & risks

- **Baseline size & breadth.** A single short host capture (hundreds of records,
  many pseudo) is far too small; legitimate-but-rare software will read as
  anomalous. Needs *days* of data, ideally fleet-wide, before trusting scores.
- **Cold start.** New-but-benign software always looks anomalous → pair with an
  allowlist / analyst feedback loop.
- **Anomalous ≠ malicious.** Correlated, not identical. Output is triage
  ranking.
- **Concept drift.** Periodic retraining / online frequency updates needed as
  the environment evolves (a v2 concern).

---

## 11. Phased delivery

| Phase | Deliverable |
|-------|-------------|
| 1 | `featurize.py` + `vocab.py` + unit tests (frequency + cmdline scalars). |
| 2 | `train.py` producing a versioned artifact; Head A (NLL) scoring only. |
| 3 | Head B (Isolation Forest) + score combination + `score.py` streaming CLI. |
| 4 | Explanations (top contributing fields) + thresholds. |
| 5 | Synthetic-injection eval + golden tests; README. |
| 6 (later) | Temporal features, autoencoder, `process_stop` lifetime signal, fleet aggregation. |
```
