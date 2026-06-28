# Implementation Plan — Anomaly Scoring Redesign

This document is a developer hand-off for redesigning how the anomaly score is
computed in the `anomaly_detection_model` package. It describes the **final
target design**, the **reasoning** behind each decision, and the **concrete
implementation steps** (files, functions, formulas, config, tests).

It assumes the existing pipeline as the starting point:

- [`vocab.py`](anomaly_detection_model/model/vocab.py) — per-field frequency
  tables + smoothed surprise lookup.
- [`featurize.py`](anomaly_detection_model/model/featurize.py) — pure feature
  extraction; categorical singles + lineage/trust pairs, boolean flags,
  command-line scalars.
- [`train.py`](anomaly_detection_model/model/train.py) — two passes: build vocab,
  build matrix, fit `StandardScaler` + `IsolationForest`, z-score the two heads,
  store quantile thresholds.
- [`score.py`](anomaly_detection_model/model/score.py) — load artifact, score
  each record, emit explanations.
- [`artifact.py`](anomaly_detection_model/model/artifact.py) — versioned
  container + schema guard.
- [`record.py`](process_stream/process_stream/record.py) — stream schema
  (`ProcessRecord`); already carries `timestamp`, `create_time`, `hour_of_day`,
  `day_of_week`.

What is already implemented (keep): trust signals are **conditioned on the
image** via the `pair_image_integrity` / `pair_image_elevated` / `pair_image_path`
features, and the raw elevation/path booleans were removed from the Isolation
Forest. This plan builds on that.

---

## 1. The problem we are solving

The current per-field score is **frequency-based** negative log-likelihood:

```
surprise(field, value) = -log( (count + alpha) / (total + alpha*(card+1)) )
```

This has two structural flaws for our use case:

1. **Unbounded for rare items.** `-log(p)` grows without limit as `p -> 0`, so a
   value seen 10x in 94k records still looks like probability ~1e-4 and scores
   ~8. A *rare-but-recurring* process is therefore indistinguishable from one
   that has *essentially never* been seen.
2. **Scales with dataset size.** The same value becomes "more surprising" simply
   because the baseline grew. The score conflates "unusual" with "infrequent".

The desired behavior, stated as a requirement:

> A known admin script that runs only occasionally (e.g. once every few days)
> must still produce **low** surprise — it is an *expected* process. A high score
> should be reserved for things that have **essentially never** occurred like
> this before.

Additionally, we want a capability the frequency model cannot express at all:

> Detect a process that is **normal in identity but occurs at an odd time** —
> wrong hour of day or wrong day of week *for that specific process* (e.g. a
> backup that always runs at 02:00 suddenly running at 14:00).

---

## 2. Final design at a glance

Three signal groups feed three scoring heads; the heads are robustly normalized,
winsorized, weighted, combined, and finally mapped to a bounded `0..1` score.

```
record
 ├─ Group 1: identity regularity      → Head A (identity NLL)      ┐
 │   saturating distinct-window count   + Isolation Forest (Head B)│
 ├─ Group 2: trust conditioning        → Isolation Forest (Head B) ├─ robust-z (E1)
 │   P(integrity/elevated/path | image)                            │  winsorize (E3)
 └─ Group 3: temporal anomaly          → Head C (temporal NLL)     ┘  weighted sum
     P(hour|image), P(dow|image)        + Isolation Forest (Head B)    → percentile (E2)
                                                                       → anomaly_score 0..1
                                                                       → low/medium/high
```

The four design pillars and why we chose them:

- **A + F1 — Saturating "distinct-window" recurrence.** Replace frequency with a
  count that *saturates*: once a value has recurred in `K` distinct time windows
  it is "regular" and scores ~0. This makes the score **bounded** and
  **independent of dataset size**, directly satisfying the requirement.
- **Window granularity instead of days.** The saturation count is measured in
  **distinct W-minute windows** (default 60 min), not distinct days. Days require
  weeks of baseline before anything is "regular"; raw row counts are fooled by a
  single burst (5 spawns in 2 seconds would look regular). Windows are the middle
  ground: they accumulate on a multi-hour baseline yet collapse bursts (many runs
  inside one window count once).
- **F2 — Temporal anomaly as its own conditional head.** Learn *when* each image
  normally runs and score deviations. Conditioned on the image so "03:00 is odd"
  means "odd for this process", not "globally rare". Kept in a **separate head**
  so an odd time can raise the score even when identity surprise is ~0.
- **E — Recalibration.** Robust (median/MAD) normalization, winsorization, and a
  percentile mapping produce a bounded, interpretable score and stop any single
  field from dominating.

---

## 3. Group 1 — Identity regularity (saturating window-count)

### 3.1 Concept

For each marginal identity field, the surprise is a saturating function of the
number of **distinct W-minute windows** in which the value was observed during
training, capped at `K`:

```
c_win(field, value) = #{distinct W-minute windows the value appeared in}, capped at K
surprise(field, value) = -log( (min(c_win, K) + alpha) / (K + alpha) )
```

Properties:

- `c_win >= K`  → surprise ≈ 0 (regular; recurred across enough windows).
- `c_win == 0`  → fixed floor `-log(alpha/(K+alpha)) = log(1 + K/alpha)`.
- The denominator is the constant `K + alpha`, **not** the dataset total, so the
  score no longer depends on baseline size or field cardinality.

Window id is derived per record:

```
window_id = floor( epoch_seconds(create_time or timestamp) / (W * 60) )
```

Use `create_time` (true process start); fall back to `timestamp` when absent
(e.g. backfill records carry scan time, not start time).

### 3.2 Why windows, defaults, and the constraint

- Default `W = 60` minutes, `K = 5`.
- A value can only reach `K` windows if the baseline spans at least `K * W` of
  wall-clock time (5 h with the defaults). On shorter captures, use an
  **adaptive cap**: `K_eff = min(K, ceil(0.5 * distinct_windows_in_training))`,
  computed once at train time and stored in the artifact so train/score agree.
- Worked examples (W=60, K=5): admin script once every few days over weeks →
  many distinct hours → ≥5 windows → 0 surprise. Hourly process on a 5 h capture
  → 5 windows → regular. One-off installer spawning 5 children in a minute → 1
  window → still surprising.

### 3.3 Fields that use it

The marginal identity fields: `image_name`, `signer`, `company_name`,
`original_file_name`, `signature_status`, `logon_type`, `user`,
`parent_image_name`, and the lineage pairs `pair_parent_image`,
`pair_user_image`.

### 3.4 Conditioned trust pairs (Group 2) also saturate

The trust pairs answer "is this integrity/elevation/path normal *for this
image*?" They must honor the same "seen in ≥K windows = regular" rule, so they
use a saturating **conditional** form built from window counts:

```
conditional_surprise(pair, context, target_field) =
    -log( (min(c_win(pair), K) + alpha) / (min(c_win(context), K) + alpha*(M+1)) )
```

where `c_win(pair)` is the distinct-window count of the joint `(context,target)`
value, `c_win(context)` is the distinct-window count of the context value (the
image), and `M = cardinality(target_field)` reserves smoothing mass for unseen
target values.

Sanity checks (K=5, M=7):

- python.exe usually elevated → `c_win(python,high)=5`, `c_win(python)=5` →
  `-log(6/13) ≈ 0.78` (low). Elevation is learned as normal for python.
- common.exe never elevated → `c_win(common,high)=0`, `c_win(common)=5` →
  `-log(1/13) ≈ 2.56` (elevated is unusual for it).
- An image whose `high` integrity recurred in ≥K windows → ~0, satisfying the
  "occasional but regular = expected" requirement even at the conditional level.

### 3.5 Vocabulary changes (`vocab.py`)

- Constructor gains `window_minutes: int = 60` and `saturation_k: int = 5`
  (stored for serialization).
- Track distinct windows per value, **capped at K** to bound memory:
  `self._windows: Dict[field, Dict[value, set[int]]]`, never growing a set past
  `K` entries. Derive the capped count `c_win = len(window_set)`.
- `observe(field, value, window_id)` — new `window_id` argument; add to the
  capped set.
- `observe_row(values, window_id)` — thread `window_id` through.
- `surprise(field, value)` — implement the saturating formula in §3.1; unknown
  field still returns `0.0`.
- `conditional_surprise(...)` — reimplement per §3.4 (window-count based).
- `cardinality(field)` stays `len(self._counts_or_windows[field])`.
- `to_dict` / `from_dict` — serialize `window_minutes`, `saturation_k`, and the
  capped window counts (store the integer counts, not the sets, to keep
  artifacts small).

> Note: once we only keep capped counts, the raw `_totals` map is no longer used
> by the marginal surprise; keep it only if still needed for diagnostics.

---

## 4. Group 3 — Temporal anomaly (Head C)

### 4.1 Concept

Model **when each image normally runs** and score how unusual the current time
is *given the image*. Unlike Group 1, this signal must **sharpen with evidence**,
so it uses a smoothed conditional **probability** (not saturation):

```
temporal_surprise(image, hour_bucket) = -log( (count(image,hour)+alpha_t)
                                              / (count(image)+alpha_t*B_h) )
temporal_surprise(image, dow_bucket)  = -log( (count(image,dow)+alpha_t)
                                              / (count(image)+alpha_t*B_d) )
```

- Uses **row counts** (concentration is the signal here; bursts at the usual
  time only reinforce the normal profile).
- A well-established image with a tight profile (200x at 02:00) → low surprise at
  02:00, high surprise at 14:00.
- A thin image → smoothed flat distribution → low surprise at any hour (not
  enough evidence to call a time "odd"). The smoothing is the safety gate.

### 4.2 Buckets, gating, defaults

- `hour_bucket`: bucket `hour_of_day` into `B_h` bands. Default `B_h = 8`
  (3-hour bands) to balance resolution against sparsity; configurable to 24.
- `dow_bucket`: start with weekday/weekend (`B_d = 2`); switch to full 7 once the
  baseline reliably spans ≥3 weeks.
- **Minimum-sample gate:** only emit a non-zero temporal surprise when
  `count(image) >= temporal_min_samples` (default 20); otherwise return 0. This
  prevents thin entities from injecting temporal noise.
- `alpha_t` default 1.0.
- Conditioning scope: `image_name` to start. Optionally add `(user, image_name)`
  as a second scope (more memory, better fidelity); keep behind a config flag.

### 4.3 Data dependency (verify first)

`hour_of_day` / `day_of_week` were deferred in the original collector plan and
may be unpopulated. Before building this:

1. Confirm `train-input.ndjson` populates `hour_of_day`, `day_of_week`, and a
   usable `create_time` / `timestamp`.
2. If the derived fields are absent, derive them in `featurize.py` from
   `create_time` (preferred) or `timestamp`.
3. Use a **consistent local timezone** (mind DST) — mixing UTC and local smears
   the per-image hour profile and defeats detection.

### 4.4 Featurize / vocab changes

- `vocab.py`: add `temporal_surprise(context_field, context_value, bucket_field,
  bucket_value, n_buckets, min_samples)` using the row-count probability form;
  track temporal pair counts and per-image base counts (these can reuse the
  normal count tables, observed without window capping since they are
  row-frequency based — keep a separate counter set for temporal to avoid
  interfering with the saturated identity counts).
- `featurize.py`:
  - Add helpers `hour_bucket(record)` and `dow_bucket(record)`.
  - Add temporal pair fields `pair_image_hour`, `pair_image_dow` (and optional
    `pair_user_image_hour`, etc.).
  - Add a `temporal_features(record, vocab)` function returning the temporal
    surprises, and a `head_c_nll(temporal_features)` summing them.
  - Include the temporal surprises as columns in `FEATURE_COLUMNS` so the
    Isolation Forest (Head B) can also use them for interactions.

---

## 5. Group 2 recap — Trust conditioning (already shipped)

`pair_image_integrity`, `pair_image_elevated`, `pair_image_path` remain, now
computed with the **saturating conditional** surprise (§3.4). `integrity_level`
and `path_bucket` stay **context-only** (tracked for conditioning, not emitted as
standalone frequency features). The raw `is_elevated` / `ran_from_temp` /
`is_user_writable_path` booleans stay **out** of the Isolation Forest. Keep
`is_signed`, `signer_is_microsoft`, `name_mismatch` as boolean flags.

---

## 6. Heads and combination

### 6.1 Three heads

- **Head A — identity NLL:** sum of the Group 1 marginal + Group 2 conditional
  saturating surprises. Interpretable "how unfamiliar is this entity/lineage".
- **Head B — Isolation Forest:** over the full engineered vector (all saturating
  surprises + trust conditionals + temporal surprises + boolean flags +
  command-line scalars), after a scaler. Captures interactions.
- **Head C — temporal NLL:** sum of the Group 3 temporal surprises. Kept
  separate so an odd time alone (identity ~0) still raises the final score.

`head_a_nll` exists; add `head_c_nll`. Head B continues to use `score_samples`
negated (higher = more anomalous).

### 6.2 Recalibration (E1 + E3 + E2)

Computed on the training score distribution at fit time and stored in the
artifact:

1. **E1 — robust z per head.** Use median and MAD (scaled by 1.4826) instead of
   mean/std: `z = (head - median) / (1.4826*MAD)`. Floor the MAD by a small
   epsilon to avoid divide-by-zero on a degenerate head.
2. **E3 — winsorize per head.** Clip each head's value to its training
   `[p0.1, p99.9]` quantiles before the robust-z, so a single essentially-never-
   seen field cannot blow up a head.
3. **Combine:** `combined = wa*za + wb*zb + wc*zc`, default weights
   `(0.4, 0.3, 0.3)`.
4. **E2 — percentile mapping.** Store a compact quantile sketch (e.g. 1000
   evenly spaced quantiles) of the training `combined` scores. At score time,
   `anomaly_score = empirical_percentile(combined) ∈ [0,1]` — "more anomalous
   than X% of the baseline".
5. **Thresholds become percentile levels:** `threshold_high = 0.99`,
   `threshold_medium = 0.90` (config). `rank_hint` compares the percentile score
   to these levels.

> Important caveat to record in code comments: recalibration is **complementary**
> to the saturating surprise, not a substitute. It bounds the scale, makes
> thresholds mean "top X% of baseline", and prevents domination — but it
> preserves ranking. Only the saturating surprise (§3) moves a rare-but-present
> item to *low*.

---

## 7. Artifact and versioning changes (`artifact.py`)

Add fields:

- Config: `window_minutes`, `saturation_k` (the resolved `K_eff`), `alpha`,
  `alpha_t`, `temporal_min_samples`, `hour_buckets`, `dow_buckets`,
  `head_weights` (3-tuple), `threshold_medium`/`threshold_high` (now percentile
  levels).
- Normalization: per-head `median`, `mad`, and winsor bounds `(lo, hi)` for
  heads A, B, C.
- `combined_quantiles`: the sorted quantile sketch for the percentile map.

Bump `MODEL_VERSION` to `2.0.0` (scoring semantics change materially; old
artifacts are not compatible). Keep `schema_version` (stream schema) and the
`guard_schema` behavior unchanged.

---

## 8. Scoring path changes (`score.py`)

- Compute the three heads, apply winsorize + robust-z with stored params,
  combine with weights, map through `combined_quantiles` to a `0..1`
  `anomaly_score`.
- `rank_hint`: compare the percentile score to the stored percentile thresholds.
- Explanations: extend `_explain` so temporal contributors render as
  `pair(image=...,hour=...)` / `pair(image=...,dow=...)`; add their labels to
  `_PAIR_LABELS`. Continue to surface only the anomalous side (rarer/odd than
  baseline).
- `augment` output shape is unchanged except `anomaly_score` is now bounded
  `0..1`.

---

## 9. Training path changes (`train.py`)

- Thread a `window_id` (from `create_time`/`timestamp`) into `vocab.observe_row`.
- Resolve `K_eff` from the number of distinct training windows (§3.2) and store
  it.
- Build the matrix including temporal columns; fit scaler + Isolation Forest.
- Compute Head A / B / C arrays; derive medians, MADs, winsor bounds; combine;
  build the `combined_quantiles` sketch.
- Persist all new artifact fields. Keep determinism (fixed `seed`, recorded).
- New `train()` keyword args with the §6/§3/§4 defaults; expose the important
  ones through the `__main__` CLI (`--window-minutes`, `--saturation-k`,
  `--alpha-t`, `--temporal-min-samples`, `--weights`, `--high-quantile`,
  `--medium-quantile`).

---

## 10. Testing

- **vocab**
  - Saturating surprise: count ≥ K → ~0; unseen → floor; monotonic up to K;
    independent of total dataset size (same value, two very different totals →
    same surprise).
  - Window dedup: many observations within one window count once; observations
    across K windows reach the cap.
  - Saturating conditional: normal-for-context (recurred) → low; established
    context + unseen target → high; thin context → moderate floor.
  - Temporal: tight profile flags off-hours; thin entity (below
    `temporal_min_samples`) → 0 at all hours.
- **featurize**: `window_id` derivation (create_time vs timestamp fallback);
  hour/dow bucketing; temporal feature column set matches `FEATURE_COLUMNS`.
- **model (integration)**
  - Recurrence lowers score: a value seen across ≥K windows scores low even
    though it is a small fraction of the dataset.
  - Temporal anomaly: identical record at the entity's normal hour vs an odd hour
    → odd hour scores materially higher.
  - `anomaly_score` is bounded in `[0,1]`; thresholds map to documented
    percentiles.
  - Determinism (same seed → same scores) and artifact round-trip preserve
    scores.
  - Schema guard still refuses mismatched `schema_version`.

---

## 11. Build order (phased, each independently verifiable)

1. **Phase 1 — A + F1 (saturating window-count).** Implement window tracking and
   the saturating marginal + conditional surprise in `vocab.py`; thread
   `window_id` through `featurize`/`train`. Retrain on `train-input.ndjson`,
   re-score, compare rank distribution. Biggest single win.
2. **Phase 2 — F2 (temporal head).** First verify the temporal/`create_time`
   fields are populated; then add temporal features, Head C, and explanations.
   Retrain, compare; construct an explicit odd-time test record.
3. **Phase 3 — E (recalibration).** Add robust-z + winsorize + percentile output
   and switch thresholds to percentile levels. Retrain and recalibrate.

---

## 12. Caveats and operational notes

- **Short baselines** limit both the reachable `K` (need ≥ `K*W` of wall-clock
  span) and the day-of-week signal (need multiple weeks). The adaptive `K_eff`
  and the temporal min-sample gate keep the model inert rather than noisy when
  data is thin.
- **Timezone/DST** must be consistent for hour/day buckets.
- **`create_time` vs `timestamp`** for the window id: backfill records carry scan
  time; prefer `create_time` and fall back to `timestamp`.
- **Memory** is bounded: window sets are capped at `K` per value; temporal
  histograms are `buckets x entities` and can be pruned with the min-sample gate.
- **Compatibility**: `MODEL_VERSION = 2.0.0`; retrain is required, old artifacts
  cannot be loaded for scoring.
