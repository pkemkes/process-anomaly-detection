"""Pure feature-extraction functions for process-start records.

Every function here is deterministic and side-effect-free: it takes a parsed
record ``dict`` (one NDJSON line from the collector) and returns plain Python
values. Three feature groups are produced:

* **Categorical** values (single fields and lineage pairs) -- used by the
  frequency/rarity head and for analyst explanations.
* **Boolean trust flags** -- signing/elevation/path signals encoded as floats.
* **Command-line scalars** -- length, entropy, suspicious-token counts, etc.

A ``null`` field is *informative*, so missing values map to an explicit
``__missing__`` sentinel rather than being silently dropped.
"""

from __future__ import annotations

import datetime
import math
from typing import Dict, Optional

from .paths import image_name as _image_name
from .vocab import Vocabulary

# Explicit "field absent" category so the model can learn a frequency for it.
MISSING = "__missing__"

# Epoch seconds below this are treated as "no real timestamp" (e.g. a backfilled
# ``create_time`` of 1970-01-01). Roughly 2001-09-09; any plausible process start
# is well after it, so it cleanly rejects the epoch-zero sentinel.
_MIN_VALID_EPOCH = 1_000_000_000.0

# Lower-cased substrings that frequently appear in obfuscated / LOLBin command
# lines. Counted (not individually flagged) so the signal generalizes.
_SUSPICIOUS_SUBSTRINGS = (
    "-enc",
    "-encodedcommand",
    "-nop",
    "-noprofile",
    "-w hidden",
    "-windowstyle hidden",
    "-executionpolicy bypass",
    "bypass",
    "iex",
    "invoke-expression",
    "frombase64string",
    "downloadstring",
    "downloadfile",
    "webclient",
    "http://",
    "https://",
)

# Ordered categorical fields (single values first, then lineage pairs). Order is
# fixed so the frequency feature columns are stable across train/score.
#
# NOTE: ``integrity_level`` and ``path_bucket`` are intentionally NOT standalone
# frequency features. Globally, ``High`` integrity and AppData/Temp paths are
# rare, so an unconditioned surprise on them flags every elevated or
# AppData-launched process regardless of identity. They are instead modelled
# *conditioned on the image* via the ``pair_image_*`` features below (and are
# still tracked by the vocabulary for that conditioning -- see CONTEXT_FIELDS).
CATEGORICAL_FIELDS = (
    "image_name",
    "signer",
    "company_name",
    "original_file_name",
    "signature_status",
    "logon_type",
    "user",
    "parent_image_name",
    "pair_parent_image",
    "pair_user_image",
    # Identity-conditioned trust pairs, scored as -log P(target | image) so the
    # model learns that elevation / integrity / path is *normal for a given
    # image* (e.g. an admin python.exe) instead of penalizing globally-rare
    # values. See CONDITIONAL_PAIRS.
    "pair_image_path",
    "pair_image_integrity",
    "pair_image_elevated",
)

# Context-only categorical fields: tracked by the vocabulary (so conditional
# surprises have the per-image base counts and target cardinalities they need)
# but never emitted as standalone frequency features.
CONTEXT_FIELDS = (
    "path_bucket",
    "integrity_level",
    "is_elevated",
)

# Pair feature -> (context field, target field). These are scored with the
# conditional surprise -log P(target | context) instead of a joint-frequency
# surprise.
CONDITIONAL_PAIRS = {
    "pair_image_path": ("image_name", "path_bucket"),
    "pair_image_integrity": ("image_name", "integrity_level"),
    "pair_image_elevated": ("image_name", "is_elevated"),
}

# Temporal pair feature -> (context field, bucket kind). These answer "is this
# time-of-run normal *for this image*?" and are scored with the conditional
# temporal surprise -log P(bucket | image), kept in their own head (Head C) so an
# odd time can raise the final score even when identity surprise is ~0. They are
# also emitted as feature columns so the Isolation Forest can use them for
# interactions.
TEMPORAL_PAIRS = {
    "pair_image_hour": ("image_name", "hour"),
    "pair_image_dow": ("image_name", "dow"),
}

# Ordered temporal pair fields (stable column order).
TEMPORAL_FIELDS = tuple(TEMPORAL_PAIRS)

# Ordered numeric (non-frequency) feature columns: boolean flags then cmdline
# scalars. Combined with the per-field frequency columns to form the vector fed
# to the Isolation Forest.
#
# Elevation and path-writability are deliberately *excluded* here: as standalone
# globally-rare booleans they dominated the StandardScaler -> Isolation Forest
# (a rare value z-scores to a large deviation regardless of the image). Those
# signals are instead captured, conditioned on the image, by the
# ``pair_image_elevated`` / ``pair_image_integrity`` / ``pair_image_path``
# frequency features above.
BOOLEAN_FLAGS = (
    "is_signed",
    "signer_is_microsoft",
    "name_mismatch",
)

COMMANDLINE_FEATURES = (
    "cmd_length",
    "cmd_token_count",
    "cmd_flag_count",
    "cmd_entropy",
    "cmd_suspicious_count",
    "cmd_non_alnum_ratio",
)

NUMERIC_FEATURES = BOOLEAN_FLAGS + COMMANDLINE_FEATURES

# Prefix distinguishing per-field frequency (surprise) columns from raw numerics.
FREQ_PREFIX = "freq::"

# Full ordered feature-vector column names fed to the Isolation Forest: identity
# surprise columns, then temporal surprise columns, then raw numerics.
FEATURE_COLUMNS = (
    tuple(FREQ_PREFIX + f for f in CATEGORICAL_FIELDS)
    + tuple(FREQ_PREFIX + f for f in TEMPORAL_FIELDS)
    + NUMERIC_FEATURES
)


def is_eligible(record: Dict[str, object]) -> bool:
    """Whether a record should be modelled (real ``process_start``, not pseudo)."""
    if record.get("is_pseudo"):
        return False
    return record.get("event", "process_start") == "process_start"


def _cat(value: object) -> str:
    """Normalize a categorical value to a non-empty string or the sentinel."""
    if value is None:
        return MISSING
    text = str(value).strip().lower()
    return text or MISSING


def _resolve_image_name(record: Dict[str, object]) -> str:
    """Image basename, preferring the precomputed field, falling back to image."""
    name = record.get("image_name")
    if name:
        return _cat(name)
    return _cat(_image_name(record.get("image")))  # type: ignore[arg-type]


def _resolve_parent_image_name(record: Dict[str, object]) -> str:
    """Parent image basename derived from the parent image path."""
    return _cat(_image_name(record.get("parent_image")))  # type: ignore[arg-type]


def _pair(left: str, right: str) -> str:
    """Join two normalized categorical values into a single pair key."""
    return f"{left}\x1f{right}"


def _parse_iso(raw: str) -> Optional[float]:
    """Parse an ISO-8601 timestamp (``...Z`` or offset) to UTC epoch seconds."""
    text = raw.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.timestamp()


def _epoch_seconds(record: Dict[str, object]) -> Optional[float]:
    """UTC epoch seconds for a record, preferring ``create_time``.

    ``create_time`` is the true process start, but backfill records carry an
    epoch-zero sentinel there; such values are rejected and ``timestamp`` (scan
    time) is used instead. A consistent UTC basis is used throughout so the
    per-image hour/day profile is not smeared by mixing zones.
    """
    for field_name in ("create_time", "timestamp"):
        raw = record.get(field_name)
        if not isinstance(raw, str):
            continue
        seconds = _parse_iso(raw)
        if seconds is None:
            continue
        if field_name == "create_time" and seconds < _MIN_VALID_EPOCH:
            continue
        return seconds
    return None


def window_id(record: Dict[str, object], window_minutes: int) -> Optional[int]:
    """Discrete ``window_minutes``-sized window index for ``record``'s start time.

    Returns ``None`` when the record carries no usable timestamp; callers treat
    that as a single collapsed window so undated records cannot inflate a
    recurrence count.
    """
    seconds = _epoch_seconds(record)
    if seconds is None:
        return None
    return int(seconds // (window_minutes * 60))


def hour_bucket(record: Dict[str, object], n_buckets: int) -> str:
    """Bucket the record's UTC hour-of-day into one of ``n_buckets`` bands."""
    seconds = _epoch_seconds(record)
    if seconds is None:
        return MISSING
    hour = datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc).hour
    return str(hour * n_buckets // 24)


def dow_bucket(record: Dict[str, object], n_buckets: int) -> str:
    """Bucket the record's UTC day-of-week into one of ``n_buckets`` bands.

    With ``n_buckets == 2`` the split is weekday (0) vs weekend (1); otherwise the
    seven weekdays (Mon=0) are scaled evenly into ``n_buckets`` bands.
    """
    seconds = _epoch_seconds(record)
    if seconds is None:
        return MISSING
    weekday = datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc).weekday()
    if n_buckets == 2:
        return "1" if weekday >= 5 else "0"
    return str(weekday * n_buckets // 7)


def temporal_values(record: Dict[str, object], vocab: Vocabulary) -> Dict[str, str]:
    """Temporal pair values (keys = :data:`TEMPORAL_FIELDS`) for ``record``.

    Each value is the ``image\\x1fbucket`` key used both to observe the per-image
    time profile during training and to look up the temporal surprise at score
    time. Bucket granularity comes from the vocabulary's stored config.
    """
    image = _resolve_image_name(record)
    return {
        "pair_image_hour": _pair(image, hour_bucket(record, vocab.hour_buckets)),
        "pair_image_dow": _pair(image, dow_bucket(record, vocab.dow_buckets)),
    }


def temporal_features(record: Dict[str, object], vocab: Vocabulary) -> Dict[str, float]:
    """Per-image conditional temporal surprises (keys prefixed :data:`FREQ_PREFIX`).

    Each :data:`TEMPORAL_FIELDS` entry is scored as ``-log P(bucket | image)``
    using row counts, gated by the vocabulary's minimum-sample threshold so thin
    images contribute no temporal signal.
    """
    image = _resolve_image_name(record)
    values = temporal_values(record, vocab)
    return {
        FREQ_PREFIX + "pair_image_hour": vocab.temporal_surprise(
            "image_name", image, "pair_image_hour", values["pair_image_hour"], vocab.hour_buckets
        ),
        FREQ_PREFIX + "pair_image_dow": vocab.temporal_surprise(
            "image_name", image, "pair_image_dow", values["pair_image_dow"], vocab.dow_buckets
        ),
    }


def categorical_values(record: Dict[str, object]) -> Dict[str, str]:
    """Categorical field values (feature fields + lineage pairs + context) for ``record``.

    Keys cover :data:`CATEGORICAL_FIELDS` (the frequency-feature fields) plus the
    :data:`CONTEXT_FIELDS` used only to condition the trust pairs. Every value is
    either a normalized string or the :data:`MISSING` sentinel.
    """
    image = _resolve_image_name(record)
    parent_image = _resolve_parent_image_name(record)
    path_bucket = _cat(record.get("path_bucket"))
    integrity_level = _cat(record.get("integrity_level"))
    is_elevated = _cat(record.get("is_elevated"))
    user = _cat(record.get("user"))
    return {
        "image_name": image,
        "signer": _cat(record.get("signer")),
        "company_name": _cat(record.get("company_name")),
        "original_file_name": _cat(record.get("original_file_name")),
        "signature_status": _cat(record.get("signature_status")),
        "logon_type": _cat(record.get("logon_type")),
        "user": user,
        "parent_image_name": parent_image,
        "pair_parent_image": _pair(parent_image, image),
        "pair_user_image": _pair(user, image),
        "pair_image_path": _pair(image, path_bucket),
        "pair_image_integrity": _pair(image, integrity_level),
        "pair_image_elevated": _pair(image, is_elevated),
        # Context-only fields (CONTEXT_FIELDS): tracked for conditioning, not
        # emitted as standalone frequency features.
        "path_bucket": path_bucket,
        "integrity_level": integrity_level,
        "is_elevated": is_elevated,
    }


def _flag(value: Optional[bool]) -> float:
    """Encode a tri-state boolean: ``True`` -> 1.0, ``False`` -> 0.0, ``None`` -> 0.5."""
    if value is None:
        return 0.5
    return 1.0 if value else 0.0


def boolean_flags(record: Dict[str, object]) -> Dict[str, float]:
    """Trust boolean flags encoded as floats (keys = BOOLEAN_FLAGS).

    Only globally-meaningful trust signals live here. Elevation/integrity and
    path-writability are intentionally absent -- they are modelled conditioned
    on the image via the ``pair_image_*`` frequency features instead.
    """
    return {
        "is_signed": _flag(record.get("is_signed")),  # type: ignore[arg-type]
        "signer_is_microsoft": _flag(record.get("signer_is_microsoft")),  # type: ignore[arg-type]
        "name_mismatch": _flag(record.get("name_mismatch")),  # type: ignore[arg-type]
    }


def shannon_entropy(text: str) -> float:
    """Shannon entropy (bits per character) of ``text``; 0.0 for empty input."""
    if not text:
        return 0.0
    counts: Dict[str, int] = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    length = len(text)
    entropy = 0.0
    for occurrences in counts.values():
        p = occurrences / length
        entropy -= p * math.log2(p)
    return entropy


def _count_suspicious(text: str) -> int:
    """Number of distinct suspicious substrings present in ``text``."""
    return sum(1 for needle in _SUSPICIOUS_SUBSTRINGS if needle in text)


def commandline_features(record: Dict[str, object]) -> Dict[str, float]:
    """Scalar characteristics of the normalized command line (keys = COMMANDLINE_FEATURES)."""
    raw = record.get("command_line_normalized")
    cmd = raw if isinstance(raw, str) else ""
    lowered = cmd.lower()
    tokens = cmd.split()
    flag_count = sum(1 for t in tokens if t.startswith("-") or t.startswith("/"))
    non_alnum = sum(1 for c in cmd if not c.isalnum() and not c.isspace())
    non_alnum_ratio = (non_alnum / len(cmd)) if cmd else 0.0
    return {
        "cmd_length": float(len(cmd)),
        "cmd_token_count": float(len(tokens)),
        "cmd_flag_count": float(flag_count),
        "cmd_entropy": shannon_entropy(cmd),
        "cmd_suspicious_count": float(_count_suspicious(lowered)),
        "cmd_non_alnum_ratio": non_alnum_ratio,
    }


def numeric_features(record: Dict[str, object]) -> Dict[str, float]:
    """All non-frequency numeric features (boolean flags + command-line scalars)."""
    features = boolean_flags(record)
    features.update(commandline_features(record))
    return features


def frequency_features(record: Dict[str, object], vocab: Vocabulary) -> Dict[str, float]:
    """Per-field smoothed-surprise features for ``record`` against ``vocab``.

    Keys are the categorical field names prefixed with :data:`FREQ_PREFIX`. The
    trust pairs in :data:`CONDITIONAL_PAIRS` use the *conditional* surprise
    -log P(target | context); all other fields use the marginal surprise.
    """
    cats = categorical_values(record)
    features: Dict[str, float] = {}
    for f in CATEGORICAL_FIELDS:
        conditional = CONDITIONAL_PAIRS.get(f)
        if conditional is None:
            features[FREQ_PREFIX + f] = vocab.surprise(f, cats[f])
        else:
            context_field, target_field = conditional
            features[FREQ_PREFIX + f] = vocab.conditional_surprise(
                f, cats[f], context_field, cats[context_field], target_field
            )
    return features


def feature_row(record: Dict[str, object], vocab: Vocabulary) -> Dict[str, float]:
    """Full numeric feature row (identity + temporal + flags + cmdline) for ``record``.

    Keys match :data:`FEATURE_COLUMNS`.
    """
    row = frequency_features(record, vocab)
    row.update(temporal_features(record, vocab))
    row.update(numeric_features(record))
    return row


def head_a_nll(freq_features: Dict[str, float]) -> float:
    """Head-A negative-log-likelihood: total identity surprise for a row."""
    return float(sum(freq_features.values()))


def head_c_nll(temporal_features_map: Dict[str, float]) -> float:
    """Head-C negative-log-likelihood: total temporal surprise for a row."""
    return float(sum(temporal_features_map.values()))

