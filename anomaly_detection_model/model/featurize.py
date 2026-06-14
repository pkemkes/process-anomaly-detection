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

import math
from typing import Dict, Optional

from .paths import image_name as _image_name
from .paths import is_user_writable_path
from .vocab import Vocabulary

# Explicit "field absent" category so the model can learn a frequency for it.
MISSING = "__missing__"

# Path buckets that indicate execution from a typically transient/user location.
_TEMP_BUCKETS = {"Temp", "Downloads", "AppData"}

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
CATEGORICAL_FIELDS = (
    "image_name",
    "path_bucket",
    "signer",
    "company_name",
    "original_file_name",
    "signature_status",
    "integrity_level",
    "logon_type",
    "user",
    "parent_image_name",
    "pair_parent_image",
    "pair_image_path",
    "pair_user_image",
)

# Ordered numeric (non-frequency) feature columns: boolean flags then cmdline
# scalars. Combined with the per-field frequency columns to form the vector fed
# to the Isolation Forest.
BOOLEAN_FLAGS = (
    "is_signed",
    "signer_is_microsoft",
    "is_elevated",
    "name_mismatch",
    "is_user_writable_path",
    "ran_from_temp",
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

# Full ordered feature-vector column names fed to the Isolation Forest.
FEATURE_COLUMNS = tuple(FREQ_PREFIX + f for f in CATEGORICAL_FIELDS) + NUMERIC_FEATURES


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


def categorical_values(record: Dict[str, object]) -> Dict[str, str]:
    """Categorical field values (single + lineage pairs) for ``record``.

    Keys match :data:`CATEGORICAL_FIELDS`; every value is either a normalized
    string or the :data:`MISSING` sentinel.
    """
    image = _resolve_image_name(record)
    parent_image = _resolve_parent_image_name(record)
    path_bucket = _cat(record.get("path_bucket"))
    user = _cat(record.get("user"))
    return {
        "image_name": image,
        "path_bucket": path_bucket,
        "signer": _cat(record.get("signer")),
        "company_name": _cat(record.get("company_name")),
        "original_file_name": _cat(record.get("original_file_name")),
        "signature_status": _cat(record.get("signature_status")),
        "integrity_level": _cat(record.get("integrity_level")),
        "logon_type": _cat(record.get("logon_type")),
        "user": user,
        "parent_image_name": parent_image,
        "pair_parent_image": _pair(parent_image, image),
        "pair_image_path": _pair(image, path_bucket),
        "pair_user_image": _pair(user, image),
    }


def _flag(value: Optional[bool]) -> float:
    """Encode a tri-state boolean: ``True`` -> 1.0, ``False`` -> 0.0, ``None`` -> 0.5."""
    if value is None:
        return 0.5
    return 1.0 if value else 0.0


def _ran_from_temp(path_bucket: object) -> Optional[bool]:
    """Whether the image path bucket is a transient/user-writable location."""
    if path_bucket is None:
        return None
    return path_bucket in _TEMP_BUCKETS


def boolean_flags(record: Dict[str, object]) -> Dict[str, float]:
    """Trust/elevation/path boolean flags encoded as floats (keys = BOOLEAN_FLAGS)."""
    return {
        "is_signed": _flag(record.get("is_signed")),  # type: ignore[arg-type]
        "signer_is_microsoft": _flag(record.get("signer_is_microsoft")),  # type: ignore[arg-type]
        "is_elevated": _flag(record.get("is_elevated")),  # type: ignore[arg-type]
        "name_mismatch": _flag(record.get("name_mismatch")),  # type: ignore[arg-type]
        "is_user_writable_path": _flag(is_user_writable_path(record.get("image"))),  # type: ignore[arg-type]
        "ran_from_temp": _flag(_ran_from_temp(record.get("path_bucket"))),
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

    Keys are the categorical field names prefixed with :data:`FREQ_PREFIX`.
    """
    cats = categorical_values(record)
    return {FREQ_PREFIX + f: vocab.surprise(f, cats[f]) for f in CATEGORICAL_FIELDS}


def feature_row(record: Dict[str, object], vocab: Vocabulary) -> Dict[str, float]:
    """Full numeric feature row (frequency + flags + cmdline) for ``record``.

    Keys match :data:`FEATURE_COLUMNS`.
    """
    row = frequency_features(record, vocab)
    row.update(numeric_features(record))
    return row


def head_a_nll(freq_features: Dict[str, float]) -> float:
    """Head-A negative-log-likelihood: total categorical surprise for a row."""
    return float(sum(freq_features.values()))

