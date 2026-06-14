"""SHA-256 hashing of image files (best-effort, non-blocking, size-capped)."""

from __future__ import annotations

import hashlib
from typing import Optional, Tuple

_CHUNK = 1024 * 1024  # 1 MiB
_MAX_HASH_BYTES = 256 * 1024 * 1024  # cap to keep the stream non-blocking


def hash_image(path: str) -> Tuple[Optional[str], bool]:
    """Return ``(sha256_hex, truncated)`` for ``path``.

    ``sha256_hex`` is the lower-case digest, or ``None`` if the file could not be
    read (locked / deleted / access denied). ``truncated`` is ``True`` when the
    file exceeded the size cap and only a prefix was hashed. Never raises.
    """
    if not path:
        return None, False
    digest = hashlib.sha256()
    read = 0
    truncated = False
    try:
        with open(path, "rb", buffering=0) as fh:
            while read < _MAX_HASH_BYTES:
                chunk = fh.read(min(_CHUNK, _MAX_HASH_BYTES - read))
                if not chunk:
                    break
                digest.update(chunk)
                read += len(chunk)
            else:
                # Hit the cap with bytes still remaining.
                if fh.read(1):
                    truncated = True
    except (OSError, ValueError):
        return None, False
    return digest.hexdigest(), truncated
