"""Pure image-path helpers used by feature extraction.

Self-contained copies of the small path-classification functions so the model
package carries no import dependency on any specific collector. They operate on
raw Windows image-path strings as found in the NDJSON records.
"""

from __future__ import annotations

import os
from typing import Optional

# Path buckets considered typically user-writable (transient / untrusted).
_USER_WRITABLE_BUCKETS = ("Temp", "Downloads", "AppData", "User")


def image_name(image: Optional[str]) -> Optional[str]:
    """Basename of the image path, lower-cased (``None`` if absent)."""
    if not image:
        return None
    return os.path.basename(image).lower() or None


def _norm(path: str) -> str:
    return path.replace("/", "\\").lower()


def path_bucket(image: Optional[str]) -> Optional[str]:
    """Classify an image path into a coarse, model-friendly bucket."""
    if not image:
        return None
    p = _norm(image)
    if "\\system32\\" in p or p.endswith("\\system32"):
        return "System32"
    if "\\syswow64\\" in p or p.endswith("\\syswow64"):
        return "SysWOW64"
    if "\\appdata\\local\\temp\\" in p or "\\windows\\temp\\" in p or "\\temp\\" in p:
        return "Temp"
    if "\\downloads\\" in p:
        return "Downloads"
    if "\\appdata\\" in p:
        return "AppData"
    if "\\program files (x86)\\" in p or "\\program files\\" in p:
        return "ProgramFiles"
    if "\\users\\" in p:
        return "User"
    return "Other"


def is_user_writable_path(image: Optional[str]) -> Optional[bool]:
    """Whether the image lives in a typically user-writable location."""
    bucket = path_bucket(image)
    if bucket is None:
        return None
    return bucket in _USER_WRITABLE_BUCKETS
