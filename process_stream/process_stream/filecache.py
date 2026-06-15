"""Bounded file-fact cache backing the hash, signature, and PE-metadata features.

A binary is touched on disk at most once per (path, size, mtime) version. Lookups
run on the main/consumer thread, never in the ETW callback.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

from .hashing import hash_image
from .peinfo import pe_info
from .signing import signature

_MAX_ENTRIES = 4096


@dataclass
class FileFacts:
    image_size: Optional[int] = None
    image_hash: Optional[str] = None
    image_hash_truncated: bool = False
    is_signed: Optional[bool] = None
    signature_status: Optional[str] = None
    signer: Optional[str] = None
    signer_is_microsoft: Optional[bool] = None
    original_file_name: Optional[str] = None
    company_name: Optional[str] = None
    product_name: Optional[str] = None
    file_description: Optional[str] = None
    file_version: Optional[str] = None


_cache: "OrderedDict[Tuple[str, int, int], FileFacts]" = OrderedDict()


def file_facts(path: Optional[str]) -> FileFacts:
    """Return cached file facts for ``path``, computing them once per version."""
    if not path:
        return FileFacts()
    try:
        st = os.stat(path)
    except OSError:
        return FileFacts()

    key = (path.lower(), st.st_size, st.st_mtime_ns)
    cached = _cache.get(key)
    if cached is not None:
        _cache.move_to_end(key)
        return cached

    facts = FileFacts(image_size=st.st_size)

    digest, truncated = hash_image(path)
    facts.image_hash = digest
    facts.image_hash_truncated = truncated

    sig = signature(path)
    facts.is_signed = sig.is_signed
    facts.signature_status = sig.signature_status
    facts.signer = sig.signer
    facts.signer_is_microsoft = sig.signer_is_microsoft

    pe = pe_info(path)
    facts.original_file_name = pe.original_file_name
    facts.company_name = pe.company_name
    facts.product_name = pe.product_name
    facts.file_description = pe.file_description
    facts.file_version = pe.file_version

    _cache[key] = facts
    if len(_cache) > _MAX_ENTRIES:
        _cache.popitem(last=False)
    return facts
