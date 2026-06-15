"""PE version-resource metadata via version.dll (ctypes, zero extra deps)."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import _debug

_version = ctypes.WinDLL("version", use_last_error=True)

# FILE_VER_GET_NEUTRAL reads the binary's own (language-neutral) version resource
# instead of transparently loading the localized .mui satellite, which would
# otherwise report OriginalFilename values like "Cmd.Exe.MUI". Value from winver.h,
# verified against the GetFileVersionInfoEx documentation (not exposed by pywin32):
# https://learn.microsoft.com/windows/win32/api/winver/nf-winver-getfileversioninfoexw
_FILE_VER_GET_NEUTRAL = 0x02

_version.GetFileVersionInfoSizeExW.argtypes = [
    wintypes.DWORD,
    wintypes.LPCWSTR,
    ctypes.POINTER(wintypes.DWORD),
]
_version.GetFileVersionInfoSizeExW.restype = wintypes.DWORD

_version.GetFileVersionInfoExW.argtypes = [
    wintypes.DWORD,
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
]
_version.GetFileVersionInfoExW.restype = wintypes.BOOL

_version.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
_version.GetFileVersionInfoSizeW.restype = wintypes.DWORD

_version.GetFileVersionInfoW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
]
_version.GetFileVersionInfoW.restype = wintypes.BOOL

_version.VerQueryValueW.argtypes = [
    ctypes.c_void_p,
    wintypes.LPCWSTR,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(wintypes.UINT),
]
_version.VerQueryValueW.restype = wintypes.BOOL

_FIELDS = (
    ("original_file_name", "OriginalFilename"),
    ("company_name", "CompanyName"),
    ("product_name", "ProductName"),
    ("file_description", "FileDescription"),
    ("file_version", "FileVersion"),
)


@dataclass
class PeInfo:
    original_file_name: Optional[str] = None
    company_name: Optional[str] = None
    product_name: Optional[str] = None
    file_description: Optional[str] = None
    file_version: Optional[str] = None


def _translations(block: ctypes.Array) -> List[Tuple[int, int]]:
    ptr = ctypes.c_void_p()
    size = wintypes.UINT()
    if _version.VerQueryValueW(block, "\\VarFileInfo\\Translation", ctypes.byref(ptr), ctypes.byref(size)):
        count = size.value // 4
        if count and ptr.value:
            arr = ctypes.cast(ptr, ctypes.POINTER(wintypes.WORD * (count * 2)))
            words = arr.contents
            pairs = [(words[i * 2], words[i * 2 + 1]) for i in range(count)]
            if pairs:
                return pairs
    # Common fallbacks: US English / Unicode and US English / multilingual.
    return [(0x0409, 0x04B0), (0x0409, 0x04E4)]


def _query_string(block: ctypes.Array, lang: int, codepage: int, name: str) -> Optional[str]:
    ptr = ctypes.c_void_p()
    size = wintypes.UINT()
    sub = f"\\StringFileInfo\\{lang:04x}{codepage:04x}\\{name}"
    if _version.VerQueryValueW(block, sub, ctypes.byref(ptr), ctypes.byref(size)) and size.value and ptr.value:
        # ``size`` is the value length in characters including the terminating
        # null. Some binaries (e.g. certain AV vendors) report a length that
        # overruns the actual string, so cut at the first null rather than only
        # stripping trailing ones -- otherwise embedded nulls and the bytes of
        # the following resource field leak into the value.
        raw = ctypes.wstring_at(ptr, size.value)
        value = raw.split("\x00", 1)[0].strip()
        return value or None
    return None


def _load_block(path: str) -> Optional[ctypes.Array]:
    """Load the language-neutral version-info block, falling back to the default."""
    handle = wintypes.DWORD(0)
    size = _version.GetFileVersionInfoSizeExW(_FILE_VER_GET_NEUTRAL, path, ctypes.byref(handle))
    if size:
        block = ctypes.create_string_buffer(size)
        if _version.GetFileVersionInfoExW(_FILE_VER_GET_NEUTRAL, path, 0, size, block):
            return block
    # Fallback for binaries without a neutral resource.
    handle = wintypes.DWORD(0)
    size = _version.GetFileVersionInfoSizeW(path, ctypes.byref(handle))
    if not size:
        return None
    block = ctypes.create_string_buffer(size)
    if not _version.GetFileVersionInfoW(path, 0, size, block):
        return None
    return block


def pe_info(path: str) -> PeInfo:
    """Read the VS_VERSIONINFO string fields from ``path``. Never raises."""
    info = PeInfo()
    if not path:
        return info
    try:
        block = _load_block(path)
        if block is None:
            return info
        for lang, codepage in _translations(block):
            found = False
            for attr, name in _FIELDS:
                if getattr(info, attr) is None:
                    value = _query_string(block, lang, codepage, name)
                    if value is not None:
                        setattr(info, attr, value)
                        found = True
            if found and info.original_file_name:
                break
    except (OSError, ValueError, ctypes.ArgumentError) as exc:
        _debug.log("pe_info", exc)
        return info
    return info
