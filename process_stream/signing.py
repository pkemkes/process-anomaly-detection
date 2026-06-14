"""Authenticode signature verification via WinVerifyTrust (ctypes, best-effort).

Handles both embedded and catalog signatures. Extracts the signer subject from
the embedded PKCS#7 or, for catalog-signed binaries, from the backing ``.cat``
file. Every entry point swallows errors and returns ``None`` rather than raising.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from typing import Optional

import win32con
import win32cryptcon
import winerror

from . import _debug


def _u32(value: int) -> int:
    """Normalize a (possibly sign-extended) 32-bit value to unsigned."""
    return value & 0xFFFFFFFF


# Numeric constants are sourced from pywin32 (an existing dependency) rather than
# transcribed, so the authoritative values ship with the library. pywin32 stores
# HRESULTs as signed 32-bit ints; they are masked to unsigned to match the codes
# WinVerifyTrust returns (also normalized via _u32).

# --- HRESULT status codes (winerror) ------------------------------------------
_ERROR_SUCCESS = _u32(winerror.ERROR_SUCCESS)
_TRUST_E_NOSIGNATURE = _u32(winerror.TRUST_E_NOSIGNATURE)
_TRUST_E_BAD_DIGEST = _u32(winerror.TRUST_E_BAD_DIGEST)
_TRUST_E_SUBJECT_NOT_TRUSTED = _u32(winerror.TRUST_E_SUBJECT_NOT_TRUSTED)
_CERT_E_EXPIRED = _u32(winerror.CERT_E_EXPIRED)
_CERT_E_UNTRUSTEDROOT = _u32(winerror.CERT_E_UNTRUSTEDROOT)
_CERT_E_CHAINING = _u32(winerror.CERT_E_CHAINING)
_CRYPT_E_SECURITY_SETTINGS = _u32(winerror.CRYPT_E_SECURITY_SETTINGS)

# --- WinVerifyTrust constants (wintrust.h; not exposed by pywin32) -------------
# Verified against the WINTRUST_DATA documentation:
# https://learn.microsoft.com/windows/win32/api/wintrust/ns-wintrust-wintrust_data
_WTD_UI_NONE = 2
_WTD_REVOKE_NONE = 0
_WTD_CHOICE_FILE = 1
_WTD_CHOICE_CATALOG = 2
_WTD_STATEACTION_VERIFY = 0x00000001
_WTD_STATEACTION_CLOSE = 0x00000002
# NOTE: WTD_SAFER_FLAG (0x100) is intentionally NOT used. Microsoft documents it
# as "Not supported" (it survives only in legacy MSDN sample code), and in
# practice it makes WinVerifyTrust report CERT_E_UNTRUSTEDROOT for binaries that
# are in fact correctly signed and chain to a trusted root (observed with some
# AV-vendor binaries), i.e. a false "untrusted".
_WTD_REVOCATION_CHECK_NONE = 0x00000010
_WTD_CACHE_ONLY_URL_RETRIEVAL = 0x00001000

# --- CryptQueryObject / cert constants (win32cryptcon) ------------------------
_CERT_QUERY_OBJECT_FILE = win32cryptcon.CERT_QUERY_OBJECT_FILE
_CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED_EMBED = win32cryptcon.CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED_EMBED
_CERT_QUERY_FORMAT_FLAG_BINARY = win32cryptcon.CERT_QUERY_FORMAT_FLAG_BINARY
_CMSG_SIGNER_CERT_INFO_PARAM = win32cryptcon.CMSG_SIGNER_CERT_INFO_PARAM
_X509_ASN_ENCODING = win32cryptcon.X509_ASN_ENCODING
_PKCS_7_ASN_ENCODING = win32cryptcon.PKCS_7_ASN_ENCODING
_CERT_FIND_SUBJECT_CERT = win32cryptcon.CERT_FIND_SUBJECT_CERT
_CERT_NAME_SIMPLE_DISPLAY_TYPE = win32cryptcon.CERT_NAME_SIMPLE_DISPLAY_TYPE

# --- file open constants (win32con) -------------------------------------------
_GENERIC_READ = _u32(win32con.GENERIC_READ)
_FILE_SHARE_READ = win32con.FILE_SHARE_READ
_OPEN_EXISTING = win32con.OPEN_EXISTING
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

_wintrust = ctypes.WinDLL("wintrust", use_last_error=True)
_crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]


# WINTRUST_ACTION_GENERIC_VERIFY_V2 {00AAC56B-CD44-11d0-8CC2-00C04FC295EE}
_WVT_GUID = GUID(
    0x00AAC56B,
    0xCD44,
    0x11D0,
    (0x8C, 0xC2, 0x00, 0xC0, 0x4F, 0xC2, 0x95, 0xEE),
)


class WINTRUST_FILE_INFO(ctypes.Structure):
    _fields_ = [
        ("cbStruct", wintypes.DWORD),
        ("pcwszFilePath", wintypes.LPCWSTR),
        ("hFile", wintypes.HANDLE),
        ("pgKnownSubject", ctypes.POINTER(GUID)),
    ]


class WINTRUST_CATALOG_INFO(ctypes.Structure):
    _fields_ = [
        ("cbStruct", wintypes.DWORD),
        ("dwCatalogVersion", wintypes.DWORD),
        ("pcwszCatalogFilePath", wintypes.LPCWSTR),
        ("pcwszMemberTag", wintypes.LPCWSTR),
        ("pcwszMemberFilePath", wintypes.LPCWSTR),
        ("hMemberFile", wintypes.HANDLE),
        ("pbCalculatedFileHash", ctypes.POINTER(ctypes.c_ubyte)),
        ("cbCalculatedFileHash", wintypes.DWORD),
        ("pcCatalogContext", ctypes.c_void_p),
        ("hCatAdmin", wintypes.HANDLE),
    ]


class WINTRUST_DATA(ctypes.Structure):
    _fields_ = [
        ("cbStruct", wintypes.DWORD),
        ("pPolicyCallbackData", ctypes.c_void_p),
        ("pSIPClientData", ctypes.c_void_p),
        ("dwUIChoice", wintypes.DWORD),
        ("fdwRevocationChecks", wintypes.DWORD),
        ("dwUnionChoice", wintypes.DWORD),
        ("pUnion", ctypes.c_void_p),
        ("dwStateAction", wintypes.DWORD),
        ("hWVTStateData", wintypes.HANDLE),
        ("pwszURLReference", wintypes.LPCWSTR),
        ("dwProvFlags", wintypes.DWORD),
        ("dwUIContext", wintypes.DWORD),
        ("pSignatureSettings", ctypes.c_void_p),
    ]


class CATALOG_INFO(ctypes.Structure):
    _fields_ = [
        ("cbStruct", wintypes.DWORD),
        ("wszCatalogFile", wintypes.WCHAR * 260),
    ]


class CRYPTOAPI_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


class CRYPT_ALGORITHM_IDENTIFIER(ctypes.Structure):
    _fields_ = [
        ("pszObjId", ctypes.c_char_p),
        ("Parameters", CRYPTOAPI_BLOB),
    ]


class CRYPT_BIT_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
        ("cUnusedBits", wintypes.DWORD),
    ]


class CERT_PUBLIC_KEY_INFO(ctypes.Structure):
    _fields_ = [
        ("Algorithm", CRYPT_ALGORITHM_IDENTIFIER),
        ("PublicKey", CRYPT_BIT_BLOB),
    ]


class CERT_INFO(ctypes.Structure):
    _fields_ = [
        ("dwVersion", wintypes.DWORD),
        ("SerialNumber", CRYPTOAPI_BLOB),
        ("SignatureAlgorithm", CRYPT_ALGORITHM_IDENTIFIER),
        ("Issuer", CRYPTOAPI_BLOB),
        ("NotBefore", wintypes.FILETIME),
        ("NotAfter", wintypes.FILETIME),
        ("Subject", CRYPTOAPI_BLOB),
        ("SubjectPublicKeyInfo", CERT_PUBLIC_KEY_INFO),
        ("IssuerUniqueId", CRYPT_BIT_BLOB),
        ("SubjectUniqueId", CRYPT_BIT_BLOB),
        ("cExtension", wintypes.DWORD),
        ("rgExtension", ctypes.c_void_p),
    ]


_wintrust.WinVerifyTrust.argtypes = [wintypes.HWND, ctypes.POINTER(GUID), ctypes.c_void_p]
_wintrust.WinVerifyTrust.restype = wintypes.LONG

_wintrust.CryptCATAdminAcquireContext.argtypes = [
    ctypes.POINTER(wintypes.HANDLE),
    ctypes.POINTER(GUID),
    wintypes.DWORD,
]
_wintrust.CryptCATAdminAcquireContext.restype = wintypes.BOOL

_wintrust.CryptCATAdminReleaseContext.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_wintrust.CryptCATAdminReleaseContext.restype = wintypes.BOOL

_wintrust.CryptCATAdminCalcHashFromFileHandle.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(wintypes.DWORD),
    ctypes.POINTER(ctypes.c_ubyte),
    wintypes.DWORD,
]
_wintrust.CryptCATAdminCalcHashFromFileHandle.restype = wintypes.BOOL

_wintrust.CryptCATAdminEnumCatalogFromHash.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(ctypes.c_ubyte),
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.HANDLE),
]
_wintrust.CryptCATAdminEnumCatalogFromHash.restype = wintypes.HANDLE

_wintrust.CryptCATCatalogInfoFromContext.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(CATALOG_INFO),
    wintypes.DWORD,
]
_wintrust.CryptCATCatalogInfoFromContext.restype = wintypes.BOOL

_wintrust.CryptCATAdminReleaseCatalogContext.argtypes = [
    wintypes.HANDLE,
    wintypes.HANDLE,
    wintypes.DWORD,
]
_wintrust.CryptCATAdminReleaseCatalogContext.restype = wintypes.BOOL

_kernel32.CreateFileW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
]
_kernel32.CreateFileW.restype = wintypes.HANDLE
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel32.CloseHandle.restype = wintypes.BOOL

_crypt32.CryptQueryObject.argtypes = [
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
    ctypes.POINTER(wintypes.DWORD),
    ctypes.POINTER(wintypes.DWORD),
    ctypes.POINTER(wintypes.HANDLE),
    ctypes.POINTER(wintypes.HANDLE),
    ctypes.POINTER(ctypes.c_void_p),
]
_crypt32.CryptQueryObject.restype = wintypes.BOOL

_crypt32.CryptMsgGetParam.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
    ctypes.POINTER(wintypes.DWORD),
]
_crypt32.CryptMsgGetParam.restype = wintypes.BOOL
_crypt32.CryptMsgClose.argtypes = [wintypes.HANDLE]
_crypt32.CryptMsgClose.restype = wintypes.BOOL

_crypt32.CertFindCertificateInStore.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_crypt32.CertFindCertificateInStore.restype = ctypes.c_void_p

_crypt32.CertGetNameStringW.argtypes = [
    ctypes.c_void_p,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.LPWSTR,
    wintypes.DWORD,
]
_crypt32.CertGetNameStringW.restype = wintypes.DWORD

_crypt32.CertFreeCertificateContext.argtypes = [ctypes.c_void_p]
_crypt32.CertFreeCertificateContext.restype = wintypes.BOOL
_crypt32.CertCloseStore.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_crypt32.CertCloseStore.restype = wintypes.BOOL


@dataclass
class SignatureInfo:
    is_signed: Optional[bool] = None
    signature_status: Optional[str] = None  # trusted|untrusted|expired|unsigned|error
    signer: Optional[str] = None
    signer_is_microsoft: Optional[bool] = None


def _status_string(code: int) -> str:
    if code == _ERROR_SUCCESS:
        return "trusted"
    if code == _CERT_E_EXPIRED:
        return "expired"
    if code == _TRUST_E_NOSIGNATURE:
        return "unsigned"
    if code in (
        _TRUST_E_BAD_DIGEST,
        _TRUST_E_SUBJECT_NOT_TRUSTED,
        _CERT_E_UNTRUSTEDROOT,
        _CERT_E_CHAINING,
        _CRYPT_E_SECURITY_SETTINGS,
    ):
        return "untrusted"
    return "error"


def _verify_embedded(path: str) -> int:
    file_info = WINTRUST_FILE_INFO()
    file_info.cbStruct = ctypes.sizeof(WINTRUST_FILE_INFO)
    file_info.pcwszFilePath = path
    file_info.hFile = None
    file_info.pgKnownSubject = None

    data = WINTRUST_DATA()
    data.cbStruct = ctypes.sizeof(WINTRUST_DATA)
    data.dwUIChoice = _WTD_UI_NONE
    data.fdwRevocationChecks = _WTD_REVOKE_NONE
    data.dwUnionChoice = _WTD_CHOICE_FILE
    data.pUnion = ctypes.cast(ctypes.byref(file_info), ctypes.c_void_p)
    data.dwStateAction = _WTD_STATEACTION_VERIFY
    # No revocation checking, and cache-only retrieval so verification never
    # touches the network (WTD_CACHE_ONLY_URL_RETRIEVAL is required alongside
    # WTD_REVOKE_NONE to guarantee that).
    data.dwProvFlags = _WTD_REVOCATION_CHECK_NONE | _WTD_CACHE_ONLY_URL_RETRIEVAL

    result = _u32(_wintrust.WinVerifyTrust(None, ctypes.byref(_WVT_GUID), ctypes.byref(data)))

    data.dwStateAction = _WTD_STATEACTION_CLOSE
    _wintrust.WinVerifyTrust(None, ctypes.byref(_WVT_GUID), ctypes.byref(data))
    return result


def _verify_catalog(path: str) -> "tuple[Optional[int], Optional[str]]":
    """Verify ``path`` against the system catalogs.

    Returns ``(status_code, catalog_path)``. ``status_code`` is ``None`` when the
    file is not catalog-signed; ``catalog_path`` points at the backing ``.cat``.
    """
    h_admin = wintypes.HANDLE()
    if not _wintrust.CryptCATAdminAcquireContext(ctypes.byref(h_admin), None, 0):
        return None, None
    try:
        h_file = _kernel32.CreateFileW(
            path, _GENERIC_READ, _FILE_SHARE_READ, None, _OPEN_EXISTING, 0, None
        )
        if not h_file or h_file == _INVALID_HANDLE_VALUE:
            return None, None
        try:
            size = wintypes.DWORD(0)
            _wintrust.CryptCATAdminCalcHashFromFileHandle(h_file, ctypes.byref(size), None, 0)
            if size.value == 0:
                return None, None
            hash_buf = (ctypes.c_ubyte * size.value)()
            if not _wintrust.CryptCATAdminCalcHashFromFileHandle(
                h_file, ctypes.byref(size), hash_buf, 0
            ):
                return None, None

            h_cat = _wintrust.CryptCATAdminEnumCatalogFromHash(
                h_admin, hash_buf, size.value, 0, None
            )
            if not h_cat:
                return None, None
            try:
                cat_info = CATALOG_INFO()
                cat_info.cbStruct = ctypes.sizeof(CATALOG_INFO)
                if not _wintrust.CryptCATCatalogInfoFromContext(
                    h_cat, ctypes.byref(cat_info), 0
                ):
                    return None, None
                catalog_path = cat_info.wszCatalogFile
                member_tag = "".join("%02X" % b for b in hash_buf)

                wci = WINTRUST_CATALOG_INFO()
                wci.cbStruct = ctypes.sizeof(WINTRUST_CATALOG_INFO)
                wci.dwCatalogVersion = 0
                wci.pcwszCatalogFilePath = catalog_path
                wci.pcwszMemberTag = member_tag
                wci.pcwszMemberFilePath = path
                wci.hMemberFile = h_file
                wci.pbCalculatedFileHash = hash_buf
                wci.cbCalculatedFileHash = size.value
                wci.pcCatalogContext = None
                wci.hCatAdmin = h_admin

                data = WINTRUST_DATA()
                data.cbStruct = ctypes.sizeof(WINTRUST_DATA)
                data.dwUIChoice = _WTD_UI_NONE
                data.fdwRevocationChecks = _WTD_REVOKE_NONE
                data.dwUnionChoice = _WTD_CHOICE_CATALOG
                data.pUnion = ctypes.cast(ctypes.byref(wci), ctypes.c_void_p)
                data.dwStateAction = _WTD_STATEACTION_VERIFY
                data.dwProvFlags = (
                    _WTD_REVOCATION_CHECK_NONE | _WTD_CACHE_ONLY_URL_RETRIEVAL
                )

                result = _u32(
                    _wintrust.WinVerifyTrust(None, ctypes.byref(_WVT_GUID), ctypes.byref(data))
                )
                data.dwStateAction = _WTD_STATEACTION_CLOSE
                _wintrust.WinVerifyTrust(None, ctypes.byref(_WVT_GUID), ctypes.byref(data))
                return result, catalog_path
            finally:
                _wintrust.CryptCATAdminReleaseCatalogContext(h_admin, h_cat, 0)
        finally:
            _kernel32.CloseHandle(h_file)
    finally:
        _wintrust.CryptCATAdminReleaseContext(h_admin, 0)


def _signer_name(file_path: str) -> Optional[str]:
    """Extract the signer subject CN from an embedded PKCS#7 (file or .cat)."""
    h_store = wintypes.HANDLE()
    h_msg = wintypes.HANDLE()
    if not _crypt32.CryptQueryObject(
        _CERT_QUERY_OBJECT_FILE,
        ctypes.c_wchar_p(file_path),
        _CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED_EMBED,
        _CERT_QUERY_FORMAT_FLAG_BINARY,
        0,
        None,
        None,
        None,
        ctypes.byref(h_store),
        ctypes.byref(h_msg),
        None,
    ):
        return None
    try:
        size = wintypes.DWORD(0)
        if not _crypt32.CryptMsgGetParam(
            h_msg, _CMSG_SIGNER_CERT_INFO_PARAM, 0, None, ctypes.byref(size)
        ) or size.value == 0:
            return None
        buf = (ctypes.c_ubyte * size.value)()
        if not _crypt32.CryptMsgGetParam(
            h_msg, _CMSG_SIGNER_CERT_INFO_PARAM, 0, buf, ctypes.byref(size)
        ):
            return None

        cert_ctx = _crypt32.CertFindCertificateInStore(
            h_store,
            _X509_ASN_ENCODING | _PKCS_7_ASN_ENCODING,
            0,
            _CERT_FIND_SUBJECT_CERT,
            ctypes.cast(buf, ctypes.c_void_p),
            None,
        )
        if not cert_ctx:
            return None
        try:
            needed = _crypt32.CertGetNameStringW(
                cert_ctx, _CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, None, None, 0
            )
            if needed <= 1:
                return None
            name_buf = ctypes.create_unicode_buffer(needed)
            _crypt32.CertGetNameStringW(
                cert_ctx, _CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, None, name_buf, needed
            )
            value = name_buf.value.strip()
            return value or None
        finally:
            _crypt32.CertFreeCertificateContext(cert_ctx)
    finally:
        if h_store:
            _crypt32.CertCloseStore(h_store, 0)
        if h_msg:
            _crypt32.CryptMsgClose(h_msg)


def _is_microsoft(signer: Optional[str]) -> Optional[bool]:
    if not signer:
        return None
    return "microsoft" in signer.lower()


def signature(path: str) -> SignatureInfo:
    """Verify the Authenticode signature of ``path``. Never raises."""
    info = SignatureInfo()
    if not path:
        info.signature_status = "error"
        return info
    try:
        code = _verify_embedded(path)
        signer_source: Optional[str] = path
        if code == _TRUST_E_NOSIGNATURE:
            cat_code, catalog_path = _verify_catalog(path)
            if cat_code is None:
                info.is_signed = False
                info.signature_status = "unsigned"
                return info
            code = cat_code
            signer_source = catalog_path

        info.signature_status = _status_string(code)
        # ``is_signed`` means a signature is present, regardless of trust. Expired
        # or untrusted-but-genuinely-signed binaries are still signed; the trust
        # outcome is carried separately by ``signature_status``.
        info.is_signed = info.signature_status not in ("unsigned", "error")
        try:
            info.signer = _signer_name(signer_source) if signer_source else None
        except (OSError, ValueError, ctypes.ArgumentError):
            info.signer = None
        info.signer_is_microsoft = _is_microsoft(info.signer)
    except (OSError, ValueError, ctypes.ArgumentError) as exc:
        info.signature_status = "error"
        _debug.log("signature", exc)
    return info
