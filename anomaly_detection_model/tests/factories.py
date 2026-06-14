"""Test data factories: synthetic process-start records and a tiny baseline."""

from __future__ import annotations

from typing import Dict, List

# Schema version the synthetic records claim. Kept in step with the collector's
# stream schema so the scorer's schema guard is exercised, but defined locally so
# the model package has no import dependency on any collector.
SCHEMA_VERSION = "1.0.0"


def make_record(**overrides) -> Dict[str, object]:
    """A plausible, "normal" signed-System32 process_start record.

    Pass keyword overrides to mutate individual fields for a test case.
    """
    record: Dict[str, object] = {
        "pid": 1000,
        "source": "new",
        "timestamp": "2026-06-14T10:00:00.000Z",
        "event": "process_start",
        "schema_version": SCHEMA_VERSION,
        "is_pseudo": False,
        "ppid": 800,
        "image": "C:\\Windows\\System32\\cmd.exe",
        "command_line": "cmd.exe /c whoami",
        "command_line_normalized": "c:\\windows\\system32\\cmd.exe /c whoami",
        "user": "MACHINE\\alice",
        "session_id": 1,
        "integrity_level": "Medium",
        "is_elevated": False,
        "parent_image": "C:\\Windows\\explorer.exe",
        "create_time": "2026-06-14T10:00:00.000Z",
        "enriched": True,
        "is_signed": True,
        "signature_status": "trusted",
        "signer": "Microsoft Windows",
        "signer_is_microsoft": True,
        "original_file_name": "Cmd.Exe",
        "company_name": "Microsoft Corporation",
        "name_mismatch": False,
        "user_sid": "S-1-5-21-1",
        "logon_type": "Interactive",
        "path_bucket": "System32",
        "image_name": "cmd.exe",
    }
    record.update(overrides)
    return record


def baseline_records(n: int = 60) -> List[Dict[str, object]]:
    """A small, repetitive "normal" baseline of common signed System32 starts."""
    images = [
        ("cmd.exe", "C:\\Windows\\System32\\cmd.exe", "explorer.exe"),
        ("svchost.exe", "C:\\Windows\\System32\\svchost.exe", "services.exe"),
        ("explorer.exe", "C:\\Windows\\explorer.exe", "userinit.exe"),
        ("notepad.exe", "C:\\Windows\\System32\\notepad.exe", "explorer.exe"),
    ]
    records: List[Dict[str, object]] = []
    for i in range(n):
        name, image, parent = images[i % len(images)]
        records.append(
            make_record(
                pid=1000 + i,
                image=image,
                image_name=name,
                parent_image=f"C:\\Windows\\System32\\{parent}",
                command_line_normalized=f"{image.lower()} /c task{i % 3}",
            )
        )
    return records


def malicious_record() -> Dict[str, object]:
    """A clearly-suspicious LOLBin start that should score in the top range."""
    return make_record(
        pid=66613,
        image="C:\\Users\\alice\\AppData\\Local\\Temp\\powershell.exe",
        image_name="powershell.exe",
        parent_image="C:\\Program Files\\Microsoft Office\\winword.exe",
        path_bucket="Temp",
        command_line_normalized=(
            "c:\\users\\alice\\appdata\\local\\temp\\powershell.exe "
            "-nop -w hidden -enc aQBlAHgAKABuAGUAdwAtAG8AYgBqAGUAYwB0A)"
        ),
        is_signed=False,
        signature_status="unsigned",
        signer=None,
        signer_is_microsoft=False,
        company_name=None,
        original_file_name=None,
        integrity_level="High",
        is_elevated=True,
    )
