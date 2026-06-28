"""Test data factories: synthetic process-start records and a tiny baseline."""

from __future__ import annotations

import datetime
from typing import Dict, List

# Schema version the synthetic records claim. Kept in step with the collector's
# stream schema so the scorer's schema guard is exercised, but defined locally so
# the model package has no import dependency on any collector.
SCHEMA_VERSION = "1.0.0"

# Anchor for synthetic timestamps; records are spread forward from here so the
# saturating window-count and temporal heads have real time structure to learn.
_BASE_TIME = datetime.datetime(2026, 6, 1, 8, 0, 0, tzinfo=datetime.timezone.utc)


def _iso(when: datetime.datetime) -> str:
    """Render a UTC datetime as the collector's ISO-8601 ``...Z`` form."""
    return when.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def make_record(**overrides) -> Dict[str, object]:
    """A plausible, "normal" signed-System32 process_start record.

    Pass keyword overrides to mutate individual fields for a test case.
    """
    record: Dict[str, object] = {
        "pid": 1000,
        "source": "new",
        "timestamp": "2026-06-15T10:00:00.000Z",
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
        "create_time": "2026-06-15T10:00:00.000Z",
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


# Common signed images with a typical run hour, forming the bulk of the baseline.
# Each runs once per day at its own hour, so identities recur across many daily
# windows (saturating to "regular") with a tight per-image time-of-day profile.
_COMMON = [
    ("cmd.exe", "C:\\Windows\\System32\\cmd.exe", "explorer.exe", 10),
    ("svchost.exe", "C:\\Windows\\System32\\svchost.exe", "services.exe", 3),
    ("explorer.exe", "C:\\Windows\\explorer.exe", "userinit.exe", 8),
    ("notepad.exe", "C:\\Windows\\System32\\notepad.exe", "explorer.exe", 14),
]
_USERS = ["MACHINE\\alice", "MACHINE\\bob"]


def baseline_records(n: int = 60) -> List[Dict[str, object]]:
    """A small but non-degenerate "normal" baseline of signed System32 starts.

    The dominant pattern is a handful of common images recurring daily at a
    consistent hour. To mirror real captures (and give each scoring head genuine
    spread to calibrate against) it also sprinkles in two kinds of variation: a
    second user, varied command-line lengths, and the occasional novel one-off
    administrative tool that recurs in only a single window.
    """
    start = _BASE_TIME.replace(hour=10)
    records: List[Dict[str, object]] = []
    for i in range(n):
        day = i // len(_COMMON)
        user = _USERS[i % len(_USERS)]
        if i % 9 == 8:
            # A novel-but-benign signed tool seen once: builds an identity-surprise
            # tail so the identity head is not perfectly constant.
            name = f"tool{i}.exe"
            when = start.replace(hour=12) + datetime.timedelta(days=day)
            stamp = _iso(when)
            records.append(
                make_record(
                    pid=2000 + i,
                    image=f"C:\\Windows\\System32\\{name}",
                    image_name=name,
                    parent_image="C:\\Windows\\System32\\cmd.exe",
                    original_file_name=name.title(),
                    command_line_normalized=f"c:\\windows\\system32\\{name} --run {i}",
                    user=user,
                    create_time=stamp,
                    timestamp=stamp,
                )
            )
            continue
        name, image, parent, hour = _COMMON[i % len(_COMMON)]
        when = start.replace(hour=hour) + datetime.timedelta(days=day)
        stamp = _iso(when)
        records.append(
            make_record(
                pid=1000 + i,
                image=image,
                image_name=name,
                parent_image=f"C:\\Windows\\System32\\{parent}",
                user=user,
                create_time=stamp,
                timestamp=stamp,
            )
        )
    return records


def fixed_hour_records(
    image_name: str,
    n: int,
    *,
    hour: int = 2,
    image: str = "C:\\Windows\\System32\\backup.exe",
) -> List[Dict[str, object]]:
    """``n`` records for one image that always runs at the same UTC ``hour``.

    Stamped on consecutive days so the runs land in many distinct recurrence
    windows but a single, tight per-image hour profile.
    """
    records: List[Dict[str, object]] = []
    start = _BASE_TIME.replace(hour=hour)
    for i in range(n):
        stamp = _iso(start + datetime.timedelta(days=i))
        records.append(
            make_record(
                pid=5000 + i,
                image=image,
                image_name=image_name,
                create_time=stamp,
                timestamp=stamp,
            )
        )
    return records


def at_hour(record: Dict[str, object], hour: int, *, day: int = 0) -> Dict[str, object]:
    """Return a copy of ``record`` re-stamped to a specific UTC ``hour``/``day``."""
    when = _BASE_TIME.replace(hour=hour) + datetime.timedelta(days=day)
    stamp = _iso(when)
    return make_record(**{**record, "create_time": stamp, "timestamp": stamp})


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
