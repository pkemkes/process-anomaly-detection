# process_stream

A lightweight Python program that emits a real-time **NDJSON stream of every process
start** on Windows, enriched with metadata useful for process-anomaly detection. On
startup it first emits a snapshot of all currently-running processes, then switches to
streaming new starts in real time.

This module is the **data-collection layer** of the
[process-anomalies](../README.md) project: it produces the labelled-by-context input
stream that downstream feature engineering and anomaly-detection models consume.

## How it works

Two layers are combined:

| Layer | Source | Provides |
|-------|--------|----------|
| Trigger + core metadata | ETW `Microsoft-Windows-Kernel-Process` provider (event ids 1 & 2) | pid, ppid, process sequence number, session id, image, create/exit time, exit code, integrity level, elevation |
| Enrichment | `psutil` | command line, exe path, user, cwd, parent image / command line / user |
| File identity | `ctypes` (SHA-256, WinVerifyTrust, version.dll) | image hash & size, Authenticode signature, PE version metadata |
| Account identity | `pywin32` token APIs | user SID, logon type, integrity, elevation |

ETW guarantees every start (event id 1) and stop (event id 2) is seen in real time
(even millisecond-lived processes); `psutil`, `ctypes`, and `pywin32` fill the
model-facing fields on a best-effort basis. All enrichment runs on the consumer
thread, never in the ETW callback, and file facts are cached per
`(path, size, mtime)` so each binary is touched on disk once per version.

## Requirements

- Windows
- Python 3.9+
- **Administrator** privileges (ETW real-time sessions require elevation)

## Install

From the repository root, install this subproject as an editable package (this
pulls in `pywintrace`, `psutil`, `pywin32`):

```powershell
pip install -e process_stream
```

## Run

From an **elevated** terminal, anywhere once installed:

```powershell
python -m process_stream
```

### Options

| Flag | Description |
|------|-------------|
| `--no-backfill` | Skip the existing-process snapshot; stream new starts only. |
| `--pretty` | Emit indented JSON (debugging aid). |
| `--queue-size N` | ETW event buffer size (default 10000). |
| `--drop-pseudo` | Omit pseudo / non-launch records (System Idle, Registry, ...) entirely. By default they are kept and flagged with `is_pseudo: true`. |
| `--debug-raw` | Dump each raw ETW event dict to stderr (for verifying field names). |
| `--debug` | Log best-effort errors (normally swallowed) to stderr for diagnosis. |

### Example: pipe into a consumer

```powershell
python -m process_stream | python -c "import sys,json; [print(json.loads(l)['image']) for l in sys.stdin]"
```

### Example: write the output to a file

```powershell
# Overwrite each run; stderr (warnings) still shows in the terminal.
python -m process_stream > processes.ndjson
```

## Output schema

One JSON object per line. `event` is `"process_start"` (or `"process_stop"` for the
lifetime/exit-code event correlated by `process_seq`):

```json
{
  "event": "process_start",
  "source": "new",
  "schema_version": "1.0.0",
  "is_pseudo": false,
  "timestamp": "2026-06-14T10:32:11.482Z",
  "pid": 12044,
  "ppid": 8123,
  "process_seq": 845123,
  "image": "C:\\Windows\\System32\\cmd.exe",
  "command_line": "cmd.exe /c whoami",
  "command_line_normalized": "c:\\windows\\system32\\cmd.exe /c whoami",
  "user": "MACHINE\\pkemkes",
  "cwd": "C:\\Users\\pkemkes",
  "session_id": 1,
  "integrity_level": "Medium",
  "is_elevated": false,
  "parent_image": "C:\\Windows\\explorer.exe",
  "parent_command_line": "C:\\Windows\\explorer.exe",
  "parent_user": "MACHINE\\pkemkes",
  "create_time": "2026-06-14T10:32:11.480Z",
  "enriched": true,
  "image_hash": "ab12...",
  "image_size": 285184,
  "is_signed": true,
  "signature_status": "trusted",
  "signer": "Microsoft Windows",
  "signer_is_microsoft": true,
  "original_file_name": "Cmd.Exe",
  "company_name": "Microsoft Corporation",
  "product_name": "Microsoft\u00ae Windows\u00ae Operating System",
  "file_description": "Windows Command Processor",
  "file_version": "10.0.19041.1",
  "name_mismatch": false,
  "user_sid": "S-1-5-21-...",
  "logon_type": "Interactive",
  "path_bucket": "System32",
  "image_name": "cmd.exe",
  "hour_of_day": 10,
  "day_of_week": 5
}
```

- `source`: `"existing"` for the startup snapshot, `"new"` for live events.
- `timestamp`: stream-emission/observation time, independent of `create_time`. For
  a live start it is roughly the create time; for a backfill record it is the scan
  time of an already-running process; for `process_stop` it is when the exit was seen.
- `schema_version`: stamped on every record so datasets can be bucketed / migrated.
- `is_signed`: `true` when an Authenticode signature is present (embedded or catalog),
  regardless of trust. The trust outcome is carried by `signature_status`
  (`trusted` / `expired` / `untrusted` / `unsigned` / `error`).
- `is_pseudo`: `true` for non-launch pseudo-processes (System Idle, Registry, ...).
- `process_stop` records carry `exit_code` and `lifetime_ms` (and `null` for the
  start-only fields). `exit_code` is a signed 32-bit integer (e.g. `-1` rather than
  `4294967295`).
- Unavailable fields are `null` rather than omitted, so the shape is stable.

## Limitations

- **Command line / user** may be `null` for ultra-short-lived processes that exit before
  `psutil` can read them; ETW core metadata is still emitted (`enriched: false`).
- **PID reuse**: mitigated by emitting `process_seq` (unique per boot session).
- **Backfill/live overlap**: a process starting between session start and snapshot may
  appear twice; dedupe on `pid` + `process_seq`.
- `process_seq` is an ETW-only counter and is structurally `null` for backfill
  (`source: "existing"`) records; integrity, elevation, and session id are filled from the
  process token so backfill and live records otherwise match.
- File identity and signature fields are `null` when the image is locked, deleted, or
  access-denied at observation time.
