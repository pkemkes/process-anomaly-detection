# process_monitor

A live, full-screen terminal monitor for the process-anomaly pipeline. It reads
the **scored** NDJSON produced by `model score`, takes over the terminal, and
shows a continuously refreshing, **non-scrolling** table of processes ranked by
anomaly score -- most suspicious at the top.

The display is rendered with [Rich](https://github.com/Textualize/rich), which
owns the alternate screen, colours, elastic columns and flicker-free repaint
(virtual-terminal mode is enabled automatically on the Windows console).

## Usage

Pipe the live collector through the scorer into the monitor:

```powershell
python -m process_stream | python -m model score --model model.joblib | python -m process_monitor
```

Press `Ctrl+C` to quit; your terminal scrollback is restored on exit.

### Hotkeys

| Key | Action |
|-----|--------|
| `s` | Sort by anomaly score (most suspicious first). |
| `t` | Sort by timestamp (most recent first). |
| `Space` / `Tab` | Toggle between sort modes. |

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--refresh` | `0.5` | Screen repaint interval in seconds. |

## Display

- **One line per process**, sorted by `anomaly_score` descending by default
  (press `t` to sort by timestamp instead).
- Rows are coloured by `anomaly_rank_hint`: **red** = high, **yellow** = medium,
  **green** = low.
- Columns: score, PID, image name, user, and the top contributing field.
- The number of rows grows with the terminal height; columns are clipped to the
  width so a line never wraps.
- `process_stop` records evict their process; pseudo / unscored records are
  ignored.

## Layout

```
process_monitor/
    process_monitor/
        __main__.py   # CLI entry point (python -m process_monitor)
        monitor.py    # run-loop: stdin reader thread + Rich Live repaint
        store.py      # live, ranked table of scored processes
        render.py     # pure frame formatter -> Rich renderable
        terminal.py   # non-blocking console hotkey reader
```
