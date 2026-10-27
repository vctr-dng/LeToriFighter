# M0 report: foundations and throughput

Run on 2026-10-27 against the reworked `ToriLLE/` (Python 3.14.6, uv), Steam-native Toribash at `~/.local/share/Steam/steamapps/common/Toribash/`, X display `:0`.

## Status against the M0 exit criteria

| Criterion | Status |
|---|---|
| `pytest` passes | Done: 32 passed, 3 deselected (native). |
| `pytest -m native` passes | Done on the real display `:0`: 3 passed in 25 s. The game directory was left clean (no controller script, `profile.tbs` restored). |
| Native tests pass under Xvfb | **Not met.** See "Xvfb finding". |
| Written throughput number | Done, measured on `:0` (table below). |
| Native tests pass under Xvfb | Dropped by decision: runs stay on `:0` (option 1 below). |
| Pin the Toribash build | Done: Steam app 248570, build id 20840792 (`appmanifest_248570.acf`, last updated by Steam 2026-10-27). |

## Throughput (random actions, `Toribash-DestroyUke-v0`, `matchframes=1000`, 30 s per cell)

Raw output: `LeToriFighter/results/m0_throughput_display0.json`. Script: `LeToriFighter/scripts/m0_throughput.py` (run with `uv run python scripts/m0_throughput.py` from `LeToriFighter/`).

| `draw_game` | `turnframes` | turns/s | physics frames/s | mean reset | launch |
|---|---|---|---|---|---|
| False | 1 | 35.2 | 35.2 | 0.27 s | 5.4 s |
| False | 5 | 17.0 | 85.0 | 0.15 s | 6.0 s |
| True | 1 | 17.8 | 17.8 | 0.18 s | 6.7 s |
| True | 5 | 8.6 | 43.0 | 0.10 s | 6.0 s |

Single run per cell, one machine, one game instance; treat as order-of-magnitude.

Reading the numbers (a two-point linear fit per row, my inference, not a profile):

- With `draw_game=False` each turn costs about 21 ms fixed plus about 7.6 ms per physics frame. With `draw_game=True` it is about 41 ms plus about 15 ms per frame.
- The fixed per-turn cost is large. The native bridge polls every 10 ms and the Lua side rewrites files each turn, so lowering the poll interval and fixing the handshake is a likely win. Check in M1.
- Drawing roughly halves throughput. Screen capture needs the window drawn, so video collection runs at the "True" rows.

## Dataset sizing (feeds M5)

One instance, drawn (needed for video), `turnframes=5`, 1000-frame matches (200 turns per episode):

- 8.6 turns/s is about 31k turns per hour, about 155 episodes per hour.
- 100k turns (about 500 episodes) takes about 3.2 hours of pure simulation; 1M turns takes about 32 hours.
- State-only data (`draw_game=False`) is about 2x faster.

Shorter matches or a larger `turnframes` change the turn/episode counts but not frames per second.

## Xvfb finding

On Linux, ToriLLE always launches the game with `steam -applaunch 248570` (`torille/torille.py:510-528`). The game inherits the display of the Steam client that is already running (here `:0`), so setting `DISPLAY=:10` in the Python process has no effect on where the game renders.

Launching `./toribash_steam` directly with `DISPLAY=:10` does not work either: the binary re-executes `steam.sh`, which hands off to the running Steam client, and no game appears on `:10`. I stopped the test after 20 s; no process was left on `:10` and the Steam client on `:0` was untouched.

So running under Xvfb needs a Steam client that itself runs on the virtual display. That conflicts with the Steam client in the desktop session, because Steam allows one client per user. Options:

1. Keep using `:0` for now. Capture the window with `x11grab` using its geometry. Simple, but the game window sits on the desktop.
2. Run a second Steam client under Xvfb with a separate `HOME` and a separate login (Steam Guard prompt, a Toribash licence on that account).
3. Close the desktop Steam and run the user's own Steam under Xvfb. Disruptive.

Also unverified: whether OpenGL works at all under Xvfb with this NVIDIA driver (the FAQ warns it does not; `glxinfo` is not installed here).

## Other M1-relevant facts found

- The control bridge lives inside the game directory (`<game dir>/.torille/`) and there is a `.launchlock` there, so one install means one instance. A copy of the game directory is 436 MB, so per-instance copies are cheap in disk terms.
- ffmpeg, Xvfb and uv are installed; `glxinfo`/`import` are not.
- `Toribash-DestroyUke-v0` emits a Gymnasium deprecation warning (use `v2`).
