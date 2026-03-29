#!/usr/bin/env python3
"""
ripvec Metal GEMM benchmark harness.

Handles thermals, timing, and result tracking for comparing Metal MPS vs
compute-shader GEMM paths and CPU baseline.

Run from repo root:
    uv run scripts/bench/bench.py [options]

Options:
    --configs mps compute cpu   Configs to run (default: mps compute cpu)
    --layers N [N ...]          Early-exit layer counts (default: 22)
    --corpus PATH               Corpus directory (default: tests/corpus/code/flask)
    --validate                  Validate correctness across configs instead of benchmarking
    --thermals                  Print thermal state and exit
    --no-build                  Skip cargo build --release
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional rich import — fall back to plain print
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table

    _console = Console()

    def _print(*args: object, **kwargs: object) -> None:
        _console.print(*args, **kwargs)

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

    def _print(*args: object, **kwargs: object) -> None:  # type: ignore[misc]
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Repo root resolution
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BINARY = REPO_ROOT / "target" / "release" / "ripvec"
DEFAULT_CORPUS = REPO_ROOT / "tests" / "corpus" / "code" / "flask"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Thermal helpers
# ---------------------------------------------------------------------------

# Intel Macs expose thermal pressure via xcpm sysctl.
# Apple Silicon does not — we fall back to pmset and powermetrics instead.
_XCPM_KEYS = {
    "machdep.xcpm.cpu_thermal_level": "cpu",
    "machdep.xcpm.gpu_thermal_level": "gpu",
}

# pmset -g therm output patterns (Apple Silicon + Intel):
#   CPU_Speed_Limit         = 100
#   CPU_Scheduler_Limit     = 100
#   GPU_Available_Power     = 100
_PMSET_PATTERN = re.compile(r"(CPU_Speed_Limit|CPU_Scheduler_Limit)\s*=\s*(\d+)")


def _read_thermal_xcpm() -> dict[str, int]:
    """Intel Macs: read via sysctl machdep.xcpm.*  Returns {} if unavailable."""
    result: dict[str, int] = {}
    for key, label in _XCPM_KEYS.items():
        try:
            out = subprocess.run(
                ["sysctl", "-n", key],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if out.returncode == 0:
                val = out.stdout.strip()
                if val.isdigit():
                    result[label] = int(val)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            break
    return result


def _read_thermal_pmset() -> dict[str, int]:
    """Apple Silicon / Intel fallback: read via pmset -g therm.

    pmset -g therm exits quickly (unlike pmset -g thermlog which blocks).
    CPU_Speed_Limit = 100 means no throttling; lower values indicate throttling.
    We invert to a 0-100 pressure scale (0=no pressure, 100=fully throttled).
    """
    try:
        out = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode != 0:
            return {}
        # If no warnings recorded, pressure = 0
        if "No thermal warning" in out.stdout:
            return {"cpu": 0}
        result: dict[str, int] = {}
        for m in _PMSET_PATTERN.finditer(out.stdout):
            key, val = m.group(1), int(m.group(2))
            # Invert: limit=100 → pressure=0, limit=50 → pressure=50
            pressure = max(0, 100 - val)
            if "CPU_Speed_Limit" in key:
                result["cpu"] = pressure
            else:
                result.setdefault("cpu", pressure)
        return result
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}


def read_thermal() -> dict[str, int]:
    """Read thermal pressure levels (no sudo required).

    Strategy (in order):
    1. sysctl machdep.xcpm.* — Intel Macs only, fast
    2. pmset -g therm        — Apple Silicon + Intel, fast (not thermlog)

    Returns a dict with keys like 'cpu'/'gpu', values 0-100 (pressure scale).
    Returns empty dict on non-Apple platforms.
    """
    # Try Intel xcpm path first
    result = _read_thermal_xcpm()
    if result:
        return result
    # Apple Silicon (or Intel where xcpm is unavailable): use pmset
    return _read_thermal_pmset()


def format_thermal(t: dict[str, int]) -> str:
    """Return a compact string like 'cpu=24 gpu=18' or 'n/a'."""
    if not t:
        return "n/a"
    parts = [f"{k}={v}" for k, v in sorted(t.items())]
    return " ".join(parts)


def thermal_level(t: dict[str, int]) -> int:
    """Return the highest thermal value across all sensors."""
    return max(t.values(), default=0)


def wait_for_cool(threshold_warn: int = 50, threshold_ok: int = 30) -> None:
    """Block until all sensors are below threshold_ok, warning if above threshold_warn."""
    t = read_thermal()
    level = thermal_level(t)
    if level <= threshold_warn:
        return
    _print(
        f"[yellow]Thermal too high ({format_thermal(t)}), waiting for cool-down (<{threshold_ok})...[/yellow]"
        if HAS_RICH
        else f"Thermal too high ({format_thermal(t)}), waiting for cool-down (<{threshold_ok})..."
    )
    while level > threshold_ok:
        time.sleep(10)
        t = read_thermal()
        level = thermal_level(t)
        _print(f"  still warm: {format_thermal(t)}")
    _print(f"  cool enough: {format_thermal(t)}")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def cargo_build() -> bool:
    """Run cargo build --release from repo root. Returns True on success."""
    _print("Building ripvec (cargo build --release)...")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        _print(
            "[red]Build failed — aborting.[/red]"
            if HAS_RICH
            else "Build failed — aborting."
        )
        return False
    _print("Build OK.")
    return True


# ---------------------------------------------------------------------------
# Config expansion
# ---------------------------------------------------------------------------


def expand_configs(
    configs: list[str], layers: list[int]
) -> list[tuple[str, dict[str, object]]]:
    """Return list of (label, spec) pairs.

    spec keys:
      env     -- extra environment variables dict
      args    -- extra CLI args list
      layers  -- layer count (int or None)
    """
    result = []
    for cfg in configs:
        for L in layers:
            label = f"{cfg}-{L}L"
            env: dict[str, str] = {}
            extra_args: list[str] = []

            if cfg == "compute":
                env["RIPVEC_NO_MPS"] = "1"
            elif cfg == "cpu":
                extra_args += ["--device", "cpu"]
            # mps: default, no extra env/args

            extra_args += ["--layers", str(L)]
            result.append(
                (label, {"env": env, "args": extra_args, "layers": L, "cfg": cfg})
            )
    return result


# ---------------------------------------------------------------------------
# Throughput parsing
# ---------------------------------------------------------------------------

_THROUGHPUT_RE = re.compile(r"done in [\d.]+s \(([\d.]+)/s\)")


def parse_throughput(output: str) -> float | None:
    """Extract chunks/s from 'embed: N/N done in Xs (Y/s)' line."""
    m = _THROUGHPUT_RE.search(output)
    if m:
        return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Thermal ticker thread
# ---------------------------------------------------------------------------


class ThermalTicker:
    """Background thread that prints thermal readings while a run is active."""

    def __init__(self, label: str, interval: float = 5.0) -> None:
        self.label = label
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start_time = time.monotonic()
        self.readings: list[dict[str, int]] = []

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval + 1)

    def _run(self) -> None:
        # First tick after one interval
        while not self._stop.wait(self.interval):
            elapsed = time.monotonic() - self._start_time
            t = read_thermal()
            self.readings.append(t)
            print(
                f"  [{self.label}] {elapsed:.0f}s thermal={format_thermal(t)}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Single-run executor
# ---------------------------------------------------------------------------


def run_benchmark(
    label: str,
    spec: dict[str, object],
    corpus: Path,
    query: str = "session",
) -> dict[str, object]:
    """Run ripvec, return result dict."""
    env = {**os.environ, **spec["env"]}  # type: ignore[arg-type]
    cmd = [
        str(BINARY),
        query,
        "-n",
        "1",
        str(corpus),
        "--profile",
        *spec["args"],  # type: ignore[arg-type]
    ]

    thermal_before = read_thermal()
    _print(
        f"\n[bold]{label}[/bold] thermal_before={format_thermal(thermal_before)}"
        if HAS_RICH
        else f"\n{label}  thermal_before={format_thermal(thermal_before)}"
    )
    _print(f"  cmd: {' '.join(cmd)}")

    ticker = ThermalTicker(label)
    ticker.start()
    t0 = time.monotonic()

    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,  # 5 min max per run
    )
    wall_time = time.monotonic() - t0
    ticker.stop()

    thermal_after = read_thermal()
    output = proc.stdout or ""

    throughput = parse_throughput(output)

    # Print the last few lines so the throughput line is visible
    lines = [l for l in output.splitlines() if l.strip()]
    for line in lines[-5:]:
        print(f"  {line}")

    result: dict[str, object] = {
        "label": label,
        "cfg": spec["cfg"],
        "layers": spec["layers"],
        "throughput": throughput,
        "wall_time": round(wall_time, 2),
        "thermal_before": thermal_before,
        "thermal_after": thermal_after,
        "returncode": proc.returncode,
    }

    status = f"{throughput:.1f}/s" if throughput is not None else "PARSE_ERROR"
    _print(
        f"  done: {status}  wall={wall_time:.1f}s  thermal_after={format_thermal(thermal_after)}"
    )
    return result


# ---------------------------------------------------------------------------
# Correctness validation
# ---------------------------------------------------------------------------


def run_validate(
    label: str,
    spec: dict[str, object],
    corpus: Path,
    query: str = "session",
) -> list[str]:
    """Run with --format json, return list of top-3 file paths."""
    env = {**os.environ, **spec["env"]}  # type: ignore[arg-type]
    cmd = [
        str(BINARY),
        query,
        "-n",
        "3",
        str(corpus),
        "--format",
        "json",
        *spec["args"],  # type: ignore[arg-type]
    ]

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    paths: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            # Support both {path: ...} and {file: ...} result shapes
            p = obj.get("path") or obj.get("file") or ""
            if p:
                paths.append(p)
        except json.JSONDecodeError:
            continue
    return paths[:3]


# ---------------------------------------------------------------------------
# Output table
# ---------------------------------------------------------------------------


def print_table(results: list[dict[str, object]]) -> None:
    """Print formatted results table."""
    # Find mps baseline throughput (first mps result)
    mps_tp: float | None = None
    for r in results:
        if r["cfg"] == "mps" and r["throughput"] is not None:
            mps_tp = float(r["throughput"])  # type: ignore[arg-type]
            break

    header = f"\n{'=' * 72}"
    print(header)
    print("  ripvec benchmark results")
    print(f"{'=' * 72}")
    fmt = "{:<18} {:>10} {:>14} {:>10} {:>8}"
    print(fmt.format("Config", "Throughput", "Thermal", "Wall Time", "vs MPS"))
    print("-" * 72)

    for r in results:
        tb = format_thermal(r["thermal_before"])  # type: ignore[arg-type]
        ta = format_thermal(r["thermal_after"])  # type: ignore[arg-type]
        thermal_str = f"{tb} → {ta}"

        tp = r["throughput"]
        tp_str = f"{tp:.1f}/s" if tp is not None else "n/a"

        wall = r["wall_time"]
        wall_str = f"{wall:.1f}s"

        if mps_tp and tp is not None:
            ratio = float(tp) / mps_tp  # type: ignore[arg-type]
            ratio_str = "baseline" if r["cfg"] == "mps" else f"{ratio:.2f}x"
        else:
            ratio_str = "n/a"

        print(fmt.format(str(r["label"]), tp_str, thermal_str, wall_str, ratio_str))

    print("=" * 72)


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------


def save_results(results: list[dict[str, object]]) -> Path:
    """Write results to a timestamped JSON file, return the path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = RESULTS_DIR / f"{ts}.json"
    out_path.write_text(json.dumps(results, indent=2))
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ripvec Metal GEMM benchmark harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=["mps", "compute", "cpu"],
        choices=["mps", "compute", "cpu"],
        metavar="CONFIG",
        help="Configs to benchmark (default: mps compute cpu)",
    )
    p.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[22],
        metavar="N",
        help="Early-exit layer counts (default: 22)",
    )
    p.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        metavar="PATH",
        help="Corpus directory (default: tests/corpus/code/flask)",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Validate correctness across configs (no throughput measurement)",
    )
    p.add_argument(
        "--thermals",
        action="store_true",
        help="Print current thermal state and exit",
    )
    p.add_argument(
        "--no-build",
        action="store_true",
        help="Skip cargo build --release",
    )
    p.add_argument(
        "--query",
        default="session",
        metavar="QUERY",
        help="Search query to use (default: 'session')",
    )
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # --thermals only
    if args.thermals:
        t = read_thermal()
        if t:
            _print(f"Thermal state: {format_thermal(t)}")
            level = thermal_level(t)
            if level > 50:
                _print(
                    "[yellow]Warning: thermal pressure is high (>50). Consider waiting.[/yellow]"
                    if HAS_RICH
                    else "Warning: thermal pressure is high (>50). Consider waiting."
                )
        else:
            _print("Thermal monitoring unavailable on this platform.")
        return 0

    # Validate corpus exists
    corpus = args.corpus.resolve()
    if not corpus.exists():
        print(f"Error: corpus not found: {corpus}", file=sys.stderr)
        print(
            f"Tip: run scripts/fetch-corpus.sh to download test corpora.",
            file=sys.stderr,
        )
        return 1

    # Build unless skipped
    if not args.no_build:
        if not cargo_build():
            return 1

    if not BINARY.exists():
        print(f"Error: binary not found: {BINARY}", file=sys.stderr)
        print("Run: cargo build --release", file=sys.stderr)
        return 1

    configs = expand_configs(args.configs, args.layers)

    # --validate mode
    if args.validate:
        _print("\n=== Correctness validation ===")
        rankings: dict[str, list[str]] = {}
        for label, spec in configs:
            wait_for_cool()
            _print(f"\n{label}...")
            paths = run_validate(label, spec, corpus, args.query)
            rankings[label] = paths
            for i, p in enumerate(paths):
                _print(f"  {i + 1}. {p}")

        # Compare all vs first
        labels = list(rankings.keys())
        if len(labels) < 2:
            _print("Only one config — nothing to compare.")
            return 0

        ref_label = labels[0]
        ref = rankings[ref_label]
        all_match = True
        for lbl, paths in rankings.items():
            if lbl == ref_label:
                continue
            if paths == ref:
                _print(f"\n{lbl} vs {ref_label}: MATCH")
            else:
                all_match = False
                _print(f"\n{lbl} vs {ref_label}: MISMATCH")
                _print(f"  Expected: {ref}")
                _print(f"  Got:      {paths}")

        if all_match:
            _print("\nAll configs produce identical rankings.")
        else:
            _print("\nSome configs produced different rankings — investigate.")
        return 0 if all_match else 2

    # Benchmark mode
    results: list[dict[str, object]] = []
    for label, spec in configs:
        wait_for_cool()
        r = run_benchmark(label, spec, corpus, args.query)
        results.append(r)

    print_table(results)

    out_path = save_results(results)
    _print(f"\nResults saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
