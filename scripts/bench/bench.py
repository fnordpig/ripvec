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
    --sudo                      Use sudo powermetrics for richer thermal data (requires sudoers entry)
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
# Thermal / resource helpers
# ---------------------------------------------------------------------------

# pmset -g therm output patterns (Apple Silicon + Intel):
#   CPU_Speed_Limit         = 100   (100 = no throttling, lower = throttled)
#   CPU_Scheduler_Limit     = 100
_PMSET_SPEED_PATTERN = re.compile(r"CPU_Speed_Limit\s*=\s*(\d+)")

# Apple Silicon GPU node names in priority order (chip generation dependent)
_GPU_NODES = [
    "AGXAcceleratorG14X",  # M2 Max / M2 Ultra
    "AGXAcceleratorG14P",  # M2 Pro
    "AGXAcceleratorG13X",  # M1 Max / M1 Ultra
    "AGXAcceleratorG13P",  # M1 Pro
    "AGXAcceleratorG13G",  # M1
    "AGXAccelerator",  # generic fallback
]
_IOREG_PERF_PATTERN = re.compile(r'"PerformanceStatistics"\s*=\s*(\{[^}]+\})')
_IOREG_KV_PATTERN = re.compile(r'"([^"]+)"=(\d+)')

# sudo powermetrics -s thermal text output pattern:
#   Thermal pressure: Nominal
_PM_THERMAL_PATTERN = re.compile(r"Thermal pressure:\s*(\w+)", re.IGNORECASE)


def _cpu_speed_limit() -> int | None:
    """Return CPU_Speed_Limit % from pmset (100 = full speed, <100 = throttled).

    Returns None if unavailable (non-Apple platform or pmset missing).
    """
    try:
        out = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode != 0:
            return None
        # "No thermal warning" means full speed
        if "No thermal warning" in out.stdout or "No performance warning" in out.stdout:
            return 100
        m = _PMSET_SPEED_PATTERN.search(out.stdout)
        if m:
            return int(m.group(1))
        # pmset returned output but no CPU_Speed_Limit line — assume full speed
        return 100
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _gpu_utilization() -> int | None:
    """Return GPU Device Utilization % via ioreg (no sudo required).

    Works on Apple Silicon Macs. Returns None if unavailable.
    """
    for node in _GPU_NODES:
        try:
            out = subprocess.run(
                ["ioreg", "-r", "-d", "2", "-n", node],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if out.returncode != 0 or not out.stdout.strip():
                continue
            m = _IOREG_PERF_PATTERN.search(out.stdout)
            if not m:
                continue
            stats = dict(_IOREG_KV_PATTERN.findall(m.group(1)))
            val = stats.get("Device Utilization %")
            if val is not None:
                return int(val)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            break
    return None


def _mem_used_gb() -> float | None:
    """Return active+inactive+wired memory in GB from vm_stat (no sudo required)."""
    try:
        out = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode != 0:
            return None
        m = re.search(r"page size of (\d+)", out.stdout)
        page_size = int(m.group(1)) if m else 16384
        pages: dict[str, int] = {}
        for pm in re.finditer(r"Pages (\w+):\s+(\d+)", out.stdout):
            pages[pm.group(1)] = int(pm.group(2))
        used = (
            pages.get("active", 0)
            + pages.get("inactive", 0)
            + pages.get("wired", 0)
            + pages.get("speculative", 0)
        )
        return used * page_size / 1024**3
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _sudo_thermal_pressure() -> str | None:
    """Return thermal pressure string via sudo powermetrics (requires sudo NOPASSWD).

    Returns a string like 'Nominal', 'Moderate', 'Heavy', 'Trapping', or None.
    """
    try:
        out = subprocess.run(
            ["sudo", "powermetrics", "-s", "thermal", "-i", "1000", "-n", "1"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if out.returncode != 0:
            return None
        m = _PM_THERMAL_PATTERN.search(out.stdout)
        if m:
            return m.group(1)
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


class ThermalSnapshot:
    """Snapshot of system thermal and resource state."""

    def __init__(
        self,
        cpu_speed: int | None,
        gpu_util: int | None,
        mem_gb: float | None,
        sudo_pressure: str | None = None,
    ) -> None:
        self.cpu_speed = cpu_speed  # CPU_Speed_Limit % (100=full, <100=throttled)
        self.gpu_util = gpu_util  # GPU Device Utilization %
        self.mem_gb = mem_gb  # Used memory in GB
        self.sudo_pressure = sudo_pressure  # powermetrics thermal pressure string

    def is_throttled(self) -> bool:
        """Return True if CPU is being speed-limited (< 100%)."""
        return self.cpu_speed is not None and self.cpu_speed < 100

    def format(self) -> str:
        """Return compact one-liner like 'speed=98% gpu=45% mem=12.4GB'."""
        parts: list[str] = []
        if self.sudo_pressure is not None:
            parts.append(f"pressure={self.sudo_pressure}")
        if self.cpu_speed is not None:
            parts.append(f"speed={self.cpu_speed}%")
        if self.gpu_util is not None:
            parts.append(f"gpu={self.gpu_util}%")
        if self.mem_gb is not None:
            parts.append(f"mem={self.mem_gb:.1f}GB")
        return " ".join(parts) if parts else "n/a"


def read_thermal(use_sudo: bool = False) -> ThermalSnapshot:
    """Read CPU throttle state, GPU utilization, and memory usage.

    Args:
        use_sudo: If True, also attempt sudo powermetrics for thermal pressure.

    No sudo required for the base readings (speed/gpu/mem).
    """
    cpu_speed = _cpu_speed_limit()
    gpu_util = _gpu_utilization()
    mem_gb = _mem_used_gb()
    sudo_pressure = _sudo_thermal_pressure() if use_sudo else None
    return ThermalSnapshot(cpu_speed, gpu_util, mem_gb, sudo_pressure)


def format_thermal(t: ThermalSnapshot) -> str:
    """Return a compact thermal string (delegates to ThermalSnapshot.format)."""
    return t.format()


def thermal_level(t: ThermalSnapshot) -> int:
    """Return a 0-100 throttle pressure score (0=full speed, 100=fully throttled).

    Used by wait_for_cool() to decide whether to wait.
    """
    if t.cpu_speed is None:
        return 0
    return max(0, 100 - t.cpu_speed)


def wait_for_cool(
    threshold_warn: int = 5,
    threshold_ok: int = 2,
    use_sudo: bool = False,
) -> None:
    """Block until CPU throttle pressure drops below threshold_ok.

    Thresholds are on the 0-100 pressure scale (100 - cpu_speed_limit).
    A threshold_warn of 5 means: warn if speed limit is below 95%.
    A threshold_ok of 2 means: wait until speed limit is >= 98%.
    """
    t = read_thermal(use_sudo)
    level = thermal_level(t)
    if level <= threshold_warn:
        return
    _print(
        f"[yellow]CPU throttled ({format_thermal(t)}), waiting for cool-down...[/yellow]"
        if HAS_RICH
        else f"CPU throttled ({format_thermal(t)}), waiting for cool-down..."
    )
    while level > threshold_ok:
        time.sleep(10)
        t = read_thermal(use_sudo)
        level = thermal_level(t)
        _print(f"  still throttled: {format_thermal(t)}")
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
    configs: list[str],
    layers: list[int],
    batch_sizes: list[int] | None = None,
) -> list[tuple[str, dict[str, object]]]:
    """Return list of (label, spec) pairs.

    spec keys:
      env     -- extra environment variables dict
      args    -- extra CLI args list
      layers  -- layer count (int or None)
      batch   -- batch size (int)
    """
    bs_list = batch_sizes or [0]  # 0 = use default (don't pass -b flag)
    result = []
    for cfg in configs:
        for L in layers:
            for bs in bs_list:
                if bs > 0:
                    label = f"{cfg}-{L}L-b{bs}"
                else:
                    label = f"{cfg}-{L}L"
                env: dict[str, str] = {}
                extra_args: list[str] = []

                if cfg == "compute":
                    env["RIPVEC_NO_MPS"] = "1"
                elif cfg == "cpu":
                    extra_args += ["--device", "cpu"]
                elif cfg == "cuda":
                    extra_args += ["--backend", "cuda"]
                # mps: default, no extra env/args

                extra_args += ["--layers", str(L)]
                if bs > 0:
                    extra_args += ["-b", str(bs)]
                result.append(
                    (
                        label,
                        {
                            "env": env,
                            "args": extra_args,
                            "layers": L,
                            "batch": bs,
                            "cfg": cfg,
                        },
                    )
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
    """Background thread that prints resource readings while a run is active."""

    def __init__(
        self, label: str, interval: float = 5.0, use_sudo: bool = False
    ) -> None:
        self.label = label
        self.interval = interval
        self.use_sudo = use_sudo
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start_time = time.monotonic()
        self.readings: list[ThermalSnapshot] = []

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
            t = read_thermal(self.use_sudo)
            self.readings.append(t)
            print(
                f"  [{self.label}] {elapsed:.0f}s {format_thermal(t)}",
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
    use_sudo: bool = False,
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

    thermal_before = read_thermal(use_sudo)
    _print(
        f"\n[bold]{label}[/bold] {format_thermal(thermal_before)}"
        if HAS_RICH
        else f"\n{label}  {format_thermal(thermal_before)}"
    )
    _print(f"  cmd: {' '.join(cmd)}")

    ticker = ThermalTicker(label, use_sudo=use_sudo)
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

    thermal_after = read_thermal(use_sudo)
    output = proc.stdout or ""

    throughput = parse_throughput(output)

    # Print the last few lines so the throughput line is visible
    lines = [ln for ln in output.splitlines() if ln.strip()]
    for line in lines[-5:]:
        print(f"  {line}")

    result: dict[str, object] = {
        "label": label,
        "cfg": spec["cfg"],
        "layers": spec["layers"],
        "throughput": throughput,
        "wall_time": round(wall_time, 2),
        "thermal_before": format_thermal(thermal_before),
        "thermal_after": format_thermal(thermal_after),
        "returncode": proc.returncode,
    }

    status = f"{throughput:.1f}/s" if throughput is not None else "PARSE_ERROR"
    _print(f"  done: {status}  wall={wall_time:.1f}s  {format_thermal(thermal_after)}")
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
        tb = str(r["thermal_before"])
        ta = str(r["thermal_after"])
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
        choices=["mps", "compute", "cpu", "cuda"],
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
        "--batch-sizes",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Batch sizes to test (default: use ripvec default). Cross-product with configs × layers.",
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
    p.add_argument(
        "--sudo",
        action="store_true",
        help="Use sudo powermetrics for thermal pressure (requires NOPASSWD sudoers entry)",
    )
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    use_sudo = args.sudo

    # --thermals only
    if args.thermals:
        t = read_thermal(use_sudo)
        snap = format_thermal(t)
        if snap != "n/a":
            _print(f"Thermal state: {snap}")
            if t.is_throttled():
                _print(
                    f"[yellow]Warning: CPU throttled to {t.cpu_speed}% speed.[/yellow]"
                    if HAS_RICH
                    else f"Warning: CPU throttled to {t.cpu_speed}% speed."
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

    configs = expand_configs(args.configs, args.layers, args.batch_sizes)

    # --validate mode
    if args.validate:
        _print("\n=== Correctness validation ===")
        rankings: dict[str, list[str]] = {}
        for label, spec in configs:
            wait_for_cool(use_sudo=use_sudo)
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
        wait_for_cool(use_sudo=use_sudo)
        r = run_benchmark(label, spec, corpus, args.query, use_sudo=use_sudo)
        results.append(r)

    print_table(results)

    out_path = save_results(results)
    _print(f"\nResults saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
