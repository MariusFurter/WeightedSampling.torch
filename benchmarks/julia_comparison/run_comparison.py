#!/usr/bin/env python
"""
run_comparison.py — Run WeightedSampling benchmarks (Python & Julia) and write
a comparison table to benchmarks/julia_comparison/results.txt.

Usage:
    python benchmarks/julia_comparison/run_comparison.py           # both languages
    python benchmarks/julia_comparison/run_comparison.py python     # Python only
    python benchmarks/julia_comparison/run_comparison.py julia       # Julia only
"""

import subprocess
import sys
import re
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results.txt")

# ── helpers ──────────────────────────────────────────────────────────────


def run_benchmark(cmd, label):
    """Run a benchmark command and return its stdout, or None on failure."""
    print(f"  Running: {label} ...", flush=True)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, cwd=ROOT
        )
        if result.returncode != 0:
            print(f"    ⚠ FAILED (exit {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().splitlines()[-5:]:
                    print(f"      {line}")
            return None
        print(f"    ✓ done")
        return result.stdout
    except FileNotFoundError:
        print(f"    ⚠ command not found: {cmd[0]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"    ⚠ timed out")
        return None


def parse_timings(output):
    """
    Parse benchmark output and return a list of dicts:
      [{"variant": str, "n_particles": int, "median_time": float}, ...]
    Recognises section headers like "  Resampling only" / "  With MH moves"
    and per-config lines like "--- n_particles = 5000 ---" / "Median time: 0.0151 s".
    """
    entries = []
    current_variant = "default"
    current_n = None

    for line in output.splitlines():
        # Variant header (indented label between ─ lines)
        m = re.match(r"^\s+(Resampling only|With MH moves)", line)
        if m:
            current_variant = m.group(1)
            continue

        m = re.search(r"n_particles\s*=\s*(\d[\d_]*)", line)
        if m:
            current_n = int(m.group(1).replace("_", ""))
            continue

        m = re.search(r"Median time:\s+([\d.]+)\s*s", line)
        if m and current_n is not None:
            entries.append(
                {
                    "variant": current_variant,
                    "n_particles": current_n,
                    "median_time": float(m.group(1)),
                }
            )

    return entries


# ── benchmark definitions ────────────────────────────────────────────────

BENCHMARKS = {
    "8-Schools": {
        "python": ["python", "benchmarks/julia_comparison/eight_schools.py"],
        "julia": ["julia", "--project", "benchmarks/julia_comparison/eight_schools.jl"],
    },
    "SSM (Bootstrap PF)": {
        "python": ["python", "benchmarks/julia_comparison/ssm.py"],
        "julia": ["julia", "--project", "benchmarks/julia_comparison/ssm.jl"],
    },
}


def collect_results(languages):
    """Run benchmarks and return {(model, variant, n_particles): {lang: time}}."""
    raw = {}  # (model, lang) -> parsed entries
    for model_name, cmds in BENCHMARKS.items():
        for lang in languages:
            if lang not in cmds:
                continue
            output = run_benchmark(cmds[lang], f"{model_name} [{lang}]")
            if output is not None:
                raw[(model_name, lang)] = parse_timings(output)

    # Merge into comparison dict
    table = {}  # (model, variant, n) -> {lang: time}
    for (model, lang), entries in raw.items():
        for e in entries:
            key = (model, e["variant"], e["n_particles"])
            table.setdefault(key, {})[lang] = e["median_time"]

    return table


def format_table(table, languages):
    """Format comparison table as a string."""
    lines = []
    lines.append(f"WeightedSampling Benchmark Comparison")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Languages: {', '.join(languages)}")
    lines.append("")

    # Group by (model, variant)
    groups = {}
    for (model, variant, n), timings in sorted(table.items()):
        groups.setdefault((model, variant), []).append((n, timings))

    for (model, variant), rows in groups.items():
        lines.append("=" * 72)
        lines.append(f"  {model}  —  {variant}")
        lines.append("=" * 72)

        # Header
        lang_cols = "".join(f"{lang:>12s}" for lang in languages)
        ratio_col = "   ratio (py/jl)" if len(languages) == 2 else ""
        lines.append(f"  {'n_particles':>12s}{lang_cols}{ratio_col}")
        lines.append(
            f"  {'-' * 12}"
            + "".join(f"  {'-' * 10}" for _ in languages)
            + ("  " + "-" * 14 if len(languages) == 2 else "")
        )

        for n, timings in sorted(rows):
            vals = []
            for lang in languages:
                t = timings.get(lang)
                vals.append(f"{t:12.4f}" if t is not None else f"{'—':>12s}")
            ratio_str = ""
            if len(languages) == 2 and "python" in timings and "julia" in timings:
                r = timings["python"] / timings["julia"]
                ratio_str = f"   {r:>10.2f}x"
            lines.append(f"  {n:>12d}" + "".join(vals) + ratio_str)

        lines.append("")

    # Summary legend
    lines.append("-" * 72)
    lines.append("Times are median of 5 runs, in seconds. Lower is better.")
    lines.append("Ratio > 1 means Julia is faster; < 1 means Python is faster.")
    lines.append("")

    return "\n".join(lines)


def main():
    lang_arg = sys.argv[1] if len(sys.argv) > 1 else "both"
    if lang_arg == "python":
        languages = ["python"]
    elif lang_arg == "julia":
        languages = ["julia"]
    elif lang_arg == "both":
        languages = ["python", "julia"]
    else:
        print(f"Usage: {sys.argv[0]} [python|julia|both]")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  WeightedSampling Benchmark Comparison")
    print(f"  Languages: {', '.join(languages)}")
    print(f"{'=' * 60}\n")

    table = collect_results(languages)

    if not table:
        print("No benchmark results collected. Check errors above.")
        sys.exit(1)

    report = format_table(table, languages)

    # Print to stdout
    print("\n" + report)

    # Write to file
    with open(RESULTS_FILE, "w") as f:
        f.write(report)
    print(f"Results written to: benchmarks/julia_comparison/results.txt")


if __name__ == "__main__":
    main()
