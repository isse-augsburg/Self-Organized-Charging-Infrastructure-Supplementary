#!/usr/bin/env python3
"""Generate a LaTeX table from simulation and real analysis reports."""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
LAYOUTS = ("circle", "grid", "random", "home")
EVENTS = ("Takeoffs", "Landings")
REPORT_LINE = re.compile(
    r"avg: (?P<mean>\d+(?:\.\d+)?) seconds,.*"
    r"min: (?P<minimum>\d+(?:\.\d+)?) seconds, "
    r"max: (?P<maximum>\d+(?:\.\d+)?) seconds"
)


def parse_args() -> argparse.Namespace:
    """Parse input and output paths."""
    parser = argparse.ArgumentParser(
        description="Generate the simulation/real min-mean-max LaTeX table."
    )
    parser.add_argument("--runs-dir", type=Path, default=HERE)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "min_max_statistics_table.txt",
    )
    return parser.parse_args()


def parse_report(
    report_path: Path,
) -> dict[str, list[tuple[float, float, float]]]:
    """Read per-transition mean, minimum, and maximum values."""
    values: dict[str, list[tuple[float, float, float]]] = {
        event: [] for event in EVENTS
    }
    current_event: str | None = None
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if line.removesuffix(":") in EVENTS:
            current_event = line.removesuffix(":")
            continue
        match = REPORT_LINE.search(line)
        if match is not None and current_event is not None:
            values[current_event].append(
                (
                    float(match.group("mean")),
                    float(match.group("minimum")),
                    float(match.group("maximum")),
                )
            )
    if any(not values[event] for event in EVENTS):
        raise ValueError(f"{report_path}: missing takeoff or landing data")
    return values


def select_transitions(
    values: dict[str, list[tuple[float, float, float]]],
    environment: str,
    report_path: Path,
) -> dict[str, list[tuple[float, float, float]]]:
    """Remove home boundaries and validate the expected sample count."""
    selected = {event: list(values[event]) for event in EVENTS}
    expected_count = 3 if environment == "Real" else 50
    if environment == "Simulation":
        if len(selected["Takeoffs"]) == expected_count + 1:
            selected["Takeoffs"] = selected["Takeoffs"][1:]
        if len(selected["Landings"]) == expected_count + 1:
            selected["Landings"] = selected["Landings"][:-1]
    counts = {event: len(selected[event]) for event in EVENTS}
    if any(count != expected_count for count in counts.values()):
        raise ValueError(
            f"{report_path}: expected {expected_count} takeoffs and "
            f"landings after selection, found {counts}"
        )
    return selected


def summarize(
    samples: list[tuple[float, float, float]],
) -> tuple[float, ...]:
    """Calculate average and population deviation of min, mean, and max."""
    means = [sample[0] for sample in samples]
    minimums = [sample[1] for sample in samples]
    maximums = [sample[2] for sample in samples]
    return (
        statistics.fmean(minimums),
        statistics.pstdev(minimums),
        statistics.fmean(means),
        statistics.pstdev(means),
        statistics.fmean(maximums),
        statistics.pstdev(maximums),
    )


def generate_rows(runs_dir: Path) -> list[str]:
    """Generate all eight LaTeX data rows."""
    rows = []
    for environment, prefix in (("Simulation", ""), ("Real", "real_")):
        for layout in LAYOUTS:
            report_path = runs_dir / f"{prefix}{layout}_analysis.txt"
            values = select_transitions(
                parse_report(report_path), environment, report_path
            )
            results = []
            for event in EVENTS:
                results.extend(summarize(values[event]))
            formatted = " & ".join(f"{result:.2f}" for result in results)
            rows.append(
                f"{environment} & {layout.title()} & {formatted} \\\\"
            )
        if environment == "Simulation":
            rows.append(r"\midrule")
    return rows


def format_table(rows: list[str]) -> str:
    """Place calculated rows in a complete, copyable LaTeX table."""
    body = "\n".join(rows)
    return rf"""\begin{{table*}}[htbp]
\centering
\caption{{Mean and population standard deviation of the minimum, mean, and
maximum individual completion times across transitions. All values are in seconds.
Simulation statistics use 50 transitions per layout; the initial home takeoff
and final home landing are excluded from the circle, grid, and random layouts.
Real-world statistics use three transitions per layout.}}
\label{{tab:min-max-transition-times}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrrrrrrrrr}}
\toprule
& & \multicolumn{{6}}{{c}}{{Takeoff}} & \multicolumn{{6}}{{c}}{{Landing}} \\
\cmidrule(lr){{3-8}}\cmidrule(lr){{9-14}}
Environment & Layout
& Avg. min & Std. min & Avg. mean & Std. mean & Avg. max & Std. max
& Avg. min & Std. min & Avg. mean & Std. mean & Avg. max & Std. max \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table*}}
"""


def main() -> int:
    """Generate and write the table."""
    args = parse_args()
    try:
        table = format_table(generate_rows(args.runs_dir))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table, encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Table written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
