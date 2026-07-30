#!/usr/bin/env python3
"""Run command-duration analysis for an extensive experiment log."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent / "runs" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from command_analysis import (  # noqa: E402
    AnalysisError,
    calculate_statistics,
    format_summary,
    parse_groups,
    plot_statistics,
    validate_groups,
)

from compare_transition_spread import (  # noqa: E402
    analyze_completion_times,
    plot_single_result,
)


def parse_args() -> argparse.Namespace:
    """Parse configurable input and output paths."""
    parser = argparse.ArgumentParser(
        description="Analyze an extensive ARCS experiment run."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=HERE / "circle.log",
        help="input launch log (default: runs_extensive/circle.log)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "circle_analysis.txt",
        help=(
            "report output path "
            "(default: runs_extensive/circle_analysis.txt)"
        ),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=HERE / "circle_analysis.png",
        help=(
            "plot output path "
            "(default: runs_extensive/circle_analysis.png)"
        ),
    )
    parser.add_argument(
        "--total-output",
        type=Path,
        default=HERE / "circle_total_time.txt",
        help="total transition-time report output path",
    )
    parser.add_argument(
        "--total-plot",
        type=Path,
        default=HERE / "circle_total_time.png",
        help="total transition-time plot output path",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=10,
        help="expected number of unique padflies in each group (default: 10)",
    )
    parser.add_argument("--cluster-gap", type=float, default=2.0)
    parser.add_argument("--speed-factor", type=float, default=4.0)
    return parser.parse_args()


def format_total_report(results: dict[str, list[float]]) -> str:
    """Format initiation-to-last-finished statistics."""
    lines = []
    for event_name, samples in results.items():
        lines.append(
            f"{event_name}: n={len(samples)}, "
            f"avg={statistics.fmean(samples):.2f} seconds, "
            f"stddev={statistics.pstdev(samples):.2f} seconds, "
            f"median={statistics.median(samples):.2f} seconds, "
            f"min={min(samples):.2f} seconds, "
            f"max={max(samples):.2f} seconds"
        )
    return "\n".join(lines)


def main() -> int:
    """Analyze the configured log and write and print its report."""
    args = parse_args()
    valid_numbers = all(
        (
            args.expected_count >= 1,
            args.cluster_gap > 0,
            args.speed_factor > 0,
        )
    )
    if not valid_numbers:
        print(
            "error: numeric analysis options must be positive",
            file=sys.stderr,
        )
        return 2
    try:
        scenario = args.log.stem
        groups = parse_groups(args.log)
        validate_groups(groups, args.expected_count)
        complete_groups = [
            group
            for group in groups
            if len(group.durations) == args.expected_count
        ]
        statistics_by_group = calculate_statistics(complete_groups)
        report = format_summary(statistics_by_group)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        plot_statistics(
            statistics_by_group,
            args.plot,
            scenario=scenario,
            exclude_initial_takeoff=scenario != "home",
        )

        total_results = analyze_completion_times(
            args.log,
            args.expected_count,
            args.cluster_gap,
            args.speed_factor,
            exclude_initial_takeoff=scenario != "home",
        )
        total_report = format_total_report(total_results)
        args.total_output.parent.mkdir(parents=True, exist_ok=True)
        args.total_output.write_text(total_report + "\n", encoding="utf-8")
        plot_single_result(scenario, total_results, args.total_plot)
    except (AnalysisError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(report)
    print(f"\nReport written to {args.output}")
    print(f"Plot written to {args.plot}")
    print(f"\n{total_report}")
    print(f"Total-time report written to {args.total_output}")
    print(f"Total-time plot written to {args.total_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
