#!/usr/bin/env python3
"""Analyze and compare all completed extensive-layout experiment logs."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent / "runs" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from command_analysis import (  # noqa: E402
    AnalysisError as CommandAnalysisError,
    GroupStatistics,
    calculate_statistics,
    format_summary,
    parse_groups,
    plot_comparison,
    plot_statistics,
    validate_groups,
)

from compare_transition_spread import (  # noqa: E402
    AnalysisError as TransitionAnalysisError,
    analyze_completion_times,
    plot_results,
    plot_single_result,
)


def parse_args() -> argparse.Namespace:
    """Parse analysis configuration."""
    parser = argparse.ArgumentParser(
        description="Analyze circle and grid extensive experiment logs."
    )
    parser.add_argument("--runs-dir", type=Path, default=HERE)
    parser.add_argument(
        "--scenarios", nargs="+", default=["circle", "grid"]
    )
    parser.add_argument("--expected-count", type=int, default=10)
    parser.add_argument("--cluster-gap", type=float, default=2.0)
    parser.add_argument("--speed-factor", type=float, default=4.0)
    parser.add_argument(
        "--home-max-cycles",
        type=int,
        default=50,
        help=(
            "maximum combined home transitions per state; "
            "0 keeps all complete transitions (default: 50)"
        ),
    )
    return parser.parse_args()


def format_total_report(results: dict[str, list[float]]) -> str:
    """Format total transition-time statistics."""
    return "\n".join(
        (
            f"{event_name}: n={len(samples)}, "
            f"avg={statistics.fmean(samples):.2f} seconds, "
            f"stddev={statistics.pstdev(samples):.2f} seconds, "
            f"median={statistics.median(samples):.2f} seconds, "
            f"min={min(samples):.2f} seconds, "
            f"max={max(samples):.2f} seconds"
        )
        for event_name, samples in results.items()
    )


def limit_group_statistics(
    statistics_by_group: list[GroupStatistics],
    maximum_per_state: int,
) -> list[GroupStatistics]:
    """Keep only the first N complete groups for each transition state."""
    counts = {5: 0, 2: 0}
    limited = []
    for result in statistics_by_group:
        if counts[result.target_state] >= maximum_per_state:
            continue
        limited.append(result)
        counts[result.target_state] += 1
    return limited


def main() -> int:
    """Generate individual and combined reports and plots."""
    args = parse_args()
    valid_numbers = all(
        (
            args.expected_count >= 1,
            args.cluster_gap > 0,
            args.speed_factor > 0,
            args.home_max_cycles >= 0,
        )
    )
    if not valid_numbers:
        print(
            "error: numeric analysis options must be positive",
            file=sys.stderr,
        )
        return 2

    individual_results = {}
    total_results = {}
    try:
        for scenario in args.scenarios:
            if scenario == "home":
                log_paths = sorted(args.runs_dir.glob("home*.log"))
            else:
                log_paths = [args.runs_dir / f"{scenario}.log"]
            if not log_paths:
                raise OSError(f"no logs found for {scenario}")

            group_statistics = []
            totals = {"Takeoffs": [], "Landings": []}
            for log_path in log_paths:
                groups = parse_groups(log_path)
                validate_groups(groups, args.expected_count)
                complete_groups = [
                    group
                    for group in groups
                    if len(group.durations) == args.expected_count
                ]
                group_statistics.extend(
                    calculate_statistics(complete_groups)
                )
                file_totals = analyze_completion_times(
                    log_path,
                    args.expected_count,
                    args.cluster_gap,
                    args.speed_factor,
                    exclude_initial_takeoff=scenario != "home",
                )
                for event_name, samples in file_totals.items():
                    totals[event_name].extend(samples)

            if scenario == "home" and args.home_max_cycles:
                group_statistics = limit_group_statistics(
                    group_statistics,
                    args.home_max_cycles,
                )
            individual_results[scenario] = group_statistics

            individual_report = format_summary(group_statistics)
            (args.runs_dir / f"{scenario}_analysis.txt").write_text(
                individual_report + "\n", encoding="utf-8"
            )
            plot_statistics(
                group_statistics,
                args.runs_dir / f"{scenario}_analysis.png",
                scenario=scenario,
                exclude_initial_takeoff=scenario != "home",
            )

            if scenario == "home" and args.home_max_cycles:
                totals = {
                    event_name: samples[:args.home_max_cycles]
                    for event_name, samples in totals.items()
                }
            total_results[scenario] = totals
            (args.runs_dir / f"{scenario}_total_time.txt").write_text(
                format_total_report(totals) + "\n",
                encoding="utf-8",
            )
            plot_single_result(
                scenario,
                totals,
                args.runs_dir / f"{scenario}_total_time.png",
            )

        plot_comparison(
            individual_results,
            args.runs_dir / "individual_time_comparison.png",
        )
        plot_results(
            total_results,
            args.runs_dir / "total_time_comparison.png",
        )
    except (
        CommandAnalysisError,
        TransitionAnalysisError,
        OSError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Analyzed: {', '.join(args.scenarios)}")
    print(f"Reports and plots written to {args.runs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
