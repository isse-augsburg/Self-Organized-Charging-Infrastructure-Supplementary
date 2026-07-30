#!/usr/bin/env python3
"""Analyze real-hardware runs as four pad-layout categories."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402, I202


HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent / "runs" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from command_analysis import (  # noqa: E402, I100
    CommandGroup,
    GroupStatistics,
    calculate_statistics,
    format_summary,
    parse_groups,
)

from compare_transition_spread import (  # noqa: E402
    AnalysisError,
    TAKEOFF_EVENT,
    analyze_completion_times,
    parse_clusters,
    parse_finished_events,
)


LAYOUTS = ("circle", "grid", "random", "home")
COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756")


def parse_args() -> argparse.Namespace:
    """Parse real-run analysis paths and settings."""
    parser = argparse.ArgumentParser(
        description="Analyze real circle, grid, random, and home transitions."
    )
    parser.add_argument("--runs-dir", type=Path, default=HERE)
    parser.add_argument("--expected-count", type=int, default=10)
    parser.add_argument(
        "--cluster-gap",
        type=float,
        default=10.0,
        help="event clustering gap in real seconds (default: 10)",
    )
    parser.add_argument(
        "--individual-plot",
        type=Path,
        default=HERE / "real_individual_time_comparison.png",
    )
    parser.add_argument(
        "--total-plot",
        type=Path,
        default=HERE / "real_total_time_comparison.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "real_analysis.txt",
    )
    return parser.parse_args()


def first_home_takeoff_total(
    log_path: Path,
    expected_count: int,
    cluster_gap: float,
) -> float:
    """Measure initial home takeoff through the first N completions."""
    clusters = parse_clusters(
        log_path,
        TAKEOFF_EVENT,
        cluster_gap,
        expected_count,
    )
    if not clusters:
        raise AnalysisError(f"{log_path.name}: no initial takeoff cluster")
    start = clusters[0].first_timestamp
    completions = sorted(
        timestamp
        for timestamp, state, _ in parse_finished_events(log_path)
        if state == 5 and timestamp >= start
    )
    if len(completions) < expected_count:
        raise AnalysisError(
            f"{log_path.name}: initial takeoff has only "
            f"{len(completions)} completions"
        )
    return completions[expected_count - 1] - start


def analyze_real_runs(
    runs_dir: Path,
    expected_count: int,
    cluster_gap: float,
) -> tuple[
    dict[str, dict[str, list[float]]],
    dict[str, dict[str, list[float]]],
    dict[str, list[GroupStatistics]],
]:
    """Classify real transitions into circle, grid, random, and home."""
    individual = {
        layout: {"Takeoffs": [], "Landings": []} for layout in LAYOUTS
    }
    total = {
        layout: {"Takeoffs": [], "Landings": []} for layout in LAYOUTS
    }
    detailed = {layout: [] for layout in LAYOUTS}

    for layout in LAYOUTS[:-1]:
        log_path = runs_dir / f"real_{layout}.log"
        groups = parse_groups(log_path)
        takeoffs = [group for group in groups if group.target_state == 5]
        landings = [group for group in groups if group.target_state == 2]
        if len(takeoffs) != 4 or len(landings) != 4:
            raise AnalysisError(
                f"{log_path.name}: expected four takeoffs and four landings, "
                f"found {len(takeoffs)} and {len(landings)}"
            )

        # The eleventh initial completion is a replacement retry. The first
        # ten completions represent the ten-agent home takeoff.
        home_durations = list(takeoffs[0].durations.values())[:expected_count]
        home_takeoff = CommandGroup(
            target_state=5,
            durations=dict(enumerate(home_durations)),
        )
        detailed["home"].extend(
            calculate_statistics([home_takeoff, landings[3]])
        )
        detailed[layout].extend(
            calculate_statistics(takeoffs[1:] + landings[:3])
        )
        individual["home"]["Takeoffs"].append(
            statistics.fmean(home_durations)
        )
        individual[layout]["Takeoffs"].extend(
            statistics.fmean(group.durations.values())
            for group in takeoffs[1:]
        )
        individual[layout]["Landings"].extend(
            statistics.fmean(group.durations.values())
            for group in landings[:3]
        )
        individual["home"]["Landings"].append(
            statistics.fmean(landings[3].durations.values())
        )

        completion_times = analyze_completion_times(
            log_path,
            expected_count,
            cluster_gap,
            speed_factor=1.0,
            exclude_initial_takeoff=False,
        )
        total["home"]["Takeoffs"].append(
            first_home_takeoff_total(
                log_path,
                expected_count,
                cluster_gap,
            )
        )
        total[layout]["Takeoffs"].extend(completion_times["Takeoffs"])
        total[layout]["Landings"].extend(
            completion_times["Landings"][:3]
        )
        total["home"]["Landings"].append(
            completion_times["Landings"][-1]
        )

    return individual, total, detailed


def plot_metric(
    results: dict[str, dict[str, list[float]]],
    output_path: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    """Plot takeoff and landing results with four layout elements."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, event_name in zip(axes, ("Takeoffs", "Landings")):
        values = [results[layout][event_name] for layout in LAYOUTS]
        positions = range(1, len(LAYOUTS) + 1)
        violins = axis.violinplot(
            values,
            positions=positions,
            showmeans=True,
            showmedians=True,
            showextrema=True,
        )
        for body, color in zip(violins["bodies"], COLORS):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.3)
        for position, color, samples in zip(
            positions, COLORS, values
        ):
            axis.scatter(
                [position] * len(samples),
                samples,
                color=color,
                s=35,
                alpha=0.75,
            )
            mean = statistics.fmean(samples)
            axis.annotate(
                f"{mean:.2f} s",
                (position, mean),
                xytext=(7, 4),
                textcoords="offset points",
            )
        axis.set_title(event_name)
        axis.set_xticks(
            list(positions), [layout.title() for layout in LAYOUTS]
        )
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def format_report(
    individual: dict[str, dict[str, list[float]]],
    total: dict[str, dict[str, list[float]]],
) -> str:
    """Format means and standard deviations for both real-run metrics."""
    lines = []
    for metric_name, results in (
        ("Individual command time", individual),
        ("Total transition time", total),
    ):
        lines.append(f"{metric_name}:")
        for layout in LAYOUTS:
            parts = []
            for event_name in ("Takeoffs", "Landings"):
                samples = results[layout][event_name]
                parts.append(
                    f"{event_name.lower()} "
                    f"{statistics.fmean(samples):.2f} s "
                    f"(stddev {statistics.pstdev(samples):.2f}, "
                    f"n={len(samples)})"
                )
            lines.append(f"  {layout.title()}: " + ", ".join(parts))
    return "\n".join(lines)


def main() -> int:
    """Analyze real logs and generate two four-layout plots."""
    args = parse_args()
    if args.expected_count < 1 or args.cluster_gap <= 0:
        print("error: numeric options must be positive", file=sys.stderr)
        return 2
    try:
        individual, total, detailed = analyze_real_runs(
            args.runs_dir,
            args.expected_count,
            args.cluster_gap,
        )
        report = format_report(individual, total)
        args.output.write_text(report + "\n", encoding="utf-8")
        for layout, group_statistics in detailed.items():
            output_path = args.runs_dir / f"real_{layout}_analysis.txt"
            output_path.write_text(
                format_summary(group_statistics) + "\n",
                encoding="utf-8",
            )
        plot_metric(
            individual,
            args.individual_plot,
            title=(
                "Average Crazyflie Takeoff and Landing Time "
                "by Pad Layout"
            ),
            ylabel="Average Crazyflie completion time [seconds]",
        )
        plot_metric(
            total,
            args.total_plot,
            title="Full-Swarm Takeoff and Landing Time by Pad Layout",
            ylabel="Full-swarm completion time [seconds]",
        )
    except (AnalysisError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(report)
    print(f"\nPlots written to {args.individual_plot} and {args.total_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
