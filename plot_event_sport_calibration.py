#!/usr/bin/env python3
"""Plot calibration by Pietro's event-joined sport labels and volume tiers.

Inputs are the saved hourly master CSV, market-to-event links, and event records.
Each available hour contributes a forecast, with each market counted at most
once per probability bin. The event-slug-first sport rules come from Pietro's
combined pipeline notebook. The script makes all-volume and volume-tier plots
for the seven largest sports, with Wilson intervals and weighted regression.
Use --exclude-final-hour to omit hour 01 and use hours 02–24.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_MASTER = ROOT / "Data" / "all_closed_sports" / "closed_sports_all_data.csv"
DEFAULT_OUTPUT_DIR = ROOT / "Figures" / "calibration_by_event_sport_volume"
DEFAULT_OVERALL_OUTPUT_DIR = ROOT / "Figures" / "calibration_by_event_sport_all_volumes"
DEFAULT_NO_FINAL_HOUR_OUTPUT_DIR = ROOT / "Figures" / "calibration_by_event_sport_volume_without_final_hour"
DEFAULT_NO_FINAL_HOUR_OVERALL_DIR = ROOT / "Figures" / "calibration_by_event_sport_all_volumes_without_final_hour"
DEFAULT_EVENTS_DIR = ROOT / "Data" / "all_closed_sports"
DEFAULT_NOTEBOOK = ROOT / "Polymarket_Closed_Sports_Combined_Pipeline.ipynb"
UNCLASSIFIED = "Other / unclassified"
VOLUME_TIERS = ["Low volume", "Mid volume", "High volume"]
def as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def hourly_market_bins(
    master: Path, n_bins: int, start_hour: int = 1, chunk_size: int = 25_000
) -> pd.DataFrame:
    """One row per market and probability bin, using every valid selected hour."""
    if not master.is_file():
        raise FileNotFoundError(master)
    usecols = ["id", "outcomes", "outcomePrices", "volumeNum"]
    for hour in range(start_hour, 25):
        prefix = f"hour_{hour:02d}"
        usecols.extend([f"{prefix}_outcomes", f"{prefix}_vwap_price"])

    forecasts = []
    scanned = 0
    for chunk in pd.read_csv(
        master,
        usecols=usecols,
        dtype={"id": "string"},
        chunksize=chunk_size,
        low_memory=False,
    ):
        scanned += len(chunk)
        markets = chunk[["id", "outcomes", "outcomePrices", "volumeNum"]].copy()
        markets["market_id"] = markets["id"].astype("string")
        markets["outcomes"] = markets["outcomes"].map(as_list)
        markets["settled"] = markets["outcomePrices"].map(as_list)
        binary = markets.loc[
            markets["outcomes"].str.len().eq(2) & markets["settled"].str.len().eq(2)
        ].copy()
        binary["first_result"] = pd.to_numeric(binary["settled"].str[0], errors="coerce")
        binary = binary.loc[binary["first_result"].le(0.02) | binary["first_result"].ge(0.98)].copy()
        binary["actual"] = binary["first_result"].ge(0.98).astype("int8")
        if binary.empty:
            print(f"Scanned {scanned:,} markets; market-bin observations: {sum(map(len, forecasts)):,}", flush=True)
            continue

        activity_parts = []
        for hour in range(start_hour, 25):
            prefix = f"hour_{hour:02d}"
            part = pd.DataFrame(index=binary.index)
            part["market_id"] = binary["market_id"]
            part["token"] = chunk.loc[binary.index, f"{prefix}_outcomes"].map(as_list)
            part["vwap_price"] = chunk.loc[binary.index, f"{prefix}_vwap_price"].map(as_list)
            part = part.explode(["token", "vwap_price"])
            part = part.loc[part["token"].map(lambda value: isinstance(value, dict))].copy()
            if part.empty:
                continue
            part["outcome_index"] = pd.to_numeric(part["token"].str.get("outcome_index"), errors="coerce")
            part["vwap_price"] = pd.to_numeric(part["vwap_price"], errors="coerce")
            part["hours_before_close"] = hour
            valid = part["outcome_index"].isin([0, 1]) & part["vwap_price"].between(0, 1)
            activity_parts.append(part.loc[valid, [
                "market_id", "hours_before_close", "outcome_index", "vwap_price",
            ]])

        if not activity_parts:
            print(f"Scanned {scanned:,} markets; market-bin observations: {sum(map(len, forecasts)):,}", flush=True)
            continue
        activity = pd.concat(activity_parts, ignore_index=True)
        prices = activity.pivot_table(
            index=["market_id", "hours_before_close"],
            columns="outcome_index", values="vwap_price", aggfunc="first"
        ).reindex(columns=[0, 1])
        prices["probability"] = prices[0].fillna(1 - prices[1])
        prices = prices.dropna(subset=["probability"]).reset_index()
        prices["probability_bin"] = np.minimum(
            (prices["probability"].clip(0, 1) * n_bins).astype(int), n_bins - 1
        )
        # A market contributes once per bin, even if several hours fall in it.
        # The mean of those hourly probabilities is its representative x value.
        result = prices.groupby(["market_id", "probability_bin"], as_index=False).agg(
            probability=("probability", "mean"), hours_observed=("probability", "size")
        )
        metadata = binary[["market_id", "actual", "volumeNum"]]
        result = result.merge(metadata, on="market_id", how="left", validate="many_to_one")
        if not result.empty:
            forecasts.append(result)
        print(f"Scanned {scanned:,} markets; market-bin observations: {sum(map(len, forecasts)):,}", flush=True)

    if not forecasts:
        return pd.DataFrame(columns=["market_id", "probability_bin", "probability", "hours_observed", "actual", "volumeNum"])
    return pd.concat(forecasts, ignore_index=True)


def calibration_bins(frame: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Summarize binary outcomes in equal-width forecast-probability bins."""
    frame = frame.copy()
    if "probability_bin" not in frame:
        frame["probability_bin"] = np.minimum(
            (frame["probability"].clip(0, 1) * n_bins).astype(int), n_bins - 1
        )
    bins = frame.groupby("probability_bin").agg(
        n=("actual", "size"),
        mean_probability=("probability", "mean"),
        observed_rate=("actual", "mean"),
    )
    bins = bins.loc[bins["n"].gt(0)].copy()

    n = bins["n"].to_numpy(dtype=float)
    rate = bins["observed_rate"].to_numpy(dtype=float)
    z = 1.96
    denominator = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denominator
    # Include the observed rate exactly at boundary cases (e.g. all events).
    bins["ci_low"] = np.minimum(center - margin, rate)
    bins["ci_high"] = np.maximum(center + margin, rate)
    return bins


def weighted_regression(bins: pd.DataFrame) -> tuple[float, float] | None:
    """Fit observed = intercept + slope * predicted, weighted by bin size."""
    if len(bins) < 2:
        return None
    x = bins["mean_probability"].to_numpy(dtype=float)
    y = bins["observed_rate"].to_numpy(dtype=float)
    weights = np.sqrt(bins["n"].to_numpy(dtype=float))
    design = np.column_stack([np.ones(len(bins)), x])
    intercept, slope = np.linalg.lstsq(
        design * weights[:, None], y * weights, rcond=None
    )[0]
    return float(intercept), float(slope)


def safe_filename(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def notebook_rule_assignments(notebook_path: Path) -> dict[str, object]:
    """Read Pietro's literal sport rules from the notebook without executing notebook cells."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    wanted = {"sport_rules", "slug_prefix_rules", "EVENT_OVERRIDE_RULES"}
    found: dict[str, object] = {}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign):
                names = [target.id for target in node.targets if isinstance(target, ast.Name)]
                for name in names:
                    if name in wanted:
                        try:
                            found[name] = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            pass
    missing = wanted - found.keys()
    if missing:
        raise ValueError(
            f"Could not read Pietro's sport rules from {notebook_path}: "
            + ", ".join(sorted(missing))
        )
    return found


def make_event_labeller(notebook_path: Path):
    """Build the notebook's event-slug-first labeler from its current rule lists."""
    rules = notebook_rule_assignments(notebook_path)
    combined: dict[str, list[str]] = {}
    for label, pattern in rules["sport_rules"]:
        combined.setdefault(label, []).append(f"(?:{pattern})")
    text_rules = [
        (label, re.compile("|".join(patterns), re.IGNORECASE))
        for label, patterns in combined.items()
    ]
    prefix_rules = [
        (label, re.compile(pattern, re.IGNORECASE))
        for label, pattern in rules["slug_prefix_rules"]
    ]
    override_rules = [
        (label, re.compile(pattern, re.IGNORECASE))
        for label, pattern in rules["EVENT_OVERRIDE_RULES"]
    ]

    def first_match(compiled_rules, text: str):
        return next((label for label, regex in compiled_rules if regex.search(text)), None)

    def label_event(title, slug):
        title, slug = str(title or ""), str(slug or "")
        label = first_match(override_rules, slug)
        if label:
            return label
        label = first_match(prefix_rules, slug)
        if label:
            return label
        head = re.split(r"[-_]", slug, maxsplit=1)[0]
        if head and len(head) <= 4:
            label = first_match(text_rules, head)
            if label:
                return label
        return first_match(text_rules, f"{title} {slug}") or UNCLASSIFIED

    return label_event


def event_sports_for_forecasts(
    forecasts: pd.DataFrame,
    events_dir: Path,
    notebook_path: Path,
) -> tuple[dict[str, str], int, int]:
    """Join forecast market IDs to saved event slugs and apply Pietro's event labeler."""
    links_path = events_dir / "market_events.jsonl"
    events_path = events_dir / "events.jsonl"
    for path in (links_path, events_path, notebook_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    target_market_ids = set(forecasts["market_id"].dropna().astype(str))
    market_to_event: dict[str, str] = {}
    with links_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            market_id = str(record.get("id", ""))
            event_ids = record.get("event_ids") or []
            if market_id in target_market_ids and event_ids:
                market_to_event.setdefault(market_id, str(event_ids[0]))

    needed_event_ids = set(market_to_event.values())
    label_event = make_event_labeller(notebook_path)
    event_sports: dict[str, str] = {}
    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_id = str(event.get("id", ""))
            if event_id in needed_event_ids:
                event_sports[event_id] = label_event(event.get("title"), event.get("slug"))

    market_sports = {
        market_id: event_sports.get(event_id, UNCLASSIFIED)
        for market_id, event_id in market_to_event.items()
    }
    return market_sports, len(market_to_event), len(event_sports)


def plot_sport(
    sport: str,
    sport_data: pd.DataFrame,
    n_bins: int,
    output_dir: Path,
    exclude_final_hour: bool = False,
) -> dict[str, object]:
    """Save the three volume-tier calibration panels for one sport."""
    # Assign tiers once per market, so a market that visits multiple
    # probability bins remains in the same volume tier everywhere.
    sport_data = sport_data.copy()
    market_volumes = sport_data.drop_duplicates("market_id")[["market_id", "volumeNum"]].copy()
    market_volumes["volume_tier"] = pd.qcut(
        market_volumes["volumeNum"].rank(method="first"),
        q=3,
        labels=VOLUME_TIERS,
    )
    sport_data["volume_tier"] = sport_data["market_id"].map(
        market_volumes.set_index("market_id")["volume_tier"]
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.8), sharex=True, sharey=True, constrained_layout=True
    )
    grid = np.linspace(0, 1, 101)
    summaries = []

    for ax, tier in zip(axes, VOLUME_TIERS):
        tier_data = sport_data.loc[sport_data["volume_tier"].eq(tier)]
        tier_markets = market_volumes.loc[market_volumes["volume_tier"].eq(tier)]
        bins = calibration_bins(tier_data, n_bins)
        regression = weighted_regression(bins)

        if regression is not None:
            intercept, slope = regression
            ax.plot(
                grid,
                intercept + slope * grid,
                color="#c06b32",
                label="Weighted regression",
            )
            slope_label = f"{slope:.2f}"
        else:
            ax.plot([], [], color="#c06b32", label="Regression unavailable")
            slope_label = "n/a"

        ax.plot([0, 1], [0, 1], "--", color="0.45", label="Perfect calibration")
        yerr = np.maximum(
            np.vstack(
                [
                    bins["observed_rate"].to_numpy() - bins["ci_low"].to_numpy(),
                    bins["ci_high"].to_numpy() - bins["observed_rate"].to_numpy(),
                ]
            ),
            0,
        )
        ax.errorbar(
            bins["mean_probability"],
            bins["observed_rate"],
            yerr=yerr,
            fmt="none",
            ecolor="#3569a8",
            elinewidth=1.3,
            capsize=4,
            label="Wilson 95% CI",
            zorder=3,
        )
        ax.scatter(
            bins["mean_probability"],
            bins["observed_rate"],
            s=20,
            color="#3569a8",
            edgecolor="white",
            linewidth=0.5,
            label=f"{100 // n_bins}% bin mean",
            zorder=4,
        )
        for _, row in bins.iterrows():
            probability = float(row["mean_probability"])
            x_offset = -5 if probability > 0.85 else 3
            horizontal_alignment = "right" if x_offset < 0 else "left"
            ax.annotate(
                f"n={int(row['n']):,}",
                (probability, row["observed_rate"]),
                xytext=(x_offset, 3),
                textcoords="offset points",
                ha=horizontal_alignment,
                fontsize=7,
            )

        volume_min = float(tier_markets["volumeNum"].min())
        volume_max = float(tier_markets["volumeNum"].max())
        ax.set(
            title=(
                f"{tier} (n={len(tier_markets):,}; slope={slope_label})\n"
                f"volume {volume_min:,.0f}–{volume_max:,.0f}"
            ),
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Predicted probability",
        )
        ax.grid(alpha=0.2)
        summaries.append(
            {
                "sport": sport,
                "volume_tier": tier,
                "markets": len(tier_markets),
                "market_bin_observations": len(tier_data),
                "volume_min": volume_min,
                "volume_median": float(tier_markets["volumeNum"].median()),
                "volume_max": volume_max,
            }
        )

    axes[0].set_ylabel("Observed event frequency")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        frameon=True,
    )
    hour_note = " (hour 1 excluded)" if exclude_final_hour else ""
    fig.suptitle(f"{sport}: calibration by lifetime-volume tier{hour_note}", y=1.14)
    output_path = output_dir / f"{safe_filename(sport)}_volume_tertiles.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"output": str(output_path), "summary": summaries}


def plot_sport_all_volumes(
    sport: str,
    sport_data: pd.DataFrame,
    n_bins: int,
    output_dir: Path,
    exclude_final_hour: bool = False,
) -> dict[str, object]:
    """Save one calibration plot using every valid forecast, irrespective of volume."""
    bins = calibration_bins(sport_data, n_bins)
    regression = weighted_regression(bins)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    grid = np.linspace(0, 1, 101)
    if regression is not None:
        intercept, slope = regression
        ax.plot(
            grid,
            intercept + slope * grid,
            color="#c06b32",
            label=f"Weighted regression; slope={slope:.2f}",
        )
        slope_label = f"{slope:.2f}"
    else:
        ax.plot([], [], color="#c06b32", label="Regression unavailable")
        slope_label = "n/a"

    ax.plot([0, 1], [0, 1], "--", color="0.45", label="Perfect calibration")
    yerr = np.maximum(
        np.vstack(
            [
                bins["observed_rate"].to_numpy() - bins["ci_low"].to_numpy(),
                bins["ci_high"].to_numpy() - bins["observed_rate"].to_numpy(),
            ]
        ),
        0,
    )
    ax.errorbar(
        bins["mean_probability"],
        bins["observed_rate"],
        yerr=yerr,
        fmt="none",
        ecolor="#3569a8",
        elinewidth=1.3,
        capsize=4,
        label="Wilson 95% CI",
        zorder=3,
    )
    ax.scatter(
        bins["mean_probability"],
        bins["observed_rate"],
        s=28,
        color="#3569a8",
        edgecolor="white",
        linewidth=0.5,
        label=f"{100 // n_bins}% bin mean",
        zorder=4,
    )
    for _, row in bins.iterrows():
        probability = float(row["mean_probability"])
        x_offset = -5 if probability > 0.85 else 3
        ax.annotate(
            f"n={int(row['n']):,}",
            (probability, row["observed_rate"]),
            xytext=(x_offset, 3),
            textcoords="offset points",
            ha="right" if x_offset < 0 else "left",
            fontsize=8,
        )

    market_count = sport_data["market_id"].nunique()
    hour_note = "; hour 1 excluded" if exclude_final_hour else "; hours 1–24"
    ax.set(
        title=(
            f"{sport}: calibration across all volume levels "
            f"(n={market_count:,}; slope={slope_label}{hour_note})"
        ),
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Predicted probability",
        ylabel="Observed event frequency",
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", frameon=True)
    output_path = output_dir / f"{safe_filename(sport)}_all_volumes.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "output": str(output_path),
        "summary": {
            "sport": sport,
            "markets": market_count,
            "market_bin_observations": len(sport_data),
            "volume_missing": int(sport_data.drop_duplicates("market_id")["volumeNum"].isna().sum()),
            "slope": slope_label,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument(
        "--exclude-final-hour",
        action="store_true",
        help="Use all hourly forecasts from bins 2–24, excluding the hour closest to market close.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overall-output-dir", type=Path, default=None)
    parser.add_argument("--events-dir", type=Path, default=DEFAULT_EVENTS_DIR)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--top-sports", type=int, default=7)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    if args.top_sports < 1:
        parser.error("--top-sports must be at least 1")
    if not 2 <= args.bins <= 100:
        parser.error("--bins must be between 2 and 100")
    if args.output_dir is None:
        args.output_dir = (
            DEFAULT_NO_FINAL_HOUR_OUTPUT_DIR if args.exclude_final_hour else DEFAULT_OUTPUT_DIR
        )
    if args.overall_output_dir is None:
        args.overall_output_dir = (
            DEFAULT_NO_FINAL_HOUR_OVERALL_DIR if args.exclude_final_hour else DEFAULT_OVERALL_OUTPUT_DIR
        )

    if not args.master.is_file():
        parser.error(f"Hourly master CSV not found: {args.master}")
    start_hour = 2 if args.exclude_final_hour else 1
    data = hourly_market_bins(args.master, n_bins=args.bins, start_hour=start_hour)
    if data.duplicated(["market_id", "probability_bin"]).any():
        raise ValueError("Duplicate market and probability-bin pairs found in the master CSV")
    print(f"Used hourly bins {start_hour:02d}–24; each market is counted once per probability bin.")

    market_sports, linked_markets, matched_events = event_sports_for_forecasts(
        data, args.events_dir, args.notebook
    )
    data["sport"] = data["market_id"].astype("string").map(market_sports).fillna(UNCLASSIFIED)

    for column in ("probability", "actual", "volumeNum"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.loc[
        data["probability"].between(0, 1)
        & data["actual"].isin([0, 1])
        & data["sport"].notna()
        & data["sport"].astype(str).str.strip().ne("")
    ].copy()
    volume_data = data.loc[data["volumeNum"].ge(0)].copy()

    # Use Pietro's event-slug-first sport label, not the older market-question classifier.
    market_counts = data.loc[data["sport"].ne(UNCLASSIFIED)].groupby("sport")["market_id"].nunique()
    top_sports = market_counts.nlargest(args.top_sports).index.tolist()
    if not top_sports:
        parser.error("No classified sports found among the hourly observations.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.overall_output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    overall_summaries = []
    print(f"Using saved event slugs from {args.events_dir}; sport rules from {args.notebook}")
    print(f"Forecast markets linked to events: {linked_markets:,}; event records found: {matched_events:,}")
    print(
        f"Valid markets: {data['market_id'].nunique():,}; market-bin observations: {len(data):,}; "
        f"markets with usable volume: {volume_data['market_id'].nunique():,}"
    )
    print("Sports selected by market count:", ", ".join(top_sports))
    for sport in top_sports:
        sport_all = data.loc[data["sport"].eq(sport)]
        overall_result = plot_sport_all_volumes(
            sport,
            sport_all,
            args.bins,
            args.overall_output_dir,
            exclude_final_hour=args.exclude_final_hour,
        )
        overall_summaries.append(overall_result["summary"])
        print(f"Saved {overall_result['output']}")

        result = plot_sport(
            sport,
            volume_data.loc[volume_data["sport"].eq(sport)],
            args.bins,
            args.output_dir,
            exclude_final_hour=args.exclude_final_hour,
        )
        all_summaries.extend(result["summary"])
        print(f"Saved {result['output']}")

    summary_path = args.output_dir / "volume_tier_summary.csv"
    pd.DataFrame(all_summaries).to_csv(summary_path, index=False)
    print(f"Saved volume-tier summary: {summary_path}")
    overall_summary_path = args.overall_output_dir / "all_volume_summary.csv"
    pd.DataFrame(overall_summaries).to_csv(overall_summary_path, index=False)
    print(f"Saved all-volume summary: {overall_summary_path}")


if __name__ == "__main__":
    main()
