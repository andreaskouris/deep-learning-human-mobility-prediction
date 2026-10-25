from __future__ import annotations

# Support direct script execution without depending on the caller PYTHONPATH.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

ROOT = Path("final_analysis/tte")

PER_SAMPLE_PATH = (
    ROOT
    / "per_sample"
    / "tte_per_sample.csv"
)

MODEL_STATS_PATH = (
    ROOT
    / "statistics"
    / "tte_model_error_statistics.csv"
)

PAIRWISE_STATS_PATH = (
    ROOT
    / "statistics"
    / "tte_pairwise_statistics.csv"
)

STRATIFIED_PAIRWISE_PATH = (
    ROOT
    / "stratified"
    / "tte_stratified_pairwise_improvements.csv"
)

OUTPUT_DIR = (
    ROOT
    / "figures"
)

BOOTSTRAP_REPETITIONS = 2000

BOOTSTRAP_CHUNK_SIZE = 100

BOOTSTRAP_SEED = 2021

EXPECTED_SAMPLES = 76923

MODELS = {
    "baseline": {
        "label": "Baseline",
        "marker": "o",
        "abs_error":
            "abs_error_baseline_seconds",
        "ape":
            "ape_baseline_percent",
    },

    "structured": {
        "label": "Structured",
        "marker": "s",
        "abs_error":
            "abs_error_structured_seconds",
        "ape":
            "ape_structured_percent",
    },

    "qwen": {
        "label": "Qwen",
        "marker": "^",
        "abs_error":
            "abs_error_qwen_seconds",
        "ape":
            "ape_qwen_percent",
    },
}

DURATION_EDGES = [
    -np.inf,
    200,
    335,
    495,
    731,
    np.inf,
]

DURATION_LABELS = [
    "D1 <=200s",
    "D2 201-335s",
    "D3 336-495s",
    "D4 496-731s",
    "D5 >=732s",
]

DURATION_DISPLAY = [
    "≤200 s\n(n=15,443)",
    "201–335 s\n(n=15,383)",
    "336–495 s\n(n=15,408)",
    "496–731 s\n(n=15,339)",
    "≥732 s\n(n=15,350)",
]

def save_all(
    fig,
    stem,
):

    for extension in [
        "png",
        "svg",
        "pdf",
    ]:

        path = (
            OUTPUT_DIR
            / f"{stem}.{extension}"
        )

        if extension == "png":

            fig.savefig(
                path,
                dpi=300,
                bbox_inches="tight",
            )

        else:

            fig.savefig(
                path,
                bbox_inches="tight",
            )

        print(
            "Saved:",
            path,
        )

def paired_bootstrap_mean_ci(
    differences,
    *,
    seed,
):

    """
    differences:
        N x K matrix of paired per-trip
        improvements.

    Positive = comparison model is better.

    Returns 95% percentile bootstrap CI
    of the mean difference.
    """

    differences = np.asarray(
        differences,
        dtype=np.float64,
    )

    if differences.ndim == 1:

        differences = (
            differences[:, None]
        )

    n, metric_count = (
        differences.shape
    )

    rng = np.random.default_rng(
        seed
    )

    bootstrap_means = np.empty(
        (
            BOOTSTRAP_REPETITIONS,
            metric_count,
        ),
        dtype=np.float64,
    )


    for start in range(
        0,
        BOOTSTRAP_REPETITIONS,
        BOOTSTRAP_CHUNK_SIZE,
    ):

        count = min(
            BOOTSTRAP_CHUNK_SIZE,
            BOOTSTRAP_REPETITIONS
            - start,
        )

        indices = rng.integers(
            0,
            n,
            size=(
                count,
                n,
            ),
            dtype=np.int32,
        )

        for metric_index in range(
            metric_count
        ):

            bootstrap_means[
                start:
                start + count,
                metric_index,
            ] = (
                differences[
                    indices,
                    metric_index,
                ]
                .mean(
                    axis=1
                )
            )


    low = np.quantile(
        bootstrap_means,
        0.025,
        axis=0,
    )

    high = np.quantile(
        bootstrap_means,
        0.975,
        axis=0,
    )

    return (
        low.astype(float),
        high.astype(float),
    )

def plot_duration_improvement(
    *,
    point_column,
    low_column,
    high_column,
    ylabel,
    title,
    stem,
):

    fig, ax = plt.subplots(
        figsize=(9.0, 5.2)
    )


    x = np.arange(
        len(
            DURATION_LABELS
        )
    )


    offsets = {
        "structured": -0.08,
        "qwen": 0.08,
    }


    all_low = []
    all_high = []


    for model in [
        "structured",
        "qwen",
    ]:

        rows = (
            duration_bootstrap[
                duration_bootstrap[
                    "model"
                ]
                == model
            ]
            .set_index(
                "duration_group"
            )
            .loc[
                DURATION_LABELS
            ]
            .reset_index()
        )


        point = (
            rows[
                point_column
            ]
            .to_numpy(
                dtype=float
            )
        )

        low = (
            rows[
                low_column
            ]
            .to_numpy(
                dtype=float
            )
        )

        high = (
            rows[
                high_column
            ]
            .to_numpy(
                dtype=float
            )
        )


        all_low.extend(
            low.tolist()
        )

        all_high.extend(
            high.tolist()
        )


        ax.errorbar(
            x
            + offsets[
                model
            ],
            point,
            yerr=np.vstack(
                [
                    point - low,
                    high - point,
                ]
            ),
            fmt=
                MODELS[
                    model
                ]["marker"]
                + "-",
            markersize=7,
            linewidth=1.5,
            capsize=4,
            label=
                MODELS[
                    model
                ]["label"],
        )


    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )


    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        DURATION_DISPLAY
    )


    ax.set_xlabel(
        "Actual trip duration"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
        loc="best",
    )


    ymin = min(
        min(all_low),
        0.0,
    )

    ymax = max(
        max(all_high),
        0.0,
    )

    padding = (
        ymax - ymin
    ) * 0.10

    if padding == 0:
        padding = 0.5


    ax.set_ylim(
        ymin - padding,
        ymax + padding,
    )


    fig.tight_layout()


    save_all(
        fig,
        stem,
    )

    plt.close(
        fig
    )

HISTORICAL = True


def main(argv=None):
    global HISTORICAL, a, all_errors, all_high, all_low, ax, axes, b, baseline_abs, baseline_ape, column, comparison_model, comparison_order, cumulative_percent, decimals, differences, display_xmax, duration_bootstrap, duration_bootstrap_path, duration_label, duration_locked, duration_rows, errors, fig, forest_metrics, group, high, index, key, locked, low, mae_difference, mae_point, mape_difference, mape_point, metric_name, metric_panels, model, model_a, model_abs, model_ape, model_b, model_order, model_stats, padding, pairwise_stats, per_sample, point, pooled_errors, row, seed_counter, sorted_errors, spec, stratified_pairs, value, value_range, values, xlabel, xmax, xmin, y
    from scripts.portable_runtime_support import analysis_cli
    analysis_cli(globals(), "tte", argv)

    per_sample = pd.read_csv(
        PER_SAMPLE_PATH
    )

    model_stats = pd.read_csv(
        MODEL_STATS_PATH
    )

    pairwise_stats = pd.read_csv(
        PAIRWISE_STATS_PATH
    )

    stratified_pairs = pd.read_csv(
        STRATIFIED_PAIRWISE_PATH
    )

    if len(per_sample) != EXPECTED_SAMPLES:
        raise RuntimeError(
            "Unexpected TTE per-sample count."
        )

    if (
        per_sample["sample_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate TTE sample IDs."
        )

    if set(
        model_stats["model"]
    ) != {
        "baseline",
        "structured",
        "qwen",
    }:
        raise RuntimeError(
            "Unexpected model-level statistics."
        )

    if len(
        pairwise_stats
    ) != 3:
        raise RuntimeError(
            "Expected exactly three pairwise comparisons."
        )

    duration_locked = stratified_pairs[
        stratified_pairs[
            "stratum_type"
        ]
        == "duration_group"
    ].copy()

    if len(duration_locked) != 15:
        raise RuntimeError(
            "Expected 15 locked duration pairwise rows."
        )

    print(
        "Input validation: PASS"
    )

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 120,
    })

    model_order = [
        "baseline",
        "structured",
        "qwen",
    ]

    metric_panels = [
        (
            "MAE",
            "mae_seconds",
            "MAE (seconds) ↓",
            2,
        ),
        (
            "RMSE",
            "rmse_seconds",
            "RMSE (seconds) ↓",
            2,
        ),
        (
            "MAPE",
            "mape_percent",
            "MAPE (%) ↓",
            2,
        ),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.8, 4.5),
    )

    for ax, (
        metric_name,
        column,
        xlabel,
        decimals,
    ) in zip(
        axes,
        metric_panels,
    ):

        y = np.arange(
            len(
                model_order
            )
        )

        values = []


        for index, model in enumerate(
            model_order
        ):

            row = model_stats[
                model_stats[
                    "model"
                ]
                == model
            ].iloc[0]

            value = float(
                row[column]
            )

            values.append(
                value
            )

            ax.scatter(
                value,
                index,
                s=75,
                marker=
                    MODELS[
                        model
                    ]["marker"],
                zorder=3,
            )

            ax.annotate(
                f"{value:.{decimals}f}",
                xy=(value, index),
                xytext=(8, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
            )


        ax.set_yticks(
            y
        )

        ax.set_yticklabels(
            [
                MODELS[
                    model
                ]["label"]
                for model in model_order
            ]
        )

        ax.invert_yaxis()

        ax.set_xlabel(
            xlabel
        )

        ax.set_title(
            metric_name
        )

        ax.grid(
            axis="x",
            alpha=0.25,
        )


        value_range = (
            max(values)
            - min(values)
        )

        padding = max(
            value_range * 0.55,
            0.6,
        )

        ax.set_xlim(
            min(values)
            - padding,
            max(values)
            + padding,
        )

    fig.suptitle(
        "Travel-Time Estimation: Aggregate Test Performance",
        fontsize=15,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.93)
    )

    save_all(
        fig,
        "figure_5_5_tte_aggregate_performance",
    )

    plt.close(
        fig
    )

    comparison_order = [
        (
            "Baseline",
            "Structured",
        ),
        (
            "Baseline",
            "Qwen",
        ),
        (
            "Structured",
            "Qwen",
        ),
    ]

    forest_metrics = [
        {
            "title":
                "MAE",

            "point":
                "mae_improvement_seconds",

            "low":
                "mae_ci95_low_seconds",

            "high":
                "mae_ci95_high_seconds",

            "xlabel":
                "MAE improvement (seconds)",
        },

        {
            "title":
                "RMSE",

            "point":
                "rmse_improvement_seconds",

            "low":
                "rmse_ci95_low_seconds",

            "high":
                "rmse_ci95_high_seconds",

            "xlabel":
                "RMSE improvement (seconds)",
        },

        {
            "title":
                "MAPE",

            "point":
                "mape_improvement_pp",

            "low":
                "mape_ci95_low_pp",

            "high":
                "mape_ci95_high_pp",

            "xlabel":
                "MAPE improvement (percentage points)",
        },
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.2, 5.0),
    )

    for ax, spec in zip(
        axes,
        forest_metrics,
    ):

        all_low = []
        all_high = []


        for y, (
            model_a,
            model_b,
        ) in enumerate(
            comparison_order
        ):

            row = pairwise_stats[
                (
                    pairwise_stats[
                        "model_a"
                    ]
                    == model_a
                )
                &
                (
                    pairwise_stats[
                        "model_b"
                    ]
                    == model_b
                )
            ]

            if len(row) != 1:

                raise RuntimeError(
                    "Could not identify "
                    f"{model_a} -> {model_b}."
                )

            row = row.iloc[0]


            point = float(
                row[
                    spec["point"]
                ]
            )

            low = float(
                row[
                    spec["low"]
                ]
            )

            high = float(
                row[
                    spec["high"]
                ]
            )


            all_low.append(
                low
            )

            all_high.append(
                high
            )


            ax.errorbar(
                point,
                y,
                xerr=np.array(
                    [
                        [
                            point - low
                        ],
                        [
                            high - point
                        ],
                    ]
                ),
                fmt=MODELS[
                    [
                        key
                        for key, value
                        in MODELS.items()
                        if value["label"]
                        == model_b
                    ][0]
                ]["marker"],
                markersize=7,
                capsize=4,
                linewidth=1.4,
            )


            ax.text(
                point,
                y - 0.20,
                f"{point:+.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )


        ax.axvline(
            0.0,
            linestyle="--",
            linewidth=1.0,
        )


        ax.set_yticks(
            np.arange(
                len(
                    comparison_order
                )
            )
        )


        ax.set_yticklabels(
            [
                f"{a} → {b}"
                for a, b
                in comparison_order
            ]
        )


        ax.invert_yaxis()

        ax.set_title(
            spec["title"]
        )

        ax.set_xlabel(
            spec["xlabel"]
        )

        ax.grid(
            axis="x",
            alpha=0.25,
        )


        xmin = min(
            min(all_low),
            0.0,
        )

        xmax = max(
            max(all_high),
            0.0,
        )

        padding = (
            xmax - xmin
        ) * 0.15

        if padding == 0:
            padding = 0.25


        ax.set_xlim(
            xmin - padding,
            xmax + padding,
        )

    fig.suptitle(
        (
            "Travel-Time Estimation: "
            "Paired Test-Set Differences with 95% Bootstrap CI\n"
            "Positive values indicate lower error for model B"
        ),
        fontsize=14,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.90)
    )

    save_all(
        fig,
        "figure_5_6_tte_paired_bootstrap_ci",
    )

    plt.close(
        fig
    )

    per_sample[
        "duration_group"
    ] = pd.cut(
        per_sample[
            "target_seconds"
        ],
        bins=
            DURATION_EDGES,
        labels=
            DURATION_LABELS,
    )

    if (
        per_sample[
            "duration_group"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Missing duration-group assignments."
        )

    duration_rows = []

    seed_counter = 0

    for duration_label in (
        DURATION_LABELS
    ):

        group = per_sample[
            per_sample[
                "duration_group"
            ]
            == duration_label
        ]


        baseline_abs = (
            group[
                MODELS[
                    "baseline"
                ]["abs_error"]
            ]
            .to_numpy(
                dtype=np.float64
            )
        )


        baseline_ape = (
            group[
                MODELS[
                    "baseline"
                ]["ape"]
            ]
            .to_numpy(
                dtype=np.float64
            )
        )


        for comparison_model in [
            "structured",
            "qwen",
        ]:

            model_abs = (
                group[
                    MODELS[
                        comparison_model
                    ]["abs_error"]
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )


            model_ape = (
                group[
                    MODELS[
                        comparison_model
                    ]["ape"]
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )


            mae_difference = (
                baseline_abs
                - model_abs
            )


            mape_difference = (
                baseline_ape
                - model_ape
            )


            differences = np.column_stack(
                [
                    mae_difference,
                    mape_difference,
                ]
            )


            low, high = (
                paired_bootstrap_mean_ci(
                    differences,
                    seed=(
                        BOOTSTRAP_SEED
                        + seed_counter
                    ),
                )
            )

            seed_counter += 1


            mae_point = float(
                mae_difference.mean()
            )

            mape_point = float(
                mape_difference.mean()
            )


            
            # Cross-check against locked descriptive file
            

            locked = duration_locked[
                (
                    duration_locked[
                        "stratum"
                    ]
                    == duration_label
                )
                &
                (
                    duration_locked[
                        "model_a"
                    ]
                    == "baseline"
                )
                &
                (
                    duration_locked[
                        "model_b"
                    ]
                    == comparison_model
                )
            ]


            if len(locked) != 1:

                raise RuntimeError(
                    "Missing locked duration result: "
                    f"{duration_label}, "
                    f"{comparison_model}"
                )


            locked = locked.iloc[0]


            if not np.isclose(
                mae_point,
                float(
                    locked[
                        "mae_improvement_seconds"
                    ]
                ),
                atol=1e-8,
            ):

                raise RuntimeError(
                    "Duration MAE reproduction failure."
                )


            if not np.isclose(
                mape_point,
                float(
                    locked[
                        "mape_improvement_pp"
                    ]
                ),
                atol=1e-8,
            ):

                raise RuntimeError(
                    "Duration MAPE reproduction failure."
                )


            duration_rows.append(
                {
                    "duration_group":
                        duration_label,

                    "model":
                        comparison_model,

                    "label":
                        MODELS[
                            comparison_model
                        ]["label"],

                    "n":
                        len(group),

                    "mae_improvement_seconds":
                        mae_point,

                    "mae_ci95_low_seconds":
                        float(
                            low[0]
                        ),

                    "mae_ci95_high_seconds":
                        float(
                            high[0]
                        ),

                    "mape_improvement_pp":
                        mape_point,

                    "mape_ci95_low_pp":
                        float(
                            low[1]
                        ),

                    "mape_ci95_high_pp":
                        float(
                            high[1]
                        ),
                }
            )

    duration_bootstrap = pd.DataFrame(
        duration_rows
    )

    duration_bootstrap_path = (
        ROOT
        / "stratified"
        / "tte_duration_group_bootstrap_ci.csv"
    )

    duration_bootstrap.to_csv(
        duration_bootstrap_path,
        index=False,
    )

    print(
        "Duration bootstrap analysis: PASS"
    )

    print(
        "Saved:",
        duration_bootstrap_path,
    )

    plot_duration_improvement(
        point_column=
            "mae_improvement_seconds",

        low_column=
            "mae_ci95_low_seconds",

        high_column=
            "mae_ci95_high_seconds",

        ylabel=
            (
                "MAE improvement over Baseline "
                "(seconds)"
            ),

        title=
            (
                "Travel-Time Estimation: "
                "MAE Improvement by Trip Duration"
            ),

        stem=
            "figure_5_7_tte_mae_improvement_by_duration",
    )

    plot_duration_improvement(
        point_column=
            "mape_improvement_pp",

        low_column=
            "mape_ci95_low_pp",

        high_column=
            "mape_ci95_high_pp",

        ylabel=
            (
                "MAPE improvement over Baseline "
                "(percentage points)"
            ),

        title=
            (
                "Travel-Time Estimation: "
                "MAPE Improvement by Trip Duration"
            ),

        stem=
            "figure_5_8_tte_mape_improvement_by_duration",
    )

    fig, ax = plt.subplots(
        figsize=(8.8, 5.2)
    )

    all_errors = []

    for model in model_order:

        errors = (
            per_sample[
                MODELS[
                    model
                ]["abs_error"]
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        all_errors.append(
            errors
        )

    pooled_errors = np.concatenate(
        all_errors
    )

    display_xmax = float(
        np.quantile(
            pooled_errors,
            0.99,
        )
    )

    for model, errors in zip(
        model_order,
        all_errors,
    ):

        sorted_errors = np.sort(
            errors
        )

        cumulative_percent = (
            np.arange(
                1,
                len(
                    sorted_errors
                )
                + 1,
            )
            / len(
                sorted_errors
            )
            * 100.0
        )


        ax.plot(
            sorted_errors,
            cumulative_percent,
            linewidth=1.7,
            label=
                MODELS[
                    model
                ]["label"],
        )

    ax.set_xlim(
        0,
        display_xmax,
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_xlabel(
        (
            "Absolute prediction error (seconds) "
            "— x-axis shown to pooled P99"
        )
    )

    ax.set_ylabel(
        "Trips with absolute error ≤ x (%)"
    )

    ax.set_title(
        (
            "Travel-Time Estimation: "
            "Absolute-Error Distribution"
        )
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
        loc="lower right",
    )

    fig.tight_layout()

    save_all(
        fig,
        "figure_5_9_tte_absolute_error_ecdf",
    )

    plt.close(
        fig
    )

    print()

    print(
        "============================================"
    )

    print(
        " FINAL TTE FIGURES GENERATED"
    )

    print(
        "============================================"
    )

    print(
        "Figure 5.5: Aggregate performance"
    )

    print(
        "Figure 5.6: Paired bootstrap differences"
    )

    print(
        "Figure 5.7: MAE improvement by duration"
    )

    print(
        "Figure 5.8: MAPE improvement by duration"
    )

    print(
        "Figure 5.9: Absolute-error ECDF"
    )

    print()

    print(
        "STATUS: ALL FINAL TTE FIGURES GENERATED"
    )


if __name__ == "__main__":
    main()
