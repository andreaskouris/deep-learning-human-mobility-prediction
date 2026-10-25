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

ROOT = Path("final_analysis/next_hop")

MODEL_METRICS_PATH = (
    ROOT
    / "statistics"
    / "next_hop_model_metrics.csv"
)

PAIRWISE_PATH = (
    ROOT
    / "statistics"
    / "next_hop_pairwise_statistics.csv"
)

OUTPUT_DIR = (
    ROOT
    / "figures"
)

def save_all(
    fig,
    stem,
):

    png = (
        OUTPUT_DIR
        / f"{stem}.png"
    )

    svg = (
        OUTPUT_DIR
        / f"{stem}.svg"
    )

    pdf = (
        OUTPUT_DIR
        / f"{stem}.pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        svg,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    print(
        "Saved:",
        png,
    )

    print(
        "Saved:",
        svg,
    )

    print(
        "Saved:",
        pdf,
    )

HISTORICAL = True


def main(argv=None):
    global HISTORICAL, _, all_values, ax, base_y, column, comparison, comparison_index, comparison_order, current_y, delta, expected_labels, fig, forest, high, low, lower, metric, metric_definitions, metric_specs, metrics, model, model_a, model_b, model_markers, model_order, name, offsets, pairwise, row, rows, subset, upper, values, x, xmax, xmin, y, y_labels, y_positions, y_value
    from scripts.portable_runtime_support import analysis_cli
    analysis_cli(globals(), "next_hop", argv)

    metrics = pd.read_csv(
        MODEL_METRICS_PATH
    )

    pairwise = pd.read_csv(
        PAIRWISE_PATH
    )

    expected_labels = {
        "Baseline",
        "POI TF-IDF",
        "Qwen POI",
    }

    if set(
        metrics["label"]
    ) != expected_labels:

        raise RuntimeError(
            "Unexpected model labels."
        )

    if len(
        pairwise
    ) != 3:

        raise RuntimeError(
            "Expected exactly 3 pairwise comparisons."
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

    metric_definitions = [
        (
            "Accuracy",
            "accuracy_percent",
        ),
        (
            "MRR@5",
            "mrr_at_5_percent",
        ),
        (
            "NDCG@5",
            "ndcg_at_5_percent",
        ),
    ]

    model_order = [
        "Baseline",
        "POI TF-IDF",
        "Qwen POI",
    ]

    model_markers = {
        "Baseline": "o",
        "POI TF-IDF": "s",
        "Qwen POI": "^",
    }

    fig, ax = plt.subplots(
        figsize=(8.8, 4.8)
    )

    base_y = np.arange(
        len(
            metric_definitions
        )
    )

    offsets = {
        "Baseline": -0.18,
        "POI TF-IDF": 0.00,
        "Qwen POI": 0.18,
    }

    for model in model_order:

        row = metrics[
            metrics["label"]
            == model
        ].iloc[0]

        values = [
            float(
                row[column]
            )
            for _, column
            in metric_definitions
        ]

        y = (
            base_y
            + offsets[
                model
            ]
        )

        ax.scatter(
            values,
            y,
            s=70,
            marker=
                model_markers[
                    model
                ],
            label=model,
            zorder=3,
        )

        for x, y_value in zip(
            values,
            y,
        ):

            ax.text(
                x + 0.08,
                y_value,
                f"{x:.2f}",
                va="center",
                fontsize=9,
            )

    ax.set_yticks(
        base_y
    )

    ax.set_yticklabels(
        [
            name
            for name, _
            in metric_definitions
        ]
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Performance (%)"
    )

    ax.set_title(
        "Next-Hop Prediction: Aggregate Test Performance"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    all_values = []

    for _, column in (
        metric_definitions
    ):

        all_values.extend(
            metrics[
                column
            ]
            .astype(float)
            .tolist()
        )

    lower = (
        min(
            all_values
        )
        - 1.2
    )

    upper = (
        max(
            all_values
        )
        + 1.2
    )

    ax.set_xlim(
        lower,
        upper,
    )

    fig.tight_layout(
        rect=(0, 0, 0.84, 1)
    )

    save_all(
        fig,
        "figure_5_1_next_hop_aggregate_performance",
    )

    plt.close(
        fig
    )

    comparison_order = [
        (
            "Baseline",
            "POI TF-IDF",
        ),
        (
            "Baseline",
            "Qwen POI",
        ),
        (
            "POI TF-IDF",
            "Qwen POI",
        ),
    ]

    metric_specs = [
        {
            "name":
                "Accuracy",

            "delta":
                "delta_accuracy_pp",

            "low":
                "delta_accuracy_ci95_low_pp",

            "high":
                "delta_accuracy_ci95_high_pp",

            "marker":
                "o",
        },

        {
            "name":
                "MRR@5",

            "delta":
                "delta_mrr_at_5_pp",

            "low":
                "delta_mrr_ci95_low_pp",

            "high":
                "delta_mrr_ci95_high_pp",

            "marker":
                "s",
        },

        {
            "name":
                "NDCG@5",

            "delta":
                "delta_ndcg_at_5_pp",

            "low":
                "delta_ndcg_ci95_low_pp",

            "high":
                "delta_ndcg_ci95_high_pp",

            "marker":
                "^",
        },
    ]

    rows = []

    for comparison_index, (
        model_a,
        model_b,
    ) in enumerate(
        comparison_order
    ):

        row = pairwise[
            (
                pairwise[
                    "model_a"
                ]
                == model_a
            )
            &
            (
                pairwise[
                    "model_b"
                ]
                == model_b
            )
        ]

        if len(
            row
        ) != 1:

            raise RuntimeError(
                "Could not uniquely identify "
                f"{model_a} -> {model_b}."
            )

        row = row.iloc[0]

        for metric in (
            metric_specs
        ):

            delta = float(
                row[
                    metric[
                        "delta"
                    ]
                ]
            )

            low = float(
                row[
                    metric[
                        "low"
                    ]
                ]
            )

            high = float(
                row[
                    metric[
                        "high"
                    ]
                ]
            )

            rows.append({
                "comparison":
                    (
                        f"{model_a} → "
                        f"{model_b}"
                    ),

                "metric":
                    metric[
                        "name"
                    ],

                "marker":
                    metric[
                        "marker"
                    ],

                "delta":
                    delta,

                "low":
                    low,

                "high":
                    high,
            })

    forest = pd.DataFrame(
        rows
    )

    fig, ax = plt.subplots(
        figsize=(9.2, 6.2)
    )

    y_positions = []

    y_labels = []

    current_y = 0

    for comparison in [
        "Baseline → POI TF-IDF",
        "Baseline → Qwen POI",
        "POI TF-IDF → Qwen POI",
    ]:

        subset = forest[
            forest[
                "comparison"
            ]
            == comparison
        ]

        for _, row in (
            subset.iterrows()
        ):

            delta = (
                row[
                    "delta"
                ]
            )

            low = (
                row[
                    "low"
                ]
            )

            high = (
                row[
                    "high"
                ]
            )

            ax.errorbar(
                delta,
                current_y,
                xerr=np.array(
                    [
                        [
                            delta
                            - low
                        ],
                        [
                            high
                            - delta
                        ],
                    ]
                ),
                fmt=row[
                    "marker"
                ],
                markersize=7,
                capsize=3,
                linewidth=1.4,
            )

            y_positions.append(
                current_y
            )

            y_labels.append(
                (
                    f"{comparison}  |  "
                    f"{row['metric']}"
                )
            )

            current_y += 1

        current_y += 0.55

    ax.axvline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        y_labels
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Paired improvement of model B over model A (percentage points)"
    )

    ax.set_title(
        "Next-Hop Prediction: Paired Test-Set Differences with 95% Bootstrap CI"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    xmin = min(
        forest[
            "low"
        ]
    ) - 0.25

    xmax = max(
        forest[
            "high"
        ]
    ) + 0.25

    ax.set_xlim(
        xmin,
        xmax,
    )

    fig.tight_layout()

    save_all(
        fig,
        "figure_5_2_next_hop_paired_bootstrap_ci",
    )

    plt.close(
        fig
    )

    print()

    print(
        "STATUS: FIGURES 5.1 AND 5.2 GENERATED"
    )


if __name__ == "__main__":
    main()
