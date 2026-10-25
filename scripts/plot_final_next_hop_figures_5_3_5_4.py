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

INPUT_PATH = (
    ROOT
    / "stratified"
    / "next_hop_stratified_pairwise.csv"
)

OUTPUT_DIR = (
    ROOT
    / "figures"
)

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

def plot_accuracy_gain(
    *,
    source,
    ordered_strata,
    display_labels,
    title,
    xlabel,
    stem,
):

    fig, ax = plt.subplots(
        figsize=(8.8, 5.0)
    )

    x = np.arange(
        len(ordered_strata)
    )

    offsets = {
        "POI TF-IDF": -0.08,
        "Qwen POI": 0.08,
    }


    all_lows = []
    all_highs = []


    for model_label in MODEL_ORDER:

        model_rows = (
            source[
                source[
                    "model_b_label"
                ]
                == model_label
            ]
            .set_index(
                "stratum"
            )
            .loc[
                ordered_strata
            ]
            .reset_index()
        )


        delta = (
            model_rows[
                "delta_accuracy_pp"
            ]
            .to_numpy(
                dtype=float
            )
        )

        low = (
            model_rows[
                "delta_accuracy_ci95_low_pp"
            ]
            .to_numpy(
                dtype=float
            )
        )

        high = (
            model_rows[
                "delta_accuracy_ci95_high_pp"
            ]
            .to_numpy(
                dtype=float
            )
        )


        all_lows.extend(
            low.tolist()
        )

        all_highs.extend(
            high.tolist()
        )


        xerr = np.vstack(
            [
                delta - low,
                high - delta,
            ]
        )


        ax.errorbar(
            x
            + offsets[
                model_label
            ],
            delta,
            yerr=xerr,
            fmt=
                MARKERS[
                    model_label
                ]
                + "-",
            markersize=7,
            linewidth=1.5,
            capsize=4,
            label=model_label,
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
        display_labels
    )


    ax.set_ylabel(
        "Accuracy improvement over Baseline (percentage points)"
    )

    ax.set_xlabel(
        xlabel
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
        loc="upper left",
    )


    ymin = min(
        min(all_lows),
        0.0,
    )

    ymax = max(
        max(all_highs),
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
    global HISTORICAL, MARKERS, MODEL_ORDER, context_df, context_display, context_order, df, missing, plot_df, required_columns, semantic_df, semantic_display, semantic_order
    from scripts.portable_runtime_support import analysis_cli
    analysis_cli(globals(), "next_hop", argv)

    df = pd.read_csv(
        INPUT_PATH
    )

    required_columns = {
        "stratum_type",
        "stratum",
        "model_a",
        "model_b",
        "model_a_label",
        "model_b_label",
        "n",
        "delta_accuracy_pp",
        "delta_accuracy_ci95_low_pp",
        "delta_accuracy_ci95_high_pp",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    plot_df = df[
        (
            df["model_a"]
            == "nh1"
        )
        &
        (
            df["model_b"].isin(
                ["nh2", "nh3"]
            )
        )
    ].copy()

    if len(plot_df) != 18:
        raise RuntimeError(
            "Expected 18 Baseline comparison rows "
            "(4 context groups + 5 semantic groups) x 2 models."
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

    MODEL_ORDER = [
        "POI TF-IDF",
        "Qwen POI",
    ]

    MARKERS = {
        "POI TF-IDF": "s",
        "Qwen POI": "^",
    }

    semantic_order = [
        "S0 0%",
        "S1 (0,25%]",
        "S2 (25,50%]",
        "S3 (50,75%]",
        "S4 (75,100%]",
    ]

    semantic_display = [
        "0%\n(n=1,029)",
        "0–25%\n(n=6,546)",
        "25–50%\n(n=26,253)",
        "50–75%\n(n=35,088)",
        "75–100%\n(n=8,007)",
    ]

    semantic_df = plot_df[
        plot_df[
            "stratum_type"
        ]
        == "semantic_coverage"
    ].copy()

    if set(
        semantic_df[
            "stratum"
        ]
    ) != set(
        semantic_order
    ):
        raise RuntimeError(
            "Unexpected semantic coverage strata."
        )

    plot_accuracy_gain(
        source=
            semantic_df,

        ordered_strata=
            semantic_order,

        display_labels=
            semantic_display,

        title=
            (
                "Next-Hop Prediction: "
                "Accuracy Gain by Semantic POI Coverage"
            ),

        xlabel=
            (
                "Share of observed context roads "
                "with POI semantic information"
            ),

        stem=
            "figure_5_3_next_hop_accuracy_gain_semantic_coverage",
    )

    context_order = [
        "C1 5-8",
        "C2 9-16",
        "C3 17-32",
        "C4 33-63",
    ]

    context_display = [
        "5–8 roads\n(n=15,756)",
        "9–16 roads\n(n=24,969)",
        "17–32 roads\n(n=25,392)",
        "33–63 roads\n(n=10,806)",
    ]

    context_df = plot_df[
        plot_df[
            "stratum_type"
        ]
        == "context_length"
    ].copy()

    if set(
        context_df[
            "stratum"
        ]
    ) != set(
        context_order
    ):
        raise RuntimeError(
            "Unexpected context-length strata."
        )

    plot_accuracy_gain(
        source=
            context_df,

        ordered_strata=
            context_order,

        display_labels=
            context_display,

        title=
            (
                "Next-Hop Prediction: "
                "Accuracy Gain by Context Length"
            ),

        xlabel=
            "Observed trajectory context length",

        stem=
            "figure_5_4_next_hop_accuracy_gain_context_length",
    )

    print()

    print(
        "STATUS: FIGURES 5.3 AND 5.4 GENERATED"
    )


if __name__ == "__main__":
    main()
