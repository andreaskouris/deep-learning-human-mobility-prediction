from __future__ import annotations

# Support direct script execution without depending on the caller PYTHONPATH.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path

import json

import numpy as np

import pandas as pd

ROOT = Path("final_analysis/tte")

INPUT = ROOT / "per_sample" / "tte_per_sample.csv"

OUTPUT = ROOT / "stratified"

def metric_rows(
    frame,
    stratum_type,
    stratum,
):
    rows = []

    for name, config in MODELS.items():

        signed = (
            frame[config["signed"]]
            .to_numpy(dtype=float)
        )

        absolute = (
            frame[config["absolute"]]
            .to_numpy(dtype=float)
        )

        ape = (
            frame[config["ape"]]
            .to_numpy(dtype=float)
        )

        rows.append({
            "stratum_type":
                stratum_type,

            "stratum":
                str(stratum),

            "model":
                name,

            "label":
                config["label"],

            "n":
                len(frame),

            "mae_seconds":
                float(
                    absolute.mean()
                ),

            "rmse_seconds":
                float(
                    np.sqrt(
                        np.mean(
                            signed ** 2
                        )
                    )
                ),

            "mape_percent":
                float(
                    ape.mean()
                ),

            "median_abs_error_seconds":
                float(
                    np.median(
                        absolute
                    )
                ),

            "p90_abs_error_seconds":
                float(
                    np.quantile(
                        absolute,
                        0.90,
                    )
                ),

            "mean_signed_error_seconds":
                float(
                    signed.mean()
                ),

            "within_60_seconds_percent":
                float(
                    np.mean(
                        absolute <= 60
                    )
                    * 100.0
                ),
        })

    return rows

def pairwise_rows(
    frame,
    stratum_type,
    stratum,
):
    rows = []

    pairs = [
        ("baseline", "structured"),
        ("baseline", "qwen"),
        ("structured", "qwen"),
    ]

    for a, b in pairs:

        abs_a = (
            frame[
                MODELS[a]["absolute"]
            ]
            .to_numpy(dtype=float)
        )

        abs_b = (
            frame[
                MODELS[b]["absolute"]
            ]
            .to_numpy(dtype=float)
        )

        ape_a = (
            frame[
                MODELS[a]["ape"]
            ]
            .to_numpy(dtype=float)
        )

        ape_b = (
            frame[
                MODELS[b]["ape"]
            ]
            .to_numpy(dtype=float)
        )

        signed_a = (
            frame[
                MODELS[a]["signed"]
            ]
            .to_numpy(dtype=float)
        )

        signed_b = (
            frame[
                MODELS[b]["signed"]
            ]
            .to_numpy(dtype=float)
        )

        # Positive = model B better.
        mae_improvement = (
            abs_a.mean()
            - abs_b.mean()
        )

        rmse_improvement = (
            np.sqrt(
                np.mean(
                    signed_a ** 2
                )
            )
            -
            np.sqrt(
                np.mean(
                    signed_b ** 2
                )
            )
        )

        mape_improvement = (
            ape_a.mean()
            - ape_b.mean()
        )

        b_wins = int(
            (
                abs_b
                < abs_a
            ).sum()
        )

        a_wins = int(
            (
                abs_a
                < abs_b
            ).sum()
        )

        rows.append({
            "stratum_type":
                stratum_type,

            "stratum":
                str(stratum),

            "model_a":
                a,

            "model_b":
                b,

            "n":
                len(frame),

            "mae_improvement_seconds":
                float(
                    mae_improvement
                ),

            "rmse_improvement_seconds":
                float(
                    rmse_improvement
                ),

            "mape_improvement_pp":
                float(
                    mape_improvement
                ),

            "model_b_lower_abs_error_count":
                b_wins,

            "model_a_lower_abs_error_count":
                a_wins,

            "model_b_win_percent":
                float(
                    b_wins
                    / len(frame)
                    * 100.0
                ),
        })

    return rows

HISTORICAL = True


def main(argv=None):
    global HISTORICAL, MODELS, absent, c, column, concept, concept_columns, concept_df, concept_path, concept_rows, count, counts, df, duration_edges, duration_labels, duration_pairs, group, handle, manifest, manifest_path, metric_output, metrics_df, metrics_path, pair_output, pairs_df, pairs_path, presence_label, present, route_edges, route_labels, row, semantic_edges, semantic_labels, semantic_pairs, stratum, support_df, support_path, support_rows
    from scripts.portable_runtime_support import analysis_cli
    analysis_cli(globals(), "tte", argv)

    df = pd.read_csv(INPUT)

    if len(df) != 76923:
        raise RuntimeError(
            "Unexpected TTE sample count."
        )

    MODELS = {
        "baseline": {
            "label": "Baseline",
            "signed": "signed_error_baseline_seconds",
            "absolute": "abs_error_baseline_seconds",
            "ape": "ape_baseline_percent",
        },
        "structured": {
            "label": "Structured",
            "signed": "signed_error_structured_seconds",
            "absolute": "abs_error_structured_seconds",
            "ape": "ape_structured_percent",
        },
        "qwen": {
            "label": "Qwen",
            "signed": "signed_error_qwen_seconds",
            "absolute": "abs_error_qwen_seconds",
            "ape": "ape_qwen_percent",
        },
    }

    metric_output = []

    pair_output = []

    df["duration_decile"] = pd.qcut(
        df["target_seconds"],
        q=10,
        duplicates="drop",
    )

    for stratum, group in df.groupby(
        "duration_decile",
        observed=True,
    ):
        metric_output.extend(
            metric_rows(
                group,
                "duration_decile",
                stratum,
            )
        )

        pair_output.extend(
            pairwise_rows(
                group,
                "duration_decile",
                stratum,
            )
        )

    duration_edges = [
        -np.inf,
        200,
        335,
        495,
        731,
        np.inf,
    ]

    duration_labels = [
        "D1 <=200s",
        "D2 201-335s",
        "D3 336-495s",
        "D4 496-731s",
        "D5 >=732s",
    ]

    df["duration_group"] = pd.cut(
        df["target_seconds"],
        bins=duration_edges,
        labels=duration_labels,
    )

    for stratum, group in df.groupby(
        "duration_group",
        observed=True,
    ):
        metric_output.extend(
            metric_rows(
                group,
                "duration_group",
                stratum,
            )
        )

        pair_output.extend(
            pairwise_rows(
                group,
                "duration_group",
                stratum,
            )
        )

    route_edges = [
        5,
        9,
        14,
        20,
        30,
        64,
    ]

    route_labels = [
        "R1 6-9 roads",
        "R2 10-14 roads",
        "R3 15-20 roads",
        "R4 21-30 roads",
        "R5 31-64 roads",
    ]

    df["route_length_group"] = pd.cut(
        df["route_length_roads"],
        bins=route_edges,
        labels=route_labels,
        include_lowest=True,
    )

    for stratum, group in df.groupby(
        "route_length_group",
        observed=True,
    ):
        metric_output.extend(
            metric_rows(
                group,
                "route_length_group",
                stratum,
            )
        )

        pair_output.extend(
            pairwise_rows(
                group,
                "route_length_group",
                stratum,
            )
        )

    semantic_edges = [
        -1,
        2,
        4,
        6,
        10,
    ]

    semantic_labels = [
        "S1 0-2 concepts",
        "S2 3-4 concepts",
        "S3 5-6 concepts",
        "S4 7-10 concepts",
    ]

    df["semantic_richness_group"] = pd.cut(
        df["distinct_semantic_concept_count"],
        bins=semantic_edges,
        labels=semantic_labels,
    )

    for stratum, group in df.groupby(
        "semantic_richness_group",
        observed=True,
    ):
        metric_output.extend(
            metric_rows(
                group,
                "semantic_richness_group",
                stratum,
            )
        )

        pair_output.extend(
            pairwise_rows(
                group,
                "semantic_richness_group",
                stratum,
            )
        )

    concept_columns = [
        c
        for c in df.columns
        if c.startswith("has_")
    ]

    concept_rows = []

    for column in concept_columns:

        concept = column.replace(
            "has_",
            "",
        )

        present = df[
            df[column].astype(bool)
        ]

        absent = df[
            ~df[column].astype(bool)
        ]

        for presence_label, group in [
            ("present", present),
            ("absent", absent),
        ]:

            if len(group) == 0:
                continue

            for row in metric_rows(
                group,
                "concept_presence",
                f"{concept}:{presence_label}",
            ):
                concept_rows.append(row)

    support_rows = []

    for column in [
        "duration_decile",
        "duration_group",
        "route_length_group",
        "semantic_richness_group",
    ]:

        counts = (
            df[column]
            .value_counts(
                sort=False
            )
        )

        for stratum, count in counts.items():

            support_rows.append({
                "stratum_type":
                    column,

                "stratum":
                    str(stratum),

                "n":
                    int(count),

                "percent":
                    float(
                        count
                        / len(df)
                        * 100.0
                    ),
            })

    metrics_df = pd.DataFrame(
        metric_output
    )

    pairs_df = pd.DataFrame(
        pair_output
    )

    concept_df = pd.DataFrame(
        concept_rows
    )

    support_df = pd.DataFrame(
        support_rows
    )

    metrics_path = (
        OUTPUT
        / "tte_stratified_metrics.csv"
    )

    pairs_path = (
        OUTPUT
        / "tte_stratified_pairwise_improvements.csv"
    )

    concept_path = (
        OUTPUT
        / "tte_concept_presence_metrics.csv"
    )

    support_path = (
        OUTPUT
        / "tte_strata_support.csv"
    )

    manifest_path = (
        OUTPUT
        / "tte_stratification_manifest.json"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    pairs_df.to_csv(
        pairs_path,
        index=False,
    )

    concept_df.to_csv(
        concept_path,
        index=False,
    )

    support_df.to_csv(
        support_path,
        index=False,
    )

    manifest = {
        "sample_count":
            len(df),

        "locked_before_model_result_inspection":
            True,

        "duration_deciles":
            10,

        "duration_groups":
            duration_labels,

        "route_length_groups":
            route_labels,

        "semantic_richness_groups":
            semantic_labels,

        "individual_concepts":
            [
                c.replace(
                    "has_",
                    "",
                )
                for c in concept_columns
            ],
    }

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            manifest,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "============================================"
    )

    print(
        " TTE STRATIFIED ANALYSIS"
    )

    print(
        "============================================"
    )

    print()

    print(
        "=== SUPPORT ==="
    )

    print(
        support_df.to_string(
            index=False
        )
    )

    print()

    print(
        "=== DURATION GROUP PAIRWISE ==="
    )

    duration_pairs = pairs_df[
        pairs_df["stratum_type"]
        == "duration_group"
    ]

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        300,
        "display.precision",
        5,
    ):

        print(
            duration_pairs.to_string(
                index=False
            )
        )

    print()

    print(
        "=== SEMANTIC RICHNESS PAIRWISE ==="
    )

    semantic_pairs = pairs_df[
        pairs_df["stratum_type"]
        == "semantic_richness_group"
    ]

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        300,
        "display.precision",
        5,
    ):

        print(
            semantic_pairs.to_string(
                index=False
            )
        )

    print()

    print(
        "Outputs:"
    )

    print(metrics_path)

    print(pairs_path)

    print(concept_path)

    print(support_path)

    print(manifest_path)

    print()

    print(
        "STATUS: FINAL TTE STRATIFIED ANALYSIS PASS"
    )


if __name__ == "__main__":
    main()
