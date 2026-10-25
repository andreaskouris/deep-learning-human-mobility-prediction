from __future__ import annotations

# Support direct script execution without depending on the caller PYTHONPATH.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib

import json

from pathlib import Path

import numpy as np

import pandas as pd

try:
    from scipy.stats import binomtest
except ImportError as exc:
    raise RuntimeError(
        "scipy is required for the exact McNemar test."
    ) from exc

ROOT = Path("final_analysis/next_hop")

INPUT_DIR = (
    ROOT
    / "per_sample"
)

OUTPUT_DIR = (
    ROOT
    / "statistics"
)

MODELS = {
    "nh1": {
        "label": "Baseline",
        "path":
            INPUT_DIR
            / "nh1_baseline_predictions.csv",
        "expected": {
            "accuracy": "77.9143",
            "mrr": "86.7511",
            "ndcg": "89.5217",
        },
    },

    "nh2": {
        "label": "POI TF-IDF",
        "path":
            INPUT_DIR
            / "nh2_poi_tfidf_predictions.csv",
        "expected": {
            "accuracy": "79.5575",
            "mrr": "88.4665",
            "ndcg": "91.1168",
        },
    },

    "nh3": {
        "label": "Qwen POI",
        "path":
            INPUT_DIR
            / "nh3_qwen_poi_predictions.csv",
        "expected": {
            "accuracy": "79.8240",
            "mrr": "88.5598",
            "ndcg": "91.1645",
        },
    },
}

PAIRS = [
    ("nh1", "nh2"),
    ("nh1", "nh3"),
    ("nh2", "nh3"),
]

EXPECTED_SAMPLES = 76923

BOOTSTRAP_REPETITIONS = 2000

BOOTSTRAP_SEED = 2021

BOOTSTRAP_CHUNK_SIZE = 50

def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest().upper()

def correctness_array(
    series: pd.Series,
) -> np.ndarray:

    if pd.api.types.is_bool_dtype(
        series
    ):
        return (
            series
            .to_numpy(
                dtype=bool
            )
        )

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().all():

        unique = set(
            numeric.unique().tolist()
        )

        if unique.issubset(
            {0, 1}
        ):
            return (
                numeric
                .to_numpy(
                    dtype=np.int8
                )
                .astype(bool)
            )

    text = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    mapped = text.map(
        mapping
    )

    if mapped.isna().any():
        raise RuntimeError(
            "Could not parse top1_correct."
        )

    return mapped.to_numpy(
        dtype=bool
    )

def paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    seed: int,
):

    """
    differences:
        N x K matrix.

    Each column is a per-sample paired
    difference B - A.

    Returns percentile 95% CIs for the
    mean difference of every column.
    """

    differences = np.asarray(
        differences,
        dtype=np.float64,
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

    lower = np.quantile(
        bootstrap_means,
        0.025,
        axis=0,
    )

    upper = np.quantile(
        bootstrap_means,
        0.975,
        axis=0,
    )

    return (
        lower.astype(float),
        upper.astype(float),
    )

def exact_mcnemar(
    regressions: int,
    corrections: int,
):

    discordant = (
        regressions
        + corrections
    )

    if discordant == 0:
        return 1.0

    result = binomtest(
        min(
            regressions,
            corrections,
        ),
        n=discordant,
        p=0.5,
        alternative="two-sided",
    )

    return float(
        result.pvalue
    )

def holm_adjust(
    p_values,
):

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    m = len(
        p_values
    )

    order = np.argsort(
        p_values
    )

    adjusted = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for rank, index in enumerate(
        order
    ):

        multiplier = (
            m - rank
        )

        candidate = min(
            1.0,
            p_values[index]
            * multiplier,
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted[index] = (
            running_max
        )

    return adjusted

HISTORICAL = True


def main(argv=None):
    global HISTORICAL, a_only_correct, accuracy, accuracy_a, accuracy_b, b_only_correct, both_correct, both_wrong, ci_high, ci_low, column, config, correct, correct_a, correct_b, corrections, count, differences, display_columns, eps, frame, frame_a, frame_b, frames, handle, key, label_a, label_b, master, master_path, mcnemar_p, model_a, model_b, model_metric_rows, model_metrics, model_metrics_path, model_name, mrr, mrr_a, mrr_b, mrr_diff, ndcg, ndcg_a, ndcg_b, ndcg_diff, observed, pair_index, pairwise, pairwise_path, pairwise_rows, path, point_delta, prediction_agreement, reference, regressions, source, summary, summary_path, transition, transition_rows, transitions, transitions_path
    from scripts.portable_runtime_support import analysis_cli
    analysis_cli(globals(), "next_hop", argv)

    print(
        "============================================"
    )

    print(
        " FINAL NEXT-HOP PAIRED STATISTICAL ANALYSIS"
    )

    print(
        "============================================"
    )

    frames = {}

    for model_name, config in (
        MODELS.items()
    ):

        path = config["path"]

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        frame = pd.read_csv(
            path
        )

        if len(frame) != EXPECTED_SAMPLES:
            raise RuntimeError(
                f"{model_name}: unexpected "
                "sample count."
            )

        if (
            frame["sample_id"]
            .duplicated()
            .any()
        ):
            raise RuntimeError(
                f"{model_name}: duplicate "
                "sample_id."
            )

        frames[
            model_name
        ] = frame

    reference = frames["nh1"]

    for model_name in (
        "nh2",
        "nh3",
    ):

        frame = frames[
            model_name
        ]

        if not reference[
            "sample_id"
        ].equals(
            frame[
                "sample_id"
            ]
        ):
            raise RuntimeError(
                "Sample-ID alignment failure."
            )

        if not reference[
            "true_next_road"
        ].equals(
            frame[
                "true_next_road"
            ]
        ):
            raise RuntimeError(
                "Ground-truth alignment failure."
            )

    print(
        "Sample alignment: PASS"
    )

    model_metric_rows = []

    for model_name, config in (
        MODELS.items()
    ):

        frame = frames[
            model_name
        ]

        correct = correctness_array(
            frame[
                "top1_correct"
            ]
        )

        accuracy = (
            correct.mean()
            * 100.0
        )

        mrr = (
            frame[
                "reciprocal_rank_at_5"
            ]
            .to_numpy(
                dtype=np.float64
            )
            .mean()
            * 100.0
        )

        ndcg = (
            frame[
                "ndcg_at_5"
            ]
            .to_numpy(
                dtype=np.float64
            )
            .mean()
            * 100.0
        )

        observed = {
            "accuracy": f"{accuracy:.4f}",
            "mrr": f"{mrr:.4f}",
            "ndcg": f"{ndcg:.4f}",
        }

        for key in (
            "accuracy",
            "mrr",
            "ndcg",
        ):

            if (
                HISTORICAL and (observed[key]
                != config[
                    "expected"
                ][key])
            ):
                raise RuntimeError(
                    f"{model_name} {key} "
                    "reproduction failure: "
                    f"{observed[key]}"
                )

        model_metric_rows.append(
            {
                "model":
                    model_name,

                "label":
                    config["label"],

                "sample_count":
                    len(frame),

                "accuracy_percent":
                    accuracy,

                "mrr_at_5_percent":
                    mrr,

                "ndcg_at_5_percent":
                    ndcg,

                "prediction_csv_sha256":
                    sha256_file(
                        config[
                            "path"
                        ]
                    ),
            }
        )

    model_metrics = pd.DataFrame(
        model_metric_rows
    )

    print()

    print(
        "=== MODEL METRICS ==="
    )

    print(
        model_metrics[
            [
                "label",
                "accuracy_percent",
                "mrr_at_5_percent",
                "ndcg_at_5_percent",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()

    print(
        "Historical metric parity: PASS" if HISTORICAL else "Fresh metrics computed; historical parity is not required"
    )

    master = reference[
        [
            "sample_id",
            "true_next_road",
        ]
    ].copy()

    for model_name in (
        "nh1",
        "nh2",
        "nh3",
    ):

        source = frames[
            model_name
        ].drop(
            columns=[
                "true_next_road",
            ]
        )

        source = source.rename(
            columns={
                column:
                    (
                        column
                        if column
                        == "sample_id"
                        else
                        f"{model_name}_{column}"
                    )
                for column
                in source.columns
            }
        )

        if model_name == "nh1":
            master = master.merge(
                source,
                on="sample_id",
                how="left",
                validate="one_to_one",
            )

        else:
            master = master.merge(
                source,
                on="sample_id",
                how="left",
                validate="one_to_one",
            )

    master_path = (
        INPUT_DIR
        / "next_hop_per_sample_master.csv"
    )

    master.to_csv(
        master_path,
        index=False,
    )

    pairwise_rows = []

    transition_rows = []

    for pair_index, (
        model_a,
        model_b,
    ) in enumerate(
        PAIRS
    ):

        frame_a = frames[
            model_a
        ]

        frame_b = frames[
            model_b
        ]

        label_a = MODELS[
            model_a
        ]["label"]

        label_b = MODELS[
            model_b
        ]["label"]

        correct_a = correctness_array(
            frame_a[
                "top1_correct"
            ]
        )

        correct_b = correctness_array(
            frame_b[
                "top1_correct"
            ]
        )


        
        # Per-sample metric contributions
        

        accuracy_a = (
            correct_a
            .astype(np.float64)
        )

        accuracy_b = (
            correct_b
            .astype(np.float64)
        )

        mrr_a = (
            frame_a[
                "reciprocal_rank_at_5"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        mrr_b = (
            frame_b[
                "reciprocal_rank_at_5"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        ndcg_a = (
            frame_a[
                "ndcg_at_5"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        ndcg_b = (
            frame_b[
                "ndcg_at_5"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )


        # Positive = model B better.
        differences = np.column_stack(
            [
                accuracy_b
                - accuracy_a,

                mrr_b
                - mrr_a,

                ndcg_b
                - ndcg_a,
            ]
        )


        point_delta = (
            differences.mean(
                axis=0
            )
            * 100.0
        )


        ci_low, ci_high = (
            paired_bootstrap_ci(
                differences,
                seed=(
                    BOOTSTRAP_SEED
                    + pair_index
                ),
            )
        )

        ci_low = (
            ci_low
            * 100.0
        )

        ci_high = (
            ci_high
            * 100.0
        )


        
        # Correctness transitions
        

        both_wrong = int(
            (
                ~correct_a
                & ~correct_b
            )
            .sum()
        )

        a_only_correct = int(
            (
                correct_a
                & ~correct_b
            )
            .sum()
        )

        b_only_correct = int(
            (
                ~correct_a
                & correct_b
            )
            .sum()
        )

        both_correct = int(
            (
                correct_a
                & correct_b
            )
            .sum()
        )


        # A -> B terminology
        regressions = (
            a_only_correct
        )

        corrections = (
            b_only_correct
        )

        mcnemar_p = (
            exact_mcnemar(
                regressions,
                corrections,
            )
        )


        prediction_agreement = int(
            (
                frame_a[
                    "top1_prediction"
                ]
                .to_numpy()
                ==
                frame_b[
                    "top1_prediction"
                ]
                .to_numpy()
            )
            .sum()
        )


        mrr_diff = (
            mrr_b
            - mrr_a
        )

        ndcg_diff = (
            ndcg_b
            - ndcg_a
        )

        eps = 1e-12


        pairwise_rows.append(
            {
                "comparison":
                    (
                        f"{model_a}_vs_"
                        f"{model_b}"
                    ),

                "model_a":
                    label_a,

                "model_b":
                    label_b,

                "sample_count":
                    EXPECTED_SAMPLES,

                # Positive means B better.
                "delta_accuracy_pp":
                    point_delta[0],

                "delta_accuracy_ci95_low_pp":
                    ci_low[0],

                "delta_accuracy_ci95_high_pp":
                    ci_high[0],

                "delta_mrr_at_5_pp":
                    point_delta[1],

                "delta_mrr_ci95_low_pp":
                    ci_low[1],

                "delta_mrr_ci95_high_pp":
                    ci_high[1],

                "delta_ndcg_at_5_pp":
                    point_delta[2],

                "delta_ndcg_ci95_low_pp":
                    ci_low[2],

                "delta_ndcg_ci95_high_pp":
                    ci_high[2],

                "both_wrong":
                    both_wrong,

                "regressions_a_correct_b_wrong":
                    regressions,

                "corrections_a_wrong_b_correct":
                    corrections,

                "both_correct":
                    both_correct,

                "net_corrections":
                    (
                        corrections
                        - regressions
                    ),

                "discordant_samples":
                    (
                        corrections
                        + regressions
                    ),

                "mcnemar_p_raw":
                    mcnemar_p,

                "same_top1_prediction_count":
                    prediction_agreement,

                "same_top1_prediction_percent":
                    (
                        prediction_agreement
                        / EXPECTED_SAMPLES
                        * 100.0
                    ),

                "mrr_better_count":
                    int(
                        (
                            mrr_diff
                            > eps
                        )
                        .sum()
                    ),

                "mrr_worse_count":
                    int(
                        (
                            mrr_diff
                            < -eps
                        )
                        .sum()
                    ),

                "mrr_equal_count":
                    int(
                        (
                            np.abs(
                                mrr_diff
                            )
                            <= eps
                        )
                        .sum()
                    ),

                "ndcg_better_count":
                    int(
                        (
                            ndcg_diff
                            > eps
                        )
                        .sum()
                    ),

                "ndcg_worse_count":
                    int(
                        (
                            ndcg_diff
                            < -eps
                        )
                        .sum()
                    ),

                "ndcg_equal_count":
                    int(
                        (
                            np.abs(
                                ndcg_diff
                            )
                            <= eps
                        )
                        .sum()
                    ),
            }
        )


        for transition, count in [
            (
                "both_wrong",
                both_wrong,
            ),
            (
                "a_only_correct",
                a_only_correct,
            ),
            (
                "b_only_correct",
                b_only_correct,
            ),
            (
                "both_correct",
                both_correct,
            ),
        ]:

            transition_rows.append(
                {
                    "comparison":
                        (
                            f"{model_a}_vs_"
                            f"{model_b}"
                        ),

                    "model_a":
                        label_a,

                    "model_b":
                        label_b,

                    "transition":
                        transition,

                    "count":
                        count,

                    "percent":
                        (
                            count
                            / EXPECTED_SAMPLES
                            * 100.0
                        ),
                }
            )

    pairwise = pd.DataFrame(
        pairwise_rows
    )

    pairwise[
        "mcnemar_p_holm"
    ] = holm_adjust(
        pairwise[
            "mcnemar_p_raw"
        ]
        .to_numpy()
    )

    transitions = pd.DataFrame(
        transition_rows
    )

    model_metrics_path = (
        OUTPUT_DIR
        / "next_hop_model_metrics.csv"
    )

    pairwise_path = (
        OUTPUT_DIR
        / "next_hop_pairwise_statistics.csv"
    )

    transitions_path = (
        OUTPUT_DIR
        / "next_hop_correctness_transitions.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "next_hop_statistics_summary.json"
    )

    model_metrics.to_csv(
        model_metrics_path,
        index=False,
    )

    pairwise.to_csv(
        pairwise_path,
        index=False,
    )

    transitions.to_csv(
        transitions_path,
        index=False,
    )

    summary = {
        "reproduction_profile": "historical" if HISTORICAL else "fresh",
        "historical_metric_parity_verified": HISTORICAL,
        "task":
            "next_hop",

        "sample_count":
            EXPECTED_SAMPLES,

        "paired_alignment":
            True,

        "bootstrap": {
            "method":
                "paired_percentile_bootstrap",

            "repetitions":
                BOOTSTRAP_REPETITIONS,

            "seed_base":
                BOOTSTRAP_SEED,

            "confidence_level":
                0.95,

            "interpretation":
                (
                    "Uncertainty across paired "
                    "test samples, not across "
                    "training seeds."
                ),
        },

        "mcnemar": {
            "method":
                "exact_two_sided_binomial_McNemar",

            "multiple_comparison_adjustment":
                "Holm",

            "family":
                [
                    "NH1_vs_NH2",
                    "NH1_vs_NH3",
                    "NH2_vs_NH3",
                ],
        },

        "master_per_sample_csv":
            str(master_path),

        "master_per_sample_sha256":
            sha256_file(
                master_path
            ),

        "outputs": {
            "model_metrics":
                str(model_metrics_path),

            "pairwise_statistics":
                str(pairwise_path),

            "correctness_transitions":
                str(transitions_path),
        },
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print()

    print(
        "============================================"
    )

    print(
        " PAIRWISE RESULTS"
    )

    print(
        " Positive delta = model B better"
    )

    print(
        "============================================"
    )

    display_columns = [
        "model_a",
        "model_b",
        "delta_accuracy_pp",
        "delta_accuracy_ci95_low_pp",
        "delta_accuracy_ci95_high_pp",
        "delta_mrr_at_5_pp",
        "delta_mrr_ci95_low_pp",
        "delta_mrr_ci95_high_pp",
        "delta_ndcg_at_5_pp",
        "delta_ndcg_ci95_low_pp",
        "delta_ndcg_ci95_high_pp",
        "corrections_a_wrong_b_correct",
        "regressions_a_correct_b_wrong",
        "mcnemar_p_raw",
        "mcnemar_p_holm",
        "same_top1_prediction_percent",
    ]

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        300,
        "display.precision",
        8,
    ):

        print(
            pairwise[
                display_columns
            ]
            .to_string(
                index=False
            )
        )

    print()

    print(
        "============================================"
    )

    print(
        " OUTPUTS"
    )

    print(
        "============================================"
    )

    print(
        "Master per-sample:",
        master_path,
    )

    print(
        "Master SHA256:",
        sha256_file(
            master_path
        ),
    )

    print(
        "Model metrics:",
        model_metrics_path,
    )

    print(
        "Pairwise statistics:",
        pairwise_path,
    )

    print(
        "Correctness transitions:",
        transitions_path,
    )

    print(
        "Summary JSON:",
        summary_path,
    )

    print()

    print(
        "STATUS: FINAL NEXT-HOP STATISTICAL ANALYSIS PASS"
    )


if __name__ == "__main__":
    main()
