from __future__ import annotations

# Support direct script execution without depending on the caller PYTHONPATH.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib

import json

import sys

from ast import literal_eval

from pathlib import Path

import numpy as np

import pandas as pd

import torch

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )

DATA_ROOT = (
    ROOT.parent
    / "dataset"
    / "xa"
)

MASTER_PATH = (
    ROOT
    / "final_analysis"
    / "next_hop"
    / "per_sample"
    / "next_hop_per_sample_master.csv"
)

TEST_CACHE = (
    DATA_ROOT
    / "cached_next_hop_dataset_test_seq64_sr1_seed2021_v2.pth"
)

SEMANTIC_CACHE = (
    DATA_ROOT
    / "cached_xa_semantic_tfidf_v1.pth"
)

OUTPUT_DIR = (
    ROOT
    / "final_analysis"
    / "next_hop"
    / "stratified"
)

EXPECTED_SAMPLES = 76923

EXPECTED_ROADS = 5269

SEQ_LEN = 64

BOOTSTRAP_REPETITIONS = 2000

BOOTSTRAP_CHUNK_SIZE = 50

BOOTSTRAP_SEED = 2021

EXPECTED_TEST_SHA = (
    "FAFCBB6A7DB3BEC52A091744A8971E24"
    "EAD3790FE705A662875D30F2C5BD2F04"
)

EXPECTED_SEMANTIC_SHA = (
    "118DADD89F2A0CCF573078C01CDC09C41"
    "BF0865BA23F433709AC32775584D989"
)

MODELS = {
    "nh1": {
        "label": "Baseline",
    },
    "nh2": {
        "label": "POI TF-IDF",
    },
    "nh3": {
        "label": "Qwen POI",
    },
}

PAIRS = [
    ("nh1", "nh2"),
    ("nh1", "nh3"),
    ("nh2", "nh3"),
]

CONTEXT_LABELS = [
    "C1 5-8",
    "C2 9-16",
    "C3 17-32",
    "C4 33-63",
]

SEMANTIC_LABELS = [
    "S0 0%",
    "S1 (0,25%]",
    "S2 (25,50%]",
    "S3 (50,75%]",
    "S4 (75,100%]",
]

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

def find_semantic_tensor(obj):

    candidates = []

    def visit(value, location):

        if isinstance(
            value,
            torch.Tensor,
        ):

            if (
                value.ndim == 2
                and tuple(
                    value.shape
                ) == (
                    EXPECTED_ROADS,
                    204,
                )
            ):
                candidates.append(
                    (
                        location,
                        value,
                    )
                )

        elif isinstance(
            value,
            dict,
        ):

            for key, item in (
                value.items()
            ):
                visit(
                    item,
                    f"{location}.{key}",
                )

        elif isinstance(
            value,
            (list, tuple),
        ):

            for index, item in enumerate(
                value
            ):
                visit(
                    item,
                    f"{location}[{index}]",
                )

    visit(
        obj,
        "root",
    )

    if len(
        candidates
    ) != 1:

        raise RuntimeError(
            "Expected exactly one "
            "(5269, 204) semantic tensor."
        )

    return candidates[0]

def correctness_array(series):

    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.to_numpy(
            dtype=bool
        )

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().all():

        unique = set(
            numeric
            .unique()
            .tolist()
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

    result = text.map(
        mapping
    )

    if result.isna().any():
        raise RuntimeError(
            "Could not parse correctness."
        )

    return result.to_numpy(
        dtype=bool
    )

def paired_bootstrap_ci(
    differences,
    *,
    seed,
):

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

HISTORICAL = True


def main(argv=None):
    global HISTORICAL, agreement, cache, cached_labels, cached_roads, column, config, context, context_lengths, context_results, correct, correct_a, correct_b, corrections, count, differences, enriched_path, group, handle, high, index, location, low, manifest, manifest_path, master, metric_rows, metrics_df, metrics_path, model_a, model_b, model_name, mrr, mrr_a, mrr_b, n, ndcg, ndcg_a, ndcg_b, ordered_labels, original_argv, pairwise_df, pairwise_path, pairwise_rows, path, paths, point, raw, raw_file_loader, real_contexts, reconstructed, reconstructed_labels, reconstructed_roads, regressions, seed_counter, semantic_context_count, semantic_context_ratio, semantic_mask, semantic_obj, semantic_results, semantic_tensor, stratifications, stratum, stratum_type, support_df, support_path, support_rows, target
    from scripts.portable_runtime_support import analysis_cli
    analysis_cli(globals(), "next_hop", argv)

    print(
        "============================================"
    )

    print(
        " FINAL NEXT-HOP STRATIFIED ANALYSIS"
    )

    print(
        "============================================"
    )

    if HISTORICAL and (sha256_file(
        TEST_CACHE
    ) != EXPECTED_TEST_SHA):

        raise RuntimeError(
            "Test cache SHA mismatch."
        )

    if HISTORICAL and (sha256_file(
        SEMANTIC_CACHE
    ) != EXPECTED_SEMANTIC_SHA):

        raise RuntimeError(
            "Semantic cache SHA mismatch."
        )

    print(
        "Artifact integrity: PASS"
    )

    master = pd.read_csv(
        MASTER_PATH
    )

    if len(
        master
    ) != EXPECTED_SAMPLES:

        raise RuntimeError(
            "Unexpected master sample count."
        )

    if not np.array_equal(
        master[
            "sample_id"
        ].to_numpy(),
        np.arange(
            EXPECTED_SAMPLES
        ),
    ):

        raise RuntimeError(
            "Unexpected sample IDs."
        )

    print(
        "Final prediction master: PASS"
    )

    original_argv = sys.argv[:]

    try:

        sys.argv = [
            original_argv[0],

            "--dataset_path", str(DATASET_PATH),
            "--city", "xa",
            "--split", "test",
            "--seq_len", "64",
            "--sample_rate", "1.0",
            "--dynamic_mode", "controlled",
            "--seed", "2021",
        ]

        from data_provider.file_loader import (
            file_loader as raw_file_loader
        )

        raw_file_loader.load_traj_dataset_file()

        raw = (
            raw_file_loader
            .get_traj_data()
            .copy()
        )

    finally:

        sys.argv = original_argv

    if len(
        raw
    ) != EXPECTED_SAMPLES:

        raise RuntimeError(
            "Raw test sample count mismatch."
        )

    paths = (
        raw[
            "path"
        ]
        .apply(
            literal_eval
        )
        .tolist()
    )

    cache = torch.load(
        TEST_CACHE,
        map_location="cpu",
        weights_only=True,
    )

    cached_roads = (
        cache[
            "traj_road_id_lists"
        ]
        .cpu()
        .numpy()
    )

    cached_labels = (
        cache[
            "nexthop_labels"
        ]
        .cpu()
        .numpy()
    )

    reconstructed_roads = np.empty(
        (
            EXPECTED_SAMPLES,
            SEQ_LEN,
        ),
        dtype=np.int64,
    )

    reconstructed_labels = np.empty(
        EXPECTED_SAMPLES,
        dtype=np.int64,
    )

    context_lengths = np.empty(
        EXPECTED_SAMPLES,
        dtype=np.int64,
    )

    real_contexts = []

    for index, path in enumerate(
        paths
    ):

        if len(path) < 2:
            raise RuntimeError(
                f"Invalid trajectory {index}."
            )

        if len(path) > SEQ_LEN:

            context = (
                path[
                    :SEQ_LEN - 1
                ]
            )

            reconstructed = (
                context
                + [
                    path[
                        SEQ_LEN - 2
                    ]
                ]
            )

            target = (
                path[
                    SEQ_LEN - 1
                ]
            )

        else:

            context = (
                path[:-1]
            )

            reconstructed = (
                context
                + [
                    path[-2]
                ]
                * (
                    SEQ_LEN
                    - len(path)
                    + 1
                )
            )

            target = (
                path[-1]
            )


        reconstructed_roads[
            index
        ] = np.asarray(
            reconstructed,
            dtype=np.int64,
        )

        reconstructed_labels[
            index
        ] = target

        context_lengths[
            index
        ] = len(
            context
        )

        real_contexts.append(
            np.asarray(
                context,
                dtype=np.int64,
            )
        )

    if not np.array_equal(
        reconstructed_roads,
        cached_roads,
    ):
        raise RuntimeError(
            "Raw/cache road alignment failed."
        )

    if not np.array_equal(
        reconstructed_labels,
        cached_labels,
    ):
        raise RuntimeError(
            "Raw/cache label alignment failed."
        )

    if not np.array_equal(
        reconstructed_labels,
        master[
            "true_next_road"
        ].to_numpy(
            dtype=np.int64
        ),
    ):
        raise RuntimeError(
            "Raw/final-label alignment failed."
        )

    print(
        "Raw/cache/final alignment: PASS"
    )

    semantic_obj = torch.load(
        SEMANTIC_CACHE,
        map_location="cpu",
        weights_only=True,
    )

    location, semantic_tensor = (
        find_semantic_tensor(
            semantic_obj
        )
    )

    semantic_mask = (
        torch.linalg.vector_norm(
            semantic_tensor
            .detach()
            .cpu(),
            dim=1,
        )
        > 0
    ).numpy()

    semantic_context_count = np.empty(
        EXPECTED_SAMPLES,
        dtype=np.int64,
    )

    semantic_context_ratio = np.empty(
        EXPECTED_SAMPLES,
        dtype=np.float64,
    )

    for index, context in enumerate(
        real_contexts
    ):

        count = int(
            semantic_mask[
                context
            ].sum()
        )

        semantic_context_count[
            index
        ] = count

        semantic_context_ratio[
            index
        ] = (
            count
            / len(context)
        )

    master[
        "context_length"
    ] = context_lengths

    master[
        "semantic_context_count"
    ] = semantic_context_count

    master[
        "semantic_context_ratio"
    ] = semantic_context_ratio

    master[
        "context_length_group"
    ] = pd.cut(
        context_lengths,
        bins=[
            4,
            8,
            16,
            32,
            63,
        ],
        labels=
            CONTEXT_LABELS,
        include_lowest=True,
    )

    master[
        "semantic_coverage_group"
    ] = pd.cut(
        semantic_context_ratio,
        bins=[
            -1e-12,
            0.0,
            0.25,
            0.50,
            0.75,
            1.0,
        ],
        labels=
            SEMANTIC_LABELS,
        include_lowest=True,
    )

    if (
        master[
            "context_length_group"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Missing context-length stratum."
        )

    if (
        master[
            "semantic_coverage_group"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Missing semantic-coverage stratum."
        )

    metric_rows = []

    pairwise_rows = []

    support_rows = []

    stratifications = [
        (
            "context_length",
            "context_length_group",
            CONTEXT_LABELS,
        ),
        (
            "semantic_coverage",
            "semantic_coverage_group",
            SEMANTIC_LABELS,
        ),
    ]

    seed_counter = 0

    for (
        stratum_type,
        column,
        ordered_labels,
    ) in stratifications:

        for stratum in ordered_labels:

            group = master[
                master[
                    column
                ].astype(str)
                == stratum
            ].copy()


            n = len(
                group
            )

            if n == 0:
                raise RuntimeError(
                    f"Empty stratum: {stratum}"
                )


            support_rows.append(
                {
                    "stratum_type":
                        stratum_type,

                    "stratum":
                        stratum,

                    "n":
                        n,

                    "percent":
                        n
                        / EXPECTED_SAMPLES
                        * 100.0,
                }
            )


            
            # Model-level metrics
            

            for model_name, config in (
                MODELS.items()
            ):

                correct = correctness_array(
                    group[
                        f"{model_name}_top1_correct"
                    ]
                )

                mrr = (
                    group[
                        f"{model_name}_reciprocal_rank_at_5"
                    ]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

                ndcg = (
                    group[
                        f"{model_name}_ndcg_at_5"
                    ]
                    .to_numpy(
                        dtype=np.float64
                    )
                )


                metric_rows.append(
                    {
                        "stratum_type":
                            stratum_type,

                        "stratum":
                            stratum,

                        "model":
                            model_name,

                        "label":
                            config[
                                "label"
                            ],

                        "n":
                            n,

                        "accuracy_percent":
                            float(
                                correct.mean()
                                * 100.0
                            ),

                        "mrr_at_5_percent":
                            float(
                                mrr.mean()
                                * 100.0
                            ),

                        "ndcg_at_5_percent":
                            float(
                                ndcg.mean()
                                * 100.0
                            ),
                    }
                )


            
            # Pairwise comparisons
            

            for (
                model_a,
                model_b,
            ) in PAIRS:

                correct_a = (
                    correctness_array(
                        group[
                            f"{model_a}_top1_correct"
                        ]
                    )
                )

                correct_b = (
                    correctness_array(
                        group[
                            f"{model_b}_top1_correct"
                        ]
                    )
                )


                mrr_a = (
                    group[
                        f"{model_a}_reciprocal_rank_at_5"
                    ]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

                mrr_b = (
                    group[
                        f"{model_b}_reciprocal_rank_at_5"
                    ]
                    .to_numpy(
                        dtype=np.float64
                    )
                )


                ndcg_a = (
                    group[
                        f"{model_a}_ndcg_at_5"
                    ]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

                ndcg_b = (
                    group[
                        f"{model_b}_ndcg_at_5"
                    ]
                    .to_numpy(
                        dtype=np.float64
                    )
                )


                differences = np.column_stack(
                    [
                        (
                            correct_b.astype(
                                np.float64
                            )
                            -
                            correct_a.astype(
                                np.float64
                            )
                        ),

                        mrr_b
                        - mrr_a,

                        ndcg_b
                        - ndcg_a,
                    ]
                )


                point = (
                    differences.mean(
                        axis=0
                    )
                    * 100.0
                )


                low, high = (
                    paired_bootstrap_ci(
                        differences,
                        seed=(
                            BOOTSTRAP_SEED
                            + seed_counter
                        ),
                    )
                )

                seed_counter += 1


                low *= 100.0
                high *= 100.0


                corrections = int(
                    (
                        ~correct_a
                        & correct_b
                    )
                    .sum()
                )


                regressions = int(
                    (
                        correct_a
                        & ~correct_b
                    )
                    .sum()
                )


                agreement = int(
                    (
                        group[
                            f"{model_a}_top1_prediction"
                        ]
                        .to_numpy()
                        ==
                        group[
                            f"{model_b}_top1_prediction"
                        ]
                        .to_numpy()
                    )
                    .sum()
                )


                pairwise_rows.append(
                    {
                        "stratum_type":
                            stratum_type,

                        "stratum":
                            stratum,

                        "model_a":
                            model_a,

                        "model_b":
                            model_b,

                        "model_a_label":
                            MODELS[
                                model_a
                            ]["label"],

                        "model_b_label":
                            MODELS[
                                model_b
                            ]["label"],

                        "n":
                            n,

                        # Positive = B better.
                        "delta_accuracy_pp":
                            float(
                                point[0]
                            ),

                        "delta_accuracy_ci95_low_pp":
                            float(
                                low[0]
                            ),

                        "delta_accuracy_ci95_high_pp":
                            float(
                                high[0]
                            ),

                        "delta_mrr_at_5_pp":
                            float(
                                point[1]
                            ),

                        "delta_mrr_ci95_low_pp":
                            float(
                                low[1]
                            ),

                        "delta_mrr_ci95_high_pp":
                            float(
                                high[1]
                            ),

                        "delta_ndcg_at_5_pp":
                            float(
                                point[2]
                            ),

                        "delta_ndcg_ci95_low_pp":
                            float(
                                low[2]
                            ),

                        "delta_ndcg_ci95_high_pp":
                            float(
                                high[2]
                            ),

                        "corrections_a_wrong_b_correct":
                            corrections,

                        "regressions_a_correct_b_wrong":
                            regressions,

                        "net_corrections":
                            corrections
                            - regressions,

                        "same_top1_prediction_percent":
                            float(
                                agreement
                                / n
                                * 100.0
                            ),
                    }
                )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    pairwise_df = pd.DataFrame(
        pairwise_rows
    )

    support_df = pd.DataFrame(
        support_rows
    )

    enriched_path = (
        OUTPUT_DIR
        / "next_hop_per_sample_stratified.csv"
    )

    metrics_path = (
        OUTPUT_DIR
        / "next_hop_stratified_metrics.csv"
    )

    pairwise_path = (
        OUTPUT_DIR
        / "next_hop_stratified_pairwise.csv"
    )

    support_path = (
        OUTPUT_DIR
        / "next_hop_strata_support.csv"
    )

    manifest_path = (
        OUTPUT_DIR
        / "next_hop_stratification_manifest.json"
    )

    master.to_csv(
        enriched_path,
        index=False,
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    pairwise_df.to_csv(
        pairwise_path,
        index=False,
    )

    support_df.to_csv(
        support_path,
        index=False,
    )

    manifest = {
        "reproduction_profile": "historical" if HISTORICAL else "fresh",
        "task":
            "next_hop",

        "sample_count":
            EXPECTED_SAMPLES,

        "strata_locked_before_subgroup_performance_inspection":
            True,

        "context_length_groups":
            CONTEXT_LABELS,

        "semantic_coverage_groups":
            SEMANTIC_LABELS,

        "bootstrap": {
            "method":
                "paired_percentile_bootstrap",

            "repetitions":
                BOOTSTRAP_REPETITIONS,

            "confidence_level":
                0.95,

            "seed_base":
                BOOTSTRAP_SEED,

            "interpretation":
                (
                    "Paired test-sample uncertainty "
                    "within each stratum; not "
                    "training-seed uncertainty."
                ),
        },

        "test_cache_sha256":
            sha256_file(TEST_CACHE),

        "semantic_cache_sha256":
            sha256_file(SEMANTIC_CACHE),

        "semantic_tensor_location":
            location,

        "semantic_road_count":
            int(
                semantic_mask.sum()
            ),
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
        "============================================"
    )

    print(
        " CONTEXT-LENGTH PAIRWISE RESULTS"
    )

    print(
        " Positive delta = model B better"
    )

    print(
        "============================================"
    )

    context_results = (
        pairwise_df[
            pairwise_df[
                "stratum_type"
            ]
            == "context_length"
        ]
    )

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        350,
        "display.precision",
        5,
    ):

        print(
            context_results[
                [
                    "stratum",
                    "model_a_label",
                    "model_b_label",
                    "n",
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
                    "same_top1_prediction_percent",
                ]
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
        " SEMANTIC-COVERAGE PAIRWISE RESULTS"
    )

    print(
        " Positive delta = model B better"
    )

    print(
        "============================================"
    )

    semantic_results = (
        pairwise_df[
            pairwise_df[
                "stratum_type"
            ]
            == "semantic_coverage"
        ]
    )

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        350,
        "display.precision",
        5,
    ):

        print(
            semantic_results[
                [
                    "stratum",
                    "model_a_label",
                    "model_b_label",
                    "n",
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
                    "same_top1_prediction_percent",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print()

    print(
        "Outputs:"
    )

    print(
        enriched_path
    )

    print(
        metrics_path
    )

    print(
        pairwise_path
    )

    print(
        support_path
    )

    print(
        manifest_path
    )

    print()

    print(
        "STATUS: FINAL NEXT-HOP STRATIFIED ANALYSIS PASS"
    )


if __name__ == "__main__":
    main()
