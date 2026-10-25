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

from scipy.stats import binomtest

ROOT = Path("final_analysis/tte")

INPUT_FILE = (
    ROOT
    / "per_sample"
    / "tte_per_sample.csv"
)

OUTPUT_DIR = (
    ROOT
    / "statistics"
)

EXPECTED_SAMPLES = 76923

BOOTSTRAP_REPETITIONS = 2000

BOOTSTRAP_SEED = 2021

BOOTSTRAP_CHUNK_SIZE = 20

MODELS = {
    "baseline": {
        "label": "Baseline",
        "prediction":
            "prediction_baseline_seconds",
        "signed_error":
            "signed_error_baseline_seconds",
        "abs_error":
            "abs_error_baseline_seconds",
        "ape":
            "ape_baseline_percent",

        "expected_mae":
            95.4379781688284,

        "expected_rmse":
            156.9819219361769,

        "expected_mape":
            22.816461166436167,
    },

    "structured": {
        "label": "Structured",
        "prediction":
            "prediction_structured_seconds",
        "signed_error":
            "signed_error_structured_seconds",
        "abs_error":
            "abs_error_structured_seconds",
        "ape":
            "ape_structured_percent",

        "expected_mae":
            94.42844921622998,

        "expected_rmse":
            154.00982398476586,

        "expected_mape":
            24.22731526821154,
    },

    "qwen": {
        "label": "Qwen",
        "prediction":
            "prediction_qwen_seconds",
        "signed_error":
            "signed_error_qwen_seconds",
        "abs_error":
            "abs_error_qwen_seconds",
        "ape":
            "ape_qwen_percent",

        "expected_mae":
            94.86113049945142,

        "expected_rmse":
            155.63926646896786,

        "expected_mape":
            22.52686125942081,
    },
}

PAIRS = [
    ("baseline", "structured"),
    ("baseline", "qwen"),
    ("structured", "qwen"),
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

def model_metrics(
    frame,
    config,
):

    signed = frame[
        config["signed_error"]
    ].to_numpy(
        dtype=np.float64
    )

    absolute = frame[
        config["abs_error"]
    ].to_numpy(
        dtype=np.float64
    )

    ape = frame[
        config["ape"]
    ].to_numpy(
        dtype=np.float64
    )

    return {
        "mae_seconds":
            float(
                absolute.mean()
            ),

        "rmse_seconds":
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            signed
                        )
                    )
                )
            ),

        "mape_percent":
            float(
                ape.mean()
            ),

        "median_absolute_error_seconds":
            float(
                np.median(
                    absolute
                )
            ),

        "p90_absolute_error_seconds":
            float(
                np.quantile(
                    absolute,
                    0.90,
                )
            ),

        "p95_absolute_error_seconds":
            float(
                np.quantile(
                    absolute,
                    0.95,
                )
            ),

        "mean_signed_error_seconds":
            float(
                signed.mean()
            ),

        "median_signed_error_seconds":
            float(
                np.median(
                    signed
                )
            ),

        "within_30_seconds_percent":
            float(
                np.mean(
                    absolute <= 30.0
                )
                * 100.0
            ),

        "within_60_seconds_percent":
            float(
                np.mean(
                    absolute <= 60.0
                )
                * 100.0
            ),

        "within_120_seconds_percent":
            float(
                np.mean(
                    absolute <= 120.0
                )
                * 100.0
            ),
    }

def paired_bootstrap(
    signed_a,
    signed_b,
    abs_a,
    abs_b,
    ape_a,
    ape_b,
    seed,
):

    """
    Positive improvement means model B
    has lower error than model A.
    """

    n = len(
        signed_a
    )

    rng = np.random.default_rng(
        seed
    )

    values = np.empty(
        (
            BOOTSTRAP_REPETITIONS,
            3,
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


        # MAE improvement
        values[
            start:
            start + count,
            0,
        ] = (
            abs_a[
                indices
            ]
            .mean(
                axis=1
            )
            -
            abs_b[
                indices
            ]
            .mean(
                axis=1
            )
        )


        # RMSE improvement
        rmse_a = np.sqrt(
            np.mean(
                np.square(
                    signed_a[
                        indices
                    ]
                ),
                axis=1,
            )
        )

        rmse_b = np.sqrt(
            np.mean(
                np.square(
                    signed_b[
                        indices
                    ]
                ),
                axis=1,
            )
        )

        values[
            start:
            start + count,
            1,
        ] = (
            rmse_a
            - rmse_b
        )


        # MAPE improvement
        values[
            start:
            start + count,
            2,
        ] = (
            ape_a[
                indices
            ]
            .mean(
                axis=1
            )
            -
            ape_b[
                indices
            ]
            .mean(
                axis=1
            )
        )


    low = np.quantile(
        values,
        0.025,
        axis=0,
    )

    high = np.quantile(
        values,
        0.975,
        axis=0,
    )

    return (
        low,
        high,
    )

def exact_sign_test(
    wins,
    losses,
):

    discordant = (
        wins
        + losses
    )

    if discordant == 0:
        return 1.0

    return float(
        binomtest(
            min(
                wins,
                losses,
            ),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
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

        candidate = min(
            1.0,
            p_values[index]
            * (
                m - rank
            ),
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted[
            index
        ] = running_max

    return adjusted

HISTORICAL = True


def main(argv=None):
    global HISTORICAL, a_wins, abs_a, abs_b, ape_a, ape_b, b_wins, columns, config, config_a, config_b, eps, expected_key, expected_sample_ids, frame, handle, high, improvement_mae, improvement_mape, improvement_rmse, key, low, metrics, model_a, model_b, model_metrics_df, model_path, model_rows, name, pair_index, pairwise, pairwise_path, pairwise_rows, row, sample_ids, sign_test_p, signed_a, signed_b, summary, summary_path, ties
    from scripts.portable_runtime_support import analysis_cli
    analysis_cli(globals(), "tte", argv)

    print(
        "============================================"
    )

    print(
        " FINAL TTE PAIRED STATISTICAL ANALYSIS"
    )

    print(
        "============================================"
    )

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            INPUT_FILE
        )

    frame = pd.read_csv(
        INPUT_FILE
    )

    if len(frame) != EXPECTED_SAMPLES:
        raise RuntimeError(
            "Unexpected sample count."
        )

    if (
        frame["sample_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate sample IDs."
        )

    sample_ids = (
        frame["sample_id"]
        .to_numpy(
            dtype=np.int64
        )
    )

    expected_sample_ids = np.arange(
        307693,
        384616,
        dtype=np.int64,
    )

    if not np.array_equal(
        sample_ids,
        expected_sample_ids,
    ):
        raise RuntimeError(
            "Unexpected canonical TTE sample-ID ordering."
        )

    if (
        frame[
            "target_seconds"
        ]
        <= 0
    ).any():
        raise RuntimeError(
            "Non-positive target duration."
        )

    print(
        "Sample validation: PASS"
    )

    model_rows = []

    for name, config in MODELS.items():

        metrics = model_metrics(
            frame,
            config,
        )


        for key, expected_key in [
            (
                "mae_seconds",
                "expected_mae",
            ),
            (
                "rmse_seconds",
                "expected_rmse",
            ),
            (
                "mape_percent",
                "expected_mape",
            ),
        ]:

            if HISTORICAL and (abs(
                metrics[key]
                - config[
                    expected_key
                ]
            ) > 1e-5):

                raise RuntimeError(
                    f"{name}: {key} "
                    "reproduction failure."
                )


        row = {
            "model":
                name,

            "label":
                config["label"],

            "sample_count":
                EXPECTED_SAMPLES,
        }

        row.update(
            metrics
        )

        model_rows.append(
            row
        )

    model_metrics_df = pd.DataFrame(
        model_rows
    )

    print()

    print(
        "=== MODEL-LEVEL ERROR STATISTICS ==="
    )

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        300,
        "display.precision",
        6,
    ):

        print(
            model_metrics_df.to_string(
                index=False
            )
        )

    print()

    print(
        "Historical metric parity: PASS" if HISTORICAL else "Fresh metrics computed; historical parity is not required"
    )

    pairwise_rows = []

    for pair_index, (
        model_a,
        model_b,
    ) in enumerate(
        PAIRS
    ):

        config_a = MODELS[
            model_a
        ]

        config_b = MODELS[
            model_b
        ]


        signed_a = frame[
            config_a[
                "signed_error"
            ]
        ].to_numpy(
            dtype=np.float64
        )

        signed_b = frame[
            config_b[
                "signed_error"
            ]
        ].to_numpy(
            dtype=np.float64
        )


        abs_a = frame[
            config_a[
                "abs_error"
            ]
        ].to_numpy(
            dtype=np.float64
        )

        abs_b = frame[
            config_b[
                "abs_error"
            ]
        ].to_numpy(
            dtype=np.float64
        )


        ape_a = frame[
            config_a[
                "ape"
            ]
        ].to_numpy(
            dtype=np.float64
        )

        ape_b = frame[
            config_b[
                "ape"
            ]
        ].to_numpy(
            dtype=np.float64
        )


        # Positive = B better
        improvement_mae = (
            abs_a.mean()
            - abs_b.mean()
        )

        improvement_rmse = (
            np.sqrt(
                np.mean(
                    np.square(
                        signed_a
                    )
                )
            )
            -
            np.sqrt(
                np.mean(
                    np.square(
                        signed_b
                    )
                )
            )
        )

        improvement_mape = (
            ape_a.mean()
            - ape_b.mean()
        )


        low, high = paired_bootstrap(
            signed_a,
            signed_b,
            abs_a,
            abs_b,
            ape_a,
            ape_b,
            seed=(
                BOOTSTRAP_SEED
                + pair_index
            ),
        )


        eps = 1e-12

        b_wins = int(
            (
                abs_b
                < abs_a - eps
            )
            .sum()
        )

        a_wins = int(
            (
                abs_a
                < abs_b - eps
            )
            .sum()
        )

        ties = int(
            EXPECTED_SAMPLES
            - a_wins
            - b_wins
        )


        sign_test_p = (
            exact_sign_test(
                b_wins,
                a_wins,
            )
        )


        pairwise_rows.append(
            {
                "comparison":
                    f"{model_a}_vs_{model_b}",

                "model_a":
                    config_a[
                        "label"
                    ],

                "model_b":
                    config_b[
                        "label"
                    ],

                "sample_count":
                    EXPECTED_SAMPLES,

                # Positive means B better.
                "mae_improvement_seconds":
                    improvement_mae,

                "mae_ci95_low_seconds":
                    low[0],

                "mae_ci95_high_seconds":
                    high[0],

                "rmse_improvement_seconds":
                    improvement_rmse,

                "rmse_ci95_low_seconds":
                    low[1],

                "rmse_ci95_high_seconds":
                    high[1],

                "mape_improvement_pp":
                    improvement_mape,

                "mape_ci95_low_pp":
                    low[2],

                "mape_ci95_high_pp":
                    high[2],

                "b_lower_abs_error_count":
                    b_wins,

                "a_lower_abs_error_count":
                    a_wins,

                "equal_abs_error_count":
                    ties,

                "b_lower_abs_error_percent":
                    (
                        b_wins
                        / EXPECTED_SAMPLES
                        * 100.0
                    ),

                "a_lower_abs_error_percent":
                    (
                        a_wins
                        / EXPECTED_SAMPLES
                        * 100.0
                    ),

                "sign_test_p_raw":
                    sign_test_p,
            }
        )

    pairwise = pd.DataFrame(
        pairwise_rows
    )

    pairwise[
        "sign_test_p_holm"
    ] = holm_adjust(
        pairwise[
            "sign_test_p_raw"
        ].to_numpy()
    )

    model_path = (
        OUTPUT_DIR
        / "tte_model_error_statistics.csv"
    )

    pairwise_path = (
        OUTPUT_DIR
        / "tte_pairwise_statistics.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "tte_statistics_summary.json"
    )

    model_metrics_df.to_csv(
        model_path,
        index=False,
    )

    pairwise.to_csv(
        pairwise_path,
        index=False,
    )

    summary = {
        "reproduction_profile": "historical" if HISTORICAL else "fresh",
        "historical_metric_parity_verified": HISTORICAL,
        "task":
            "tte",

        "sample_count":
            EXPECTED_SAMPLES,

        "source_csv":
            str(
                INPUT_FILE
            ),

        "source_csv_sha256":
            sha256_file(
                INPUT_FILE
            ),

        "paired_analysis":
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
                    "test trips, not across "
                    "training seeds."
                ),
        },

        "sign_test": {
            "method":
                (
                    "exact two-sided binomial "
                    "sign test on absolute-error "
                    "wins/losses"
                ),

            "multiple_comparison_adjustment":
                "Holm",
        },

        "outputs": {
            "model_statistics":
                str(
                    model_path
                ),

            "pairwise_statistics":
                str(
                    pairwise_path
                ),
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
        " Positive improvement = model B better"
    )

    print(
        "============================================"
    )

    columns = [
        "model_a",
        "model_b",

        "mae_improvement_seconds",
        "mae_ci95_low_seconds",
        "mae_ci95_high_seconds",

        "rmse_improvement_seconds",
        "rmse_ci95_low_seconds",
        "rmse_ci95_high_seconds",

        "mape_improvement_pp",
        "mape_ci95_low_pp",
        "mape_ci95_high_pp",

        "b_lower_abs_error_count",
        "a_lower_abs_error_count",
        "equal_abs_error_count",

        "b_lower_abs_error_percent",

        "sign_test_p_raw",
        "sign_test_p_holm",
    ]

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        320,
        "display.precision",
        8,
    ):

        print(
            pairwise[
                columns
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
        "Model statistics:",
        model_path,
    )

    print(
        "Pairwise statistics:",
        pairwise_path,
    )

    print(
        "Summary:",
        summary_path,
    )

    print()

    print(
        "STATUS: FINAL TTE STATISTICAL ANALYSIS PASS"
    )


if __name__ == "__main__":
    main()
