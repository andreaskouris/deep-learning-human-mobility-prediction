from __future__ import annotations

# Support direct script execution without depending on the caller PYTHONPATH.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import gc

from scripts.portable_runtime_support import add_execution_paths, bind_checkpoints, prepare_output, validate_checkpoint_metadata

import hashlib

import json

import sys

from pathlib import Path

import numpy as np

import pandas as pd

import torch

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_ROOT = ROOT.parent / "dataset" / "xa"

TEST_CACHE = (
    DATA_ROOT
    / "cached_tte_dataset_test_seq64_sr1_seed2021_v1.pth"
)

STRUCTURED_CACHE = (
    DATA_ROOT
    / "cached_xa_tte_semantics_binary_v1.pth"
)

QWEN_CACHE = (
    DATA_ROOT
    / "cached_xa_tte_semantics_qwen_v1.pth"
)

PRETRAIN = (
    ROOT
    / "checkpoints"
    / "xa_pretrain.pth"
)

OUTPUT_DIR = (
    ROOT
    / "final_analysis"
    / "tte"
    / "per_sample"
)

MODELS = {
    "baseline": {
        "label": "TTE-1 Baseline",
        "semantic_mode": "none",
        "checkpoint": (
            ROOT
            / "final_results"
            / "tte"
            / "01_baseline"
            / "checkpoint"
            / "best_tte_none_seed2021.pth"
        ),
        "checkpoint_sha256":
            "D3D3D03A116943F5BCD4DC7CEBB6FAE8B786643B90A942B06B17A43F4560B37E",
        "best_epoch": 16,
        "mae": 95.4379781688284,
        "rmse": 156.9819219361769,
        "mape": 22.816461166436167,
    },

    "structured": {
        "label": "TTE-2 Structured",
        "semantic_mode": "tte_structured",
        "checkpoint": (
            ROOT
            / "final_results"
            / "tte"
            / "02_structured"
            / "checkpoint"
            / "best_tte_tte_structured_seed2021.pth"
        ),
        "checkpoint_sha256":
            "1425066115D9BF346208A3D0F5F34DCA8CF915E82177B008EB8185264B58B142",
        "best_epoch": 11,
        "mae": 94.42844921622998,
        "rmse": 154.00982398476586,
        "mape": 24.22731526821154,
    },

    "qwen": {
        "label": "TTE-3 Qwen",
        "semantic_mode": "tte_qwen",
        "checkpoint": (
            ROOT
            / "final_results"
            / "tte"
            / "03_qwen"
            / "checkpoint"
            / "best_tte_tte_qwen_seed2021.pth"
        ),
        "checkpoint_sha256":
            "75AEF28F6CB81EDF9315583E1A129F5BA9F03E465A11C295EF4EB43669DC713C",
        "best_epoch": 13,
        "mae": 94.86113049945142,
        "rmse": 155.63926646896786,
        "mape": 22.52686125942081,
    },
}

EXPECTED_HASHES = {
    TEST_CACHE:
        "72E4B427CC9EDD5F66ACFD8931E5A495583EA58265B62F5B0093AFA5058AA0DA",

    STRUCTURED_CACHE:
        "081952F5ABD0A5E005430113EACC7E516A76CF2210E8FF346039508982416717",

    QWEN_CACHE:
        "31FA484DDB3FC9423F44C495E8A26366C4EBD21518124F28E33902EE06966DAF",

    PRETRAIN:
        "612A663D81A3EB3173173D242023D7C4B86B650D103DE9EFB868FCFE0C2DBDA4",
}

HISTORICAL = True
CHECKPOINT_EPOCHS = {}

FEATURE_NAMES = [
    "tunnel",
    "bridge",
    "roundabout",
    "oneway",
    "fuel_station",
    "parking",
    "transport_hub",
    "taxi_stand",
    "transit_stop",
    "logistics_freight",
]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest().upper()

def metric_block(prediction, target):
    error = prediction - target
    absolute_error = np.abs(error)

    return {
        "mae_seconds":
            float(np.mean(absolute_error)),

        "rmse_seconds":
            float(
                np.sqrt(
                    np.mean(np.square(error))
                )
            ),

        "mape_percent":
            float(
                np.mean(
                    absolute_error
                    / np.abs(target)
                )
                * 100.0
            ),
    }

def verify_metrics(name, actual):
    expected = MODELS[name]

    checks = {
        "mae_seconds": expected["mae"],
        "rmse_seconds": expected["rmse"],
        "mape_percent": expected["mape"],
    }

    for metric, expected_value in checks.items():

        actual_value = actual[metric]

        if HISTORICAL and (abs(
            actual_value
            - expected_value
        ) > 0.02):

            raise RuntimeError(
                f"{name} {metric} reproduction failed. "
                f"Expected {expected_value}, "
                f"found {actual_value}."
            )

def predict_checkpoint(
    *,
    checkpoint_path,
    semantic_mode,
    payload,
    device,
):

    print()
    print(
        "============================================"
    )
    print(
        f" PREDICTING {semantic_mode.upper()}"
    )
    print(
        "============================================"
    )

    args.semantic_mode = semantic_mode

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    validate_checkpoint_metadata(checkpoint, semantic_mode, "tte")
    CHECKPOINT_EPOCHS[semantic_mode] = int(checkpoint["epoch"])

    normalization = checkpoint[
        "target_normalization"
    ]

    mean_seconds = float(
        normalization["mean_seconds"]
    )

    std_seconds = float(
        normalization["std_seconds"]
    )

    if abs(
        mean_seconds
        - 485.50155132816224
    ) > 1e-9:
        raise RuntimeError(
            "Unexpected target mean."
        )

    if abs(
        std_seconds
        - 350.44401804624164
    ) > 1e-9:
        raise RuntimeError(
            "Unexpected target std."
        )

    model = BigCity4TTE(
        device,
        checkpoint=str(PRETRAIN),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    road_ids_all = payload["road_ids"]
    time_ids_all = payload["time_ids"]
    time_features_all = payload["time_features"]
    valid_mask_all = payload["valid_mask"]

    sample_count = int(
        road_ids_all.shape[0]
    )

    predictions = np.empty(
        sample_count,
        dtype=np.float64,
    )

    batch_size = 32

    with torch.inference_mode():

        for start in tqdm(
            range(
                0,
                sample_count,
                batch_size,
            ),
            total=(
                sample_count
                + batch_size
                - 1
            ) // batch_size,
            desc=semantic_mode,
        ):

            end = min(
                start + batch_size,
                sample_count,
            )

            road_ids = (
                road_ids_all[start:end]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            time_ids = (
                time_ids_all[start:end]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            time_features = (
                time_features_all[start:end]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            valid_mask = (
                valid_mask_all[start:end]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            prediction_normalized = (
                model.forward_tte(
                    road_ids,
                    time_ids,
                    time_features,
                    valid_mask,
                )
            )

            prediction_seconds = (
                prediction_normalized
                * std_seconds
                + mean_seconds
            )

            predictions[start:end] = (
                prediction_seconds
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64,
                    copy=False,
                )
            )

    del model
    del checkpoint

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not np.isfinite(
        predictions
    ).all():
        raise RuntimeError(
            "Non-finite predictions."
        )

    return predictions

def main(argv=None):
    global BigCity4TTE, DATA_ROOT, EXPECTED_HASHES, HISTORICAL, MODELS, OUTPUT_DIR, PRETRAIN, QWEN_CACHE, STRUCTURED_CACHE, TEST_CACHE, abs_error, actual_hash, ape, args, concept_name, concept_presence, config, csv_path, data, device, distinct_concept_count, expected_hash, handle, index, lengths, metadata, metrics, name, path, payload, pred, predictions, required_keys, road_concepts, road_ids, safe_road_ids, sample_ids, semantic_exposure_fraction, semantic_features, semantic_payload, semantic_road_mask, semantic_token_count, semantic_token_mask, signed_error, summary, summary_path, target, train_tte, valid_mask
    parser = add_execution_paths(argparse.ArgumentParser(description="TTE per-sample export"), output_default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    cli = parser.parse_args(argv)
    HISTORICAL = cli.profile == "historical"
    DATA_ROOT = cli.dataset_path.resolve() / "xa"
    TEST_CACHE, STRUCTURED_CACHE, QWEN_CACHE = [DATA_ROOT / p.name for p in (TEST_CACHE, STRUCTURED_CACHE, QWEN_CACHE)]
    PRETRAIN = cli.pretrained_checkpoint.resolve()
    OUTPUT_DIR = cli.output_dir.resolve()
    MODELS = bind_checkpoints(cli, MODELS, "tte")
    EXPECTED_HASHES = {TEST_CACHE: "72E4B427CC9EDD5F66ACFD8931E5A495583EA58265B62F5B0093AFA5058AA0DA", STRUCTURED_CACHE: "081952F5ABD0A5E005430113EACC7E516A76CF2210E8FF346039508982416717", QWEN_CACHE: "31FA484DDB3FC9423F44C495E8A26366C4EBD21518124F28E33902EE06966DAF", PRETRAIN: "612A663D81A3EB3173173D242023D7C4B86B650D103DE9EFB868FCFE0C2DBDA4"}
    from scripts.portable_runtime_support import require_hash
    require_hash(PRETRAIN, EXPECTED_HASHES[PRETRAIN])
    prepare_output(OUTPUT_DIR, overwrite=cli.overwrite, protected=[TEST_CACHE, STRUCTURED_CACHE, QWEN_CACHE, PRETRAIN, *[c["checkpoint"] for c in MODELS.values()]])
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], "--city", "xa", "--split", "test", "--seq_len", "64", "--sample_rate", "1", "--seed", "2021", "--dynamic_mode", "controlled", "--static_ablation", "none", "--semantic_mode", "none", "--batch_size", "32", "--learning_rate", "0.0002", "--weight", "0", "--train_epochs", "20", "--patience", "3", "--use_gpu", "--device", "0", "--pretrained_checkpoint", str(PRETRAIN), "--dataset_path", str(cli.dataset_path.resolve()), "--gpt2_path", str(cli.gpt2_path.resolve()), "--wandb_mode", "offline"]
        from config.args_config import args
        from models.bigcity4tte import BigCity4TTE
        import train_tte
    finally:
        sys.argv = original_argv

    print(
        "============================================"
    )

    print(
        " FINAL TTE PER-SAMPLE EXPORT"
    )

    print(
        "============================================"
    )

    print()

    print("=== ARTIFACT INTEGRITY ===")

    for path, expected_hash in EXPECTED_HASHES.items():

        if not path.is_file():
            raise FileNotFoundError(path)

        actual_hash = sha256_file(path)

        print(
            f"{path.name}: {actual_hash}"
        )

        if HISTORICAL and (actual_hash != expected_hash):
            raise RuntimeError(
                f"SHA256 mismatch: {path}"
            )

    for name, config in MODELS.items():

        path = config["checkpoint"]

        if not path.is_file():
            raise FileNotFoundError(path)

        actual_hash = sha256_file(path)

        print(
            f"{name} checkpoint: "
            f"{actual_hash}"
        )

        if (
            actual_hash
            != config[
                "checkpoint_sha256"
            ]
        ):
            raise RuntimeError(
                f"{name} checkpoint SHA mismatch."
            )

    print(
        "Artifact integrity: PASS"
    )

    train_tte.load_model_support_data()

    print(
        "BIGCity support data: PASS"
    )

    payload = torch.load(
        TEST_CACHE,
        map_location="cpu",
        weights_only=True,
    )

    required_keys = {
        "road_ids",
        "time_ids",
        "time_features",
        "valid_mask",
        "lengths",
        "departure_timestamps",
        "duration_seconds",
        "sample_ids",
        "metadata",
    }

    if set(payload.keys()) != required_keys:
        raise RuntimeError(
            "Unexpected TTE test cache keys."
        )

    from data_provider.tte_dataset import validate_tte_payload
    validate_tte_payload(payload)
    metadata = payload["metadata"]

    if metadata["task"] != "tte":
        raise RuntimeError(
            "Unexpected TTE task metadata."
        )

    if metadata["split"] != "test":
        raise RuntimeError(
            "Unexpected split metadata."
        )

    if int(
        metadata["sample_count"]
    ) != 76923:
        raise RuntimeError(
            "Unexpected sample count."
        )

    if int(
        metadata["seq_len"]
    ) != 64:
        raise RuntimeError(
            "Unexpected sequence length."
        )

    if (
        metadata["temporal_input"]
        != "departure_only"
    ):
        raise RuntimeError(
            "Unexpected temporal input."
        )

    if (
        metadata["source_sha256"]
        !=
        "71F0BA280E9F4C40F7CBBE355930EA1A82AD45E11AFC2F31E6B2506440F47568"
    ):
        raise RuntimeError(
            "Unexpected raw test source SHA."
        )

    print(
        "Canonical TTE test cache: PASS"
    )

    semantic_payload = torch.load(
        STRUCTURED_CACHE,
        map_location="cpu",
        weights_only=True,
    )

    if not HISTORICAL:
        from scripts.build_tte_structured_semantic_cache import validate_payload
        validate_payload(semantic_payload, canonical=True)

    semantic_features = (
        semantic_payload[
            "semantic_features"
        ]
        .to(torch.bool)
    )

    if tuple(
        semantic_features.shape
    ) != (5269, 10):
        raise RuntimeError(
            "Unexpected structured semantic shape."
        )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "CUDA is required for final TTE export."
        )

    print(
        "Device:",
        device,
    )

    predictions = {}

    metrics = {}

    for name, config in MODELS.items():

        predictions[name] = predict_checkpoint(
            checkpoint_path=
                config["checkpoint"],

            semantic_mode=
                config["semantic_mode"],

            payload=
                payload,

            device=
                device,
        )

    target = (
        payload["duration_seconds"]
        .numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    if np.any(target <= 0):
        raise RuntimeError(
            "Non-positive TTE target detected."
        )

    print()

    print(
        "============================================"
    )

    print(
        " METRIC REPRODUCTION"
    )

    print(
        "============================================"
    )

    for name in MODELS:

        metrics[name] = metric_block(
            predictions[name],
            target,
        )

        print()
        print(
            MODELS[name]["label"]
        )

        print(
            "MAE :",
            f"{metrics[name]['mae_seconds']:.6f}",
            "seconds",
        )

        print(
            "RMSE:",
            f"{metrics[name]['rmse_seconds']:.6f}",
            "seconds",
        )

        print(
            "MAPE:",
            f"{metrics[name]['mape_percent']:.6f}",
            "%",
        )

        verify_metrics(
            name,
            metrics[name],
        )

        print(
            "Historical metric parity: PASS" if HISTORICAL else "Fresh metrics computed; historical parity is not required"
        )

    road_ids = payload["road_ids"]

    valid_mask = payload["valid_mask"]

    lengths = payload["lengths"]

    safe_road_ids = road_ids.clone()

    safe_road_ids[~valid_mask] = 0

    semantic_road_mask = (
        semantic_features.any(dim=1)
    )

    semantic_token_mask = (
        semantic_road_mask[
            safe_road_ids
        ]
        & valid_mask
    )

    semantic_token_count = (
        semantic_token_mask.sum(dim=1)
    )

    semantic_exposure_fraction = (
        semantic_token_count.float()
        / lengths.float()
    )

    road_concepts = (
        semantic_features[
            safe_road_ids
        ]
        & valid_mask.unsqueeze(-1)
    )

    concept_presence = (
        road_concepts.any(dim=1)
    )

    distinct_concept_count = (
        concept_presence.sum(dim=1)
    )

    sample_ids = payload["sample_ids"]

    if torch.is_tensor(sample_ids):
        sample_ids = (
            sample_ids
            .cpu()
            .numpy()
        )

    data = pd.DataFrame(
        {
            "sample_id":
                sample_ids,

            "target_seconds":
                target,

            "route_length_roads":
                lengths.numpy(),

            "semantic_token_count":
                semantic_token_count.numpy(),

            "semantic_exposure_fraction":
                semantic_exposure_fraction.numpy(),

            "distinct_semantic_concept_count":
                distinct_concept_count.numpy(),
        }
    )

    for name in MODELS:

        pred = predictions[name]

        signed_error = (
            pred - target
        )

        abs_error = np.abs(
            signed_error
        )

        ape = (
            abs_error
            / target
            * 100.0
        )

        data[
            f"prediction_{name}_seconds"
        ] = pred

        data[
            f"signed_error_{name}_seconds"
        ] = signed_error

        data[
            f"abs_error_{name}_seconds"
        ] = abs_error

        data[
            f"ape_{name}_percent"
        ] = ape

    for index, concept_name in enumerate(
        FEATURE_NAMES
    ):

        data[
            f"has_{concept_name}"
        ] = (
            concept_presence[
                :,
                index,
            ]
            .numpy()
            .astype(bool)
        )

    if len(data) != 76923:
        raise RuntimeError(
            "Unexpected final dataframe size."
        )

    csv_path = (
        OUTPUT_DIR
        / "tte_per_sample.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "tte_per_sample_summary.json"
    )

    data.to_csv(
        csv_path,
        index=False,
    )

    summary = {
        "reproduction_profile": cli.profile,
        "task":
            "tte",

        "sample_count":
            int(len(data)),

        "models":
            {
                name: {
                    "label":
                        config["label"],

                    "semantic_mode":
                        config["semantic_mode"],

                    "best_epoch":
                        CHECKPOINT_EPOCHS[config["semantic_mode"]],

                    "checkpoint":
                        str(
                            config["checkpoint"]
                        ),

                    "checkpoint_sha256":
                        config[
                            "checkpoint_sha256"
                        ],

                    "reconstructed_metrics":
                        metrics[name],
                }

                for name, config
                in MODELS.items()
            },

        "test_cache_sha256":
            sha256_file(TEST_CACHE),

        "structured_cache_sha256":
            sha256_file(STRUCTURED_CACHE),

        "qwen_cache_sha256":
            sha256_file(QWEN_CACHE),

        "pretrained_sha256":
            sha256_file(PRETRAIN),

        "output_csv_sha256":
            sha256_file(
                csv_path
            ),
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
        " FINAL EXPORT COMPLETE"
    )

    print(
        "============================================"
    )

    print(
        "Samples:",
        len(data),
    )

    print(
        "Output:",
        csv_path,
    )

    print(
        "Output SHA256:",
        sha256_file(csv_path),
    )

    print(
        "Summary:",
        summary_path,
    )

    print()

    print(
        "STATUS: FINAL TTE PER-SAMPLE EXPORT PASS"
    )


if __name__ == "__main__":
    main()
