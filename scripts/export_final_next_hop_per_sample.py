from __future__ import annotations

# Support direct script execution without depending on the caller PYTHONPATH.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
from scripts.portable_runtime_support import add_execution_paths, bind_checkpoints, prepare_output, validate_checkpoint_metadata
import hashlib
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]

PRETRAINED = (
    ROOT
    / "checkpoints"
    / "xa_pretrain.pth"
)

TEST_CACHE = (
    ROOT.parent
    / "dataset"
    / "xa"
    / "cached_next_hop_dataset_test_seq64_sr1_seed2021_v2.pth"
)

OUTPUT_DIR = (
    ROOT
    / "final_analysis"
    / "next_hop"
    / "per_sample"
)

EXPECTED_PRETRAINED_SHA256 = (
    "612A663D81A3EB3173173D242023D7C4B86B650D103DE9EFB868FCFE0C2DBDA4"
)

EXPECTED_TEST_CACHE_SHA256 = (
    "FAFCBB6A7DB3BEC52A091744A8971E24EAD3790FE705A662875D30F2C5BD2F04"
)

EXPERIMENTS = {
    "nh1": {
        "name": "nh1_baseline",
        "semantic_mode": "none",
        "checkpoint": (
            ROOT
            / "final_results"
            / "next_hop"
            / "01_baseline"
            / "checkpoint"
            / "best_next_hop.pth"
        ),
        "checkpoint_sha256":
            "E1AC9DD9010D53814D4B0FB5B79FA38E3D7949EB591E85F3163B5815377580EE",
        "best_epoch": 10,
        "accuracy_pct": "77.9143",
        "mrr_pct": "86.7511",
        "ndcg_pct": "89.5217",
    },

    "nh2": {
        "name": "nh2_poi_tfidf",
        "semantic_mode": "poi_tfidf",
        "checkpoint": (
            ROOT
            / "final_results"
            / "next_hop"
            / "02_poi_tfidf"
            / "checkpoint"
            / "best_next_hop.pth"
        ),
        "checkpoint_sha256":
            "FE8D2A506ECDA4D0EC4595003B5680D0A8C591D21DA5BE7E90B786CAAF0EECC1",
        "best_epoch": 6,
        "accuracy_pct": "79.5575",
        "mrr_pct": "88.4665",
        "ndcg_pct": "91.1168",
    },

    "nh3": {
        "name": "nh3_qwen_poi",
        "semantic_mode": "poi_qwen_tfidf",
        "checkpoint": (
            ROOT
            / "final_results"
            / "next_hop"
            / "03_qwen_poi"
            / "checkpoint"
            / "best_next_hop.pth"
        ),
        "checkpoint_sha256":
            "339A6734FE4A02AF870DE5FFDAA7FF7F2BF32900C93BD8689ACBC040BD51676A",
        "best_epoch": 8,
        "accuracy_pct": "79.8240",
        "mrr_pct": "88.5598",
        "ndcg_pct": "91.1645",
    },
}

EXPECTED_SAMPLES = 76923
HISTORICAL = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for chunk in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest().upper()


def parse_cli(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Export locked next-hop predictions "
            "at per-sample level."
        )
    )

    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS.keys()),
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    add_execution_paths(parser, output_default=OUTPUT_DIR)
    return parser.parse_args(argv)


def configure_bigcity_runtime(config):
    sys.argv = [
        str(Path(__file__).name),

        "--dataset_path", str(DATASET_PATH),
        "--gpt2_path", str(GPT2_PATH),
        "--city", "xa",
        "--seq_len", "64",
        "--sample_rate", "1.0",
        "--dynamic_mode", "controlled",
        "--seed", "2021",
        "--batch_size", "32",
        "--learning_rate", "0.0002",
        "--weight", "0",
        "--train_epochs", "20",
        "--patience", "3",
        "--static_ablation", "none",
        "--semantic_mode", config["semantic_mode"],
        "--pretrained_checkpoint", str(PRETRAINED),
        "--checkpoint_path",
        str(
            config["checkpoint"].parent
        ),
        "--use_gpu",
    ]


def main(argv=None):
    global PRETRAINED, TEST_CACHE, OUTPUT_DIR, EXPERIMENTS, HISTORICAL, DATASET_PATH, GPT2_PATH
    cli = parse_cli(argv)
    HISTORICAL = cli.profile == "historical"
    DATASET_PATH, GPT2_PATH = cli.dataset_path.resolve(), cli.gpt2_path.resolve()
    PRETRAINED = cli.pretrained_checkpoint.resolve()
    TEST_CACHE = DATASET_PATH / "xa" / TEST_CACHE.name
    OUTPUT_DIR = cli.output_dir.resolve()
    EXPERIMENTS = bind_checkpoints(cli, EXPERIMENTS, "next_hop")
    prepare_output(OUTPUT_DIR, overwrite=cli.overwrite, protected=[PRETRAINED, TEST_CACHE, *[c["checkpoint"] for c in EXPERIMENTS.values()]])

    config = EXPERIMENTS[
        cli.experiment
    ]

    checkpoint = config["checkpoint"]

    output_csv = (
        OUTPUT_DIR
        / f"{config['name']}_predictions.csv"
    )

    print(
        f"=== {cli.experiment.upper()} "
        "PER-SAMPLE PREDICTION EXPORT ==="
    )

    for path in (
        PRETRAINED,
        TEST_CACHE,
        checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        output_csv.exists()
        and not cli.overwrite
    ):
        raise FileExistsError(
            f"Output already exists: {output_csv}"
        )

    print()
    print("=== PROVENANCE ===")

    pretrained_sha = sha256_file(
        PRETRAINED
    )

    test_sha = sha256_file(
        TEST_CACHE
    )

    checkpoint_sha = sha256_file(
        checkpoint
    )

    print(
        "Pretrained SHA256:",
        pretrained_sha,
    )

    print(
        "Pretrained parity:",
        "PASS"
        if pretrained_sha
        == EXPECTED_PRETRAINED_SHA256
        else "FAIL",
    )

    print(
        "Test cache SHA256:",
        test_sha,
    )

    print(
        "Test cache parity:",
        "PASS"
        if test_sha
        == EXPECTED_TEST_CACHE_SHA256
        else "FAIL",
    )

    print(
        "Checkpoint SHA256:",
        checkpoint_sha,
    )

    print(
        "Checkpoint parity:",
        "PASS"
        if checkpoint_sha
        == config["checkpoint_sha256"]
        else "FAIL",
    )

    if (
        pretrained_sha
        != EXPECTED_PRETRAINED_SHA256
    ):
        raise RuntimeError(
            "Pretrained checkpoint hash mismatch."
        )

    if (
        HISTORICAL and (test_sha
        != EXPECTED_TEST_CACHE_SHA256)
    ):
        raise RuntimeError(
            "Test cache hash mismatch."
        )

    if (
        checkpoint_sha
        != config["checkpoint_sha256"]
    ):
        raise RuntimeError(
            "Final checkpoint hash mismatch."
        )

    original_argv = sys.argv[:]
    configure_bigcity_runtime(
        config
    )

    if str(ROOT) not in sys.path:
        sys.path.insert(
            0,
            str(ROOT),
        )

    try:
        import train_next_hop as training
    finally:
        sys.argv = original_argv

    from utils.next_hop_metrics import (
        NextHopMetricAccumulator,
    )

    device = (
        training.resolve_device()
    )

    print()
    print("=== RUNTIME ===")
    print("Device:", device)
    print(
        "Semantic mode:",
        training.args.semantic_mode,
    )
    print(
        "Static ablation:",
        training.args.static_ablation,
    )
    print(
        "Batch size:",
        training.args.batch_size,
    )

    if device.type != "cuda":
        raise RuntimeError(
            f"Expected CUDA, got {device}"
        )

    if (
        training.args.semantic_mode
        != config["semantic_mode"]
    ):
        raise RuntimeError(
            "Semantic mode mismatch."
        )

    training.load_model_support_data()

    dataset = (
        training.CachedNextHopDataset(
            str(TEST_CACHE)
        )
    )

    if len(dataset) != EXPECTED_SAMPLES:
        raise RuntimeError(
            "Unexpected test sample count."
        )

    dataloader = (
        training.create_dataloader(
            dataset,
            shuffle=False,
            device=device,
        )
    )

    pretrained_path = (
        training.resolve_pretrained_checkpoint()
    )

    model = (
        training.BigCity4FineTune(
            device,
            checkpoint=pretrained_path,
        ).to(device)
    )

    selected_checkpoint = (
        training.load_best_checkpoint(
            str(checkpoint),
            model,
            device,
        )
    )

    validate_checkpoint_metadata(selected_checkpoint, config["semantic_mode"], "next_hop")

    selected_epoch = int(
        selected_checkpoint["epoch"]
    )

    print()
    print(
        "Checkpoint epoch:",
        selected_epoch,
    )

    if (
        HISTORICAL and (selected_epoch
        != config["best_epoch"])
    ):
        raise RuntimeError(
            "Unexpected best checkpoint epoch: "
            f"{selected_epoch}"
        )

    model.eval()

    accumulator = (
        NextHopMetricAccumulator(
            k=5
        )
    )

    temporary_output = (
        output_csv.with_suffix(
            output_csv.suffix + ".tmp"
        )
    )

    temporary_output.unlink(
        missing_ok=True
    )

    fieldnames = [
        "sample_id",
        "true_next_road",
        "top1_prediction",
        "top1_correct",
        "rank_at_5",
        "reciprocal_rank_at_5",
        "ndcg_at_5",
        "top5_1",
        "top5_2",
        "top5_3",
        "top5_4",
        "top5_5",
    ]

    sample_offset = 0

    print()
    print(
        f"Running locked "
        f"{cli.experiment.upper()} inference..."
    )

    with temporary_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        with torch.no_grad():

            for batch in tqdm(
                dataloader,
                total=len(dataloader),
                unit="batch",
                desc=(
                    f"{cli.experiment.upper()} "
                    "test export"
                ),
            ):

                (
                    road_ids,
                    time_ids,
                    time_features,
                    labels,
                ) = (
                    training.move_batch_to_device(
                        batch,
                        device,
                    )
                )

                loss, logits = model(
                    training.TASK_NAME,
                    road_ids,
                    time_ids,
                    time_features,
                    labels,
                )

                if not torch.isfinite(
                    loss
                ):
                    raise RuntimeError(
                        "Non-finite loss."
                    )

                if not torch.isfinite(
                    logits
                ).all():
                    raise RuntimeError(
                        "Non-finite logits."
                    )

                accumulator.update(
                    logits,
                    labels,
                )

                top5 = torch.topk(
                    logits,
                    k=5,
                    dim=1,
                ).indices

                matches = (
                    top5
                    == labels.unsqueeze(1)
                )

                matched = (
                    matches.any(dim=1)
                )

                ranks = torch.zeros(
                    labels.shape[0],
                    dtype=torch.int64,
                    device=device,
                )

                if matched.any():

                    ranks[matched] = (
                        matches[matched]
                        .float()
                        .argmax(dim=1)
                        .to(torch.int64)
                        + 1
                    )

                rr = torch.zeros(
                    labels.shape[0],
                    dtype=torch.float32,
                    device=device,
                )

                ndcg = torch.zeros(
                    labels.shape[0],
                    dtype=torch.float32,
                    device=device,
                )

                if matched.any():

                    matched_ranks = (
                        ranks[matched]
                        .float()
                    )

                    rr[matched] = (
                        1.0
                        / matched_ranks
                    )

                    ndcg[matched] = (
                        1.0
                        / torch.log2(
                            matched_ranks + 1.0
                        )
                    )

                labels_cpu = labels.cpu()
                top5_cpu = top5.cpu()
                ranks_cpu = ranks.cpu()
                rr_cpu = rr.cpu()
                ndcg_cpu = ndcg.cpu()

                batch_size = (
                    labels_cpu.shape[0]
                )

                for row_index in range(
                    batch_size
                ):

                    top5_row = (
                        top5_cpu[
                            row_index
                        ].tolist()
                    )

                    true_label = int(
                        labels_cpu[
                            row_index
                        ]
                    )

                    top1 = int(
                        top5_row[0]
                    )

                    writer.writerow(
                        {
                            "sample_id":
                                sample_offset
                                + row_index,

                            "true_next_road":
                                true_label,

                            "top1_prediction":
                                top1,

                            "top1_correct":
                                int(
                                    top1
                                    == true_label
                                ),

                            "rank_at_5":
                                int(
                                    ranks_cpu[
                                        row_index
                                    ]
                                ),

                            "reciprocal_rank_at_5":
                                (
                                    f"{float(rr_cpu[row_index]):.9f}"
                                ),

                            "ndcg_at_5":
                                (
                                    f"{float(ndcg_cpu[row_index]):.9f}"
                                ),

                            "top5_1":
                                int(
                                    top5_row[0]
                                ),

                            "top5_2":
                                int(
                                    top5_row[1]
                                ),

                            "top5_3":
                                int(
                                    top5_row[2]
                                ),

                            "top5_4":
                                int(
                                    top5_row[3]
                                ),

                            "top5_5":
                                int(
                                    top5_row[4]
                                ),
                        }
                    )

                sample_offset += (
                    batch_size
                )

    if (
        sample_offset
        != EXPECTED_SAMPLES
    ):
        temporary_output.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Export sample count mismatch."
        )

    metrics = (
        accumulator.compute()
    )

    observed = {
        "accuracy_pct":
            f"{metrics['accuracy'] * 100:.4f}",

        "mrr_pct":
            f"{metrics['mrr@5'] * 100:.4f}",

        "ndcg_pct":
            f"{metrics['ndcg@5'] * 100:.4f}",
    }

    print()
    print(
        "=== AGGREGATE RECONSTRUCTION ==="
    )

    print(
        "Samples:",
        metrics["num_samples"],
    )

    print(
        "Accuracy:",
        observed["accuracy_pct"],
        "%",
    )

    print(
        "MRR@5:",
        observed["mrr_pct"],
        "%",
    )

    print(
        "NDCG@5:",
        observed["ndcg_pct"],
        "%",
    )

    checks = {
        "samples":
            (
                metrics["num_samples"]
                == EXPECTED_SAMPLES
            ),

        "accuracy":
            (
                observed["accuracy_pct"]
                == config["accuracy_pct"]
            ),

        "mrr":
            (
                observed["mrr_pct"]
                == config["mrr_pct"]
            ),

        "ndcg":
            (
                observed["ndcg_pct"]
                == config["ndcg_pct"]
            ),
    }

    print()
    if not HISTORICAL:
        checks = {"samples": checks["samples"]}

    print("=== VALIDATION ===")

    for name, passed in checks.items():
        print(
            f"{name}:",
            "PASS"
            if passed
            else "FAIL",
        )

    if not all(
        checks.values()
    ):
        temporary_output.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Export did not reproduce "
            "locked metrics."
        )

    os.replace(
        temporary_output,
        output_csv,
    )

    output_sha = sha256_file(
        output_csv
    )

    print()
    print(
        "Output:",
        output_csv,
    )

    print(
        "Output SHA256:",
        output_sha,
    )

    print()
    print(
        f"STATUS: "
        f"{cli.experiment.upper()} "
        "PER-SAMPLE EXPORT PASS"
    )


if __name__ == "__main__":
    main()
