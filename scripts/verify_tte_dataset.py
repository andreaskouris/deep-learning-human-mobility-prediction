from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from data_provider.tte_dataset import (
    build_tte_payload,
    sha256_file,
    validate_tte_payload,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory_file",
        required=True,
    )
    parser.add_argument(
        "--split",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--road_count",
        type=int,
        default=5269,
    )
    parser.add_argument(
        "--start_time",
        default="2018-10-01T00:00:00Z",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1800,
    )
    parser.add_argument(
        "--time_slots",
        type=int,
        default=2928,
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2021,
    )

    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    output_path = Path(args.output)
    if output_path.resolve() == Path(args.trajectory_file).resolve():
        raise ValueError("Output aliases trajectory input")
    if output_path.exists() and not args.verify_only:
        raise FileExistsError("Output exists; use --verify-only")

    start_timestamp = int(
        pd.Timestamp(
            args.start_time
        ).timestamp()
    )

    payload = build_tte_payload(
        trajectory_file=args.trajectory_file,
        split=args.split,
        seq_len=args.seq_len,
        road_count=args.road_count,
        start_timestamp=start_timestamp,
        interval_seconds=args.interval,
        time_slots_count=args.time_slots,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )

    report = validate_tte_payload(
        payload
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.verify_only:
        actual = torch.load(output_path, map_location="cpu", weights_only=True)
        def equal(left, right):
            if torch.is_tensor(left):
                return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left, right)
            if isinstance(left, dict):
                return isinstance(right, dict) and left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
            return left == right
        if not equal(payload, actual):
            raise ValueError("Reconstructed TTE cache content differs")
    else:
        with output_path.open("xb") as stream:
            torch.save(payload, stream)

    print("\n=== TTE CACHE VERIFICATION ===")

    for key, value in report.items():
        print(
            f"{key}: {value}"
        )

    print(
        "source_sha256:",
        payload["metadata"][
            "source_sha256"
        ],
    )

    print(
        "cache_sha256:",
        sha256_file(
            output_path
        ),
    )

    print(
        "output:",
        output_path,
    )

    print(
        "\nTTE dataset build and "
        "validation: PASS"
    )


if __name__ == "__main__":
    main()
