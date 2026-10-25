"""Path binding and reproduction profiles; no model/data is loaded on import."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

CANONICAL_COMMON = {"city": "xa", "seq_len": 64, "sample_rate": 1.0,
    "dynamic_mode": "controlled", "seed": 2021, "batch_size": 32,
    "learning_rate": 0.0002, "weight_decay": 0, "max_epochs": 20, "patience": 3}


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest().upper()


def add_execution_paths(parser, *, dataset_default="../dataset", output_default=None):
    parser.add_argument("--profile", choices=("historical", "fresh"), default="historical")
    parser.add_argument("--dataset-path", type=Path, default=Path(dataset_default))
    parser.add_argument("--gpt2-path", type=Path, default=Path("./models/gpt2"))
    parser.add_argument("--pretrained-checkpoint", type=Path, default=Path("./checkpoints/xa_pretrain.pth"))
    parser.add_argument("--output-dir", type=Path, default=output_default)
    parser.add_argument("--checkpoints-manifest", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./final_results"), help="Historical checkpoint tree root; hashes remain fixed")
    return parser


def bind_checkpoints(options, templates, task):
    """Fresh runs require their own locked manifest, preserving historical hashes."""
    configs = deepcopy(templates)
    if options.profile == "historical":
        if options.checkpoints_manifest is not None:
            raise ValueError("Historical profile uses the fixed canonical checkpoint manifest")
        checkpoint_root = getattr(options, "checkpoint_root", None)
        if checkpoint_root is not None:
            for config in configs.values():
                parts = Path(config["checkpoint"]).parts
                if "final_results" not in parts:
                    raise ValueError("Historical checkpoint layout is unknown")
                config["checkpoint"] = checkpoint_root.resolve().joinpath(*parts[parts.index("final_results") + 1:])
        return configs
    if options.checkpoints_manifest is None:
        raise ValueError("Fresh profile requires --checkpoints-manifest")
    path = options.checkpoints_manifest.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if set(manifest) != {"schema_version", "task", "experiments"} or manifest["schema_version"] != 1 or manifest["task"] != task:
        raise ValueError("Invalid fresh checkpoint manifest")
    if set(manifest["experiments"]) != set(templates):
        raise ValueError("Fresh manifest must identify every canonical experiment for the task")
    for name, supplied in manifest["experiments"].items():
        expected = {"checkpoint", "sha256", "semantic_mode", "selection_metric", "validation_selected"}
        if set(supplied) != expected or not re.fullmatch("[0-9A-Fa-f]{64}", supplied["sha256"]):
            raise ValueError("Fresh checkpoint needs an explicit path, digest and selection provenance")
        metric = "accuracy" if task == "next_hop" else "mae_seconds"
        if (supplied["semantic_mode"] != templates[name]["semantic_mode"] or
                supplied["selection_metric"] != metric or supplied["validation_selected"] is not True):
            raise ValueError("Fresh semantic mode/validation selection mismatch")
        checkpoint = Path(supplied["checkpoint"])
        configs[name]["checkpoint"] = (path.parent / checkpoint).resolve()
        configs[name]["checkpoint_sha256"] = supplied["sha256"].upper()
    return configs


def validate_checkpoint_metadata(checkpoint, semantic_mode, task):
    """Check metadata emitted by the actual runners; state shapes remain strict."""
    configuration = checkpoint["configuration"]
    for key, value in CANONICAL_COMMON.items():
        if configuration.get(key) != value:
            raise ValueError("Checkpoint canonical configuration mismatch: " + key)
    metric = "accuracy" if task == "next_hop" else "mae_seconds"
    if configuration.get("selection_metric") != metric or not 1 <= int(checkpoint["epoch"]) <= 20:
        raise ValueError("Checkpoint model-selection metadata mismatch")
    if metric not in checkpoint["validation_metrics"]:
        raise ValueError("Missing validation selection metric")
    if task == "tte":
        for key, value in {"task": "tte", "semantic_mode": semantic_mode, "static_ablation": "none",
                           "temporal_input": "departure_only", "prediction_clipping": False,
                           "loss": "SmoothL1(beta=1.0)", "target": "trip_duration_seconds"}.items():
            if configuration.get(key) != value:
                raise ValueError("TTE checkpoint mismatch: " + key)
        if checkpoint["target_normalization"].get("source") != "training_split_only":
            raise ValueError("TTE normalization must come from training split only")
    # NH runner did not record semantic_mode/static_ablation; do not invent it.
    # The explicit manifest attests mode; load_state_dict still checks all shapes.


def require_hash(path, expected):
    actual = sha256_file(path)
    if actual != expected.upper():
        raise ValueError("SHA256 mismatch: " + str(path))
    return actual


def prepare_output(directory, *, overwrite=False, protected=()):
    directory = Path(directory).resolve()
    for source in protected:
        source = Path(source).resolve()
        if source == directory or source.is_relative_to(directory):
            raise ValueError("Output overlaps a protected input")
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError("Output is not empty; explicit --overwrite is required")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def analysis_cli(namespace, task, argv=None):
    """Rebind path globals before executing unchanged analysis calculations."""
    project_root = Path(__file__).resolve().parents[1]
    old_root = Path(namespace["ROOT"])
    old_analysis = project_root / "final_analysis" / task if "DATA_ROOT" in namespace else old_root
    parser = argparse.ArgumentParser(description="Canonical analysis with explicit portable roots")
    parser.add_argument("--analysis-root", type=Path, default=old_analysis)
    parser.add_argument("--dataset-path", type=Path, default=project_root.parent / "dataset")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile", choices=("historical", "fresh"), default="historical")
    parser.add_argument("--overwrite", action="store_true")
    options = parser.parse_args(argv)
    old_data = namespace.get("DATA_ROOT")
    for name, value in list(namespace.items()):
        if not isinstance(value, Path) or name == "ROOT":
            continue
        if value.is_relative_to(old_analysis):
            namespace[name] = options.analysis_root.resolve() / value.relative_to(old_analysis)
        elif old_data is not None and value.is_relative_to(old_data):
            namespace[name] = options.dataset_path.resolve() / "xa" / value.relative_to(old_data)
    if old_data is None:
        namespace["ROOT"] = options.analysis_root.resolve()
    else:
        namespace["DATA_ROOT"] = options.dataset_path.resolve() / "xa"
    # NH model dictionaries contain nested paths bound at import time.
    for config in namespace.get("MODELS", {}).values():
        if isinstance(config, dict) and isinstance(config.get("path"), Path):
            config["path"] = options.analysis_root.resolve() / "per_sample" / config["path"].name
    target = "OUTPUT_DIR" if "OUTPUT_DIR" in namespace else "OUTPUT"
    if options.output_dir:
        namespace[target] = options.output_dir.resolve()
    namespace["HISTORICAL"] = options.profile == "historical"
    namespace["DATASET_PATH"] = options.dataset_path.resolve()
    protected = [p for k,p in namespace.items() if isinstance(p,Path) and k in
                 {"INPUT", "INPUT_FILE", "INPUT_PATH", "MASTER_PATH", "TEST_CACHE", "SEMANTIC_CACHE", "PER_SAMPLE_PATH"}]
    prepare_output(namespace[target], overwrite=options.overwrite, protected=protected)
    return options


def resolve_experiment_command(config_file, *, runtime_root, dataset_root, pretrained_checkpoint, gpt2_path, run_root):
    """Return an argv list only; never launch the training runner."""
    import sys
    config = json.loads(Path(config_file).read_text(encoding="utf-8"))
    roots = {"BIGCITY_ROOT": str(Path(runtime_root).resolve()), "DATASET_ROOT": str(Path(dataset_root).resolve()),
             "PRETRAINED_CHECKPOINT": str(Path(pretrained_checkpoint).resolve()), "RUN_ROOT": str(Path(run_root).resolve())}
    argv = [sys.executable, str(Path(runtime_root).resolve() / config["runner"])]
    for flag, value in config["cli"].items():
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if isinstance(value, str):
            for key, root in roots.items():
                value = value.replace("{" + key + "}", root)
            if "{" in value or "}" in value:
                raise ValueError("Unresolved canonical path placeholder")
        argv.extend([flag, str(value)])
    argv.extend(["--gpt2_path", str(Path(gpt2_path).resolve()), "--log_path",
                 str(Path(run_root).resolve() / config["id"] / "logs")])
    return argv


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resolve canonical paths and print argv; never execute training")
    parser.add_argument("--experiment", type=Path, required=True)
    for name in ("runtime-root", "dataset-root", "pretrained-checkpoint", "gpt2-path", "run-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    options = parser.parse_args(argv)
    print(json.dumps(resolve_experiment_command(options.experiment, runtime_root=options.runtime_root,
        dataset_root=options.dataset_root, pretrained_checkpoint=options.pretrained_checkpoint,
        gpt2_path=options.gpt2_path, run_root=options.run_root), indent=2))


if __name__ == "__main__":
    main()
