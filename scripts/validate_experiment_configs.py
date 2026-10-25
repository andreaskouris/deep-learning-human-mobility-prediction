"""Validate six experiment specifications without importing or running model code.

Only Python's standard library is used. The real argparse declarations are read
with AST, then rebuilt as data; the original module and runners are never
imported, compiled, evaluated or executed. Printed commands are argument lists.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr
import importlib.metadata
import io
import json
from pathlib import Path
import re
import sys

EXPECTED = {
    "nh_baseline": ("next_hop", "none"),
    "nh_poi_tfidf": ("next_hop", "poi_tfidf"),
    "nh_qwen_poi": ("next_hop", "poi_qwen_tfidf"),
    "tte_baseline": ("tte", "none"),
    "tte_structured": ("tte", "tte_structured"),
    "tte_qwen": ("tte", "tte_qwen"),
}
LOCKED = {
    "--city": "xa", "--split": "train", "--seq_len": 64, "--sample_rate": 1,
    "--seed": 2021, "--batch_size": 32, "--learning_rate": 0.0002,
    "--weight": 0, "--train_epochs": 20, "--patience": 3,
    "--dynamic_mode": "controlled", "--static_ablation": "none",
}
PLACEHOLDERS = {"BIGCITY_ROOT", "DATASET_ROOT", "PRETRAINED_CHECKPOINT", "RUN_ROOT"}
TOKEN = re.compile(r"\{([A-Z_]+)\}")
SHA256 = re.compile(r"[0-9A-F]{64}")
STATUS = {"CONFIRMED", "INFERRED"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    """Reject duplicate JSON keys rather than silently selecting the last one."""
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs)


def parser_from_source(path):
    """Rebuild literal add_argument declarations; execute no source statements."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"), filename=str(path))
    functions = [n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == "get_args_parser"]
    require(len(functions) == 1, "Expected one get_args_parser function")
    parser = argparse.ArgumentParser(add_help=False)
    types = {"str": str, "int": int, "float": float}
    for statement in functions[0].body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        if not (isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "parser" and call.func.attr == "add_argument"):
            continue
        flags = [ast.literal_eval(n) for n in call.args]
        options = {}
        for keyword in call.keywords:
            require(keyword.arg is not None, "Dynamic parser kwargs are unsupported")
            if keyword.arg == "help":
                continue
            if keyword.arg == "type":
                require(isinstance(keyword.value, ast.Name)
                        and keyword.value.id in types, "Nonliteral argument type")
                options["type"] = types[keyword.value.id]
            else:
                options[keyword.arg] = ast.literal_eval(keyword.value)
        parser.add_argument(*flags, **options)
    require(bool(parser._actions), "No literal argparse declarations found")
    return parser


def argv_from_cli(cli, bindings):
    """Substitute path tokens and preserve spaces as single argv entries."""
    require(set(bindings) <= PLACEHOLDERS, "Unknown path binding")
    require(all(isinstance(v, str) and v for v in bindings.values()), "Empty binding")
    argv = []
    for flag, value in cli.items():
        require(isinstance(flag, str) and flag.startswith("--"), "Invalid CLI flag")
        require(type(value) in (str, int, float, bool), f"Nonscalar CLI value: {flag}")
        if type(value) is bool:
            if value:
                argv.append(flag)
            continue
        text = str(value)
        for token in TOKEN.findall(text):
            require(token in PLACEHOLDERS and token in bindings,
                    f"Missing or unknown placeholder: {token}")
            text = text.replace("{" + token + "}", bindings[token])
        require("{" not in text and "}" not in text, "Unresolved placeholder")
        argv.extend((flag, text))
    return argv


def parse_cli(parser, cli):
    
    bindings = {name: f"/EXTERNAL/{name}" for name in PLACEHOLDERS}
    known = parser._option_string_actions
    for flag, value in cli.items():
        require(flag in known, f"Unknown runner argument: {flag}")
        action = known[flag]
        if isinstance(action, argparse._StoreTrueAction):
            require(type(value) is bool, f"{flag} must be Boolean")
        else:
            require(type(value) is not bool, f"{flag} requires a scalar argument")
    with redirect_stderr(io.StringIO()) as captured:
        try:
            return parser.parse_args(argv_from_cli(cli, bindings))
        except SystemExit as exc:
            raise ValueError(captured.getvalue().strip()) from exc


def artifact_index(manifest):
    """Validate external-only identities and a closed, acyclic build graph."""
    records = manifest["artifacts"]
    index = {item["id"]: item for item in records}
    require(len(records) == len(index), "Duplicate artifact id")
    for item in records:
        require(item["distribution"] == "EXTERNAL_ONLY", "Artifact is not external-only")
        require(item["kind"] != "finetuned_checkpoint", "Fine-tuned checkpoint dependency")
        digest = item.get("canonical_sha256")
        if digest is not None:
            require(bool(SHA256.fullmatch(digest)), f"Invalid SHA256: {item['id']}")
        for dependency in item["depends_on"]:
            require(dependency in index, f"Unknown artifact dependency: {dependency}")
    visiting, complete = set(), set()
    def visit(key):
        require(key not in visiting, f"Cyclic artifact dependency: {key}")
        if key in complete:
            return
        visiting.add(key)
        for dependency in index[key]["depends_on"]:
            visit(dependency)
        visiting.remove(key)
        complete.add(key)
    for key in index:
        visit(key)
    policy = manifest["historical_finetuned_policy"]
    require(policy["required_for_new_training"] is False
            and policy["hashes_are_acceptance_criteria"] is False,
            "Historical fine-tuned hashes cannot gate new training")
    require(bool(re.fullmatch(r"[0-9a-f]{40}",
                              manifest["source_revisions"]["bigcity"]["commit"])),
            "BIGCity source must be pinned")
    return index


def dependency_closure(keys, index):
    result = set()
    def visit(key):
        require(key in index, f"Unknown artifact: {key}")
        if key in result:
            return
        result.add(key)
        for dependency in index[key]["depends_on"]:
            visit(dependency)
    for key in keys:
        visit(key)
    return sorted(result)


def validate_config(config, parser, index):
    """Enforce experiment identity, scientific settings and evidence coverage."""
    key = config["id"]
    require(key in EXPECTED, "Noncanonical experiment")
    task, mode = EXPECTED[key]
    require((config["task"], config["semantic_mode"]) == (task, mode),
            "Task/semantic-mode mismatch")
    require(config["runner"] == f"train_{task}.py", "Wrong runner")
    require(config["environment"] == "bigcity", "Training environment must be BIGCity")
    cli = config["cli"]
    require(cli.get("--task_name") == task and cli.get("--semantic_mode") == mode,
            "CLI task/semantic mismatch")
    require(set(cli) <= set(LOCKED) | {
        "--task_name", "--semantic_mode", "--dataset_path",
        "--pretrained_checkpoint", "--checkpoint_path", "--log_path",
        "--use_gpu", "--device", "--wandb_mode"}, "Unreviewed CLI setting")
    for flag, value in LOCKED.items():
        require(cli.get(flag) == value and type(cli.get(flag)) is type(value),
                f"Canonical value changed: {flag}")
    require(cli.get("--use_gpu") is True and cli.get("--device") == "0",
            "Configuration specifies reviewed single-GPU protocol")
    require(cli.get("--wandb_mode") == "offline", "W&B must be offline")
    require(cli.get("--dataset_path") == "{DATASET_ROOT}", "Incorrect dataset root")
    require(cli.get("--pretrained_checkpoint") == "{PRETRAINED_CHECKPOINT}",
            "Use upstream pretrained checkpoint identity")
    require(cli.get("--checkpoint_path") == f"{{RUN_ROOT}}/{key}/checkpoint",
            "New checkpoint output must be isolated")
    if task == "tte":
        require("--log_path" not in cli, "TTE runner ignores --log_path")
    else:
        require(cli.get("--log_path") == f"{{RUN_ROOT}}/{key}/logs", "NH log destination")
    parsed = parse_cli(parser, cli)
    provenance = config["provenance"]["cli"]
    require(set(provenance) == set(cli), "Missing CLI provenance")
    for flag, record in provenance.items():
        require(record["status"] in STATUS and bool(record.get("evidence")),
                f"Invalid evidence record: {flag}")
    historical = config["historical"]["checkpoint_metadata"]["configuration"]
    mapping = {"max_epochs": "--train_epochs", "weight_decay": "--weight",
               "task": "--task_name"}
    for field, value in historical.items():
        flag = mapping.get(field, "--" + field)
        if flag in cli:
            require(cli[flag] == value, f"Saved checkpoint mismatch: {field}")
            if flag in LOCKED or field in {"semantic_mode", "task"}:
                require(provenance[flag]["status"] == "CONFIRMED",
                        f"Recorded checkpoint setting incorrectly marked INFERRED: {flag}")
    for setting in config["effective_settings"].values():
        require(setting["status"] in STATUS and bool(setting.get("evidence")),
                "Missing effective-setting provenance")
    runtime = config["required_runtime_artifacts"]
    prefix = "nh" if task == "next_hop" else "tte"
    require({f"{prefix}_{split}_cache" for split in ("train", "validation", "test")}
            <= set(runtime), "Missing official split cache")
    semantic_id = {"poi_tfidf": "poi_tfidf_cache", "poi_qwen_tfidf": "qwen_poi_road_cache",
                   "tte_structured": "tte_structured_cache", "tte_qwen": "tte_qwen_cache"}
    require(set(runtime) >= {"bigcity_pretrained", "full_metadata", "relation_cache",
                            "static_cache", "dynamic_controlled_cache",
                            "gpt2_config", "gpt2_weights", "gpt2_merges",
                            "gpt2_vocab", "gpt2_tokenizer_config"}, "Missing runtime asset")
    semantics = set(semantic_id.values()) & set(runtime)
    require(semantics == ({semantic_id[mode]} if mode != "none" else set()),
            "Wrong semantic cache dependency")
    closure = dependency_closure(runtime, index)
    return {"id": key, "runner": config["runner"], "cli_arguments": len(cli),
            "confirmed_cli_fields": sum(r["status"] == "CONFIRMED"
                                        for r in provenance.values()),
            "inferred_cli_fields": sum(r["status"] == "INFERRED"
                                       for r in provenance.values()),
            "semantic_mode": parsed.semantic_mode, "runtime_artifacts": len(runtime),
            "reproduction_dependency_closure": closure, "valid": True}


def validate_runner_source(path, task):
    """Check optimizer, selection and batching contracts by AST, never execution."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    constants = {n.targets[0].id: ast.literal_eval(n.value)
                 for n in tree.body if isinstance(n, ast.Assign)
                 and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                 and isinstance(n.value, ast.Constant)}
    require(constants.get("TASK_NAME") == task, "Runner TASK_NAME changed")
    metric = "accuracy" if task == "next_hop" else "mae_seconds"
    require(constants.get("PRIMARY_METRIC") == metric, "Selection metric changed")
    func_name = "validation_improved" if task == "next_hop" else "tte_validation_improved"
    selection = next(n for n in tree.body
                     if isinstance(n, ast.FunctionDef) and n.name == func_name)
    comparisons = [n for n in ast.walk(selection) if isinstance(n, ast.Compare)]
    expected_op = ast.Gt if task == "next_hop" else ast.Lt
    require(len(comparisons) == 1 and len(comparisons[0].ops) == 1
            and isinstance(comparisons[0].ops[0], expected_op),
            "Best-checkpoint selection must be strict")
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    adam = [n for n in calls if ast.unparse(n.func) == "torch.optim.Adam"]
    require(len(adam) == 1, "Expected Adam optimizer")
    options = {n.arg: ast.unparse(n.value) for n in adam[0].keywords}
    require(options.get("lr") == "args.learning_rate"
            and options.get("weight_decay") == "args.weight", "Optimizer CLI wiring changed")
    loaders = [n for n in calls if ast.unparse(n.func) == "DataLoader"]
    require(bool(loaders) and all(any(k.arg == "num_workers"
                    and isinstance(k.value, ast.Constant) and k.value.value == 0
                    for k in n.keywords) for n in loaders), "DataLoader worker contract changed")
    return {"runner": str(Path(path).name), "task": task, "selection_metric": metric,
            "optimizer": "Adam", "weight_decay_argument": "--weight", "num_workers": 0}


def read_pins(path):
    pins = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "--extra-index-url https://download.pytorch.org/whl/cu126":
            continue
        require(bool(re.fullmatch(r"[a-z0-9-]+==[0-9A-Za-z.+-]+", line)),
                f"Unpinned or unsupported dependency: {line}")
        package, version = line.split("==")
        require(package not in pins, f"Duplicate dependency: {package}")
        pins[package] = version
    return pins


def validate_environments(root, check_installed=None):
    records = read_json(root / "environments/provenance.json")["environments"]
    report = {}
    for name, record in records.items():
        pins = read_pins(root / f"environments/{name}.txt")
        require(pins == record["installed_dependency_closure"], "Dependency snapshot mismatch")
        require(not record["missing_dependencies"]
                and not record["unsatisfied_installed_requirements"],
                "Recorded dependency metadata is inconsistent")
        differences = []
        if check_installed == name:
            for package, expected in pins.items():
                try:
                    actual = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    actual = None
                if actual != expected:
                    differences.append({"package": package, "expected": expected, "actual": actual})
            require(not differences, f"Installed package differences: {differences}")
        report[name] = {"pinned_packages": len(pins),
                        "installed_versions_checked": check_installed == name,
                        "clean_install_tested": False}
    require(records["bigcity"]["installed_dependency_closure"]["transformers"] == "4.47.1"
            and records["qwen"]["installed_dependency_closure"]["transformers"] == "4.51.3",
            "Keep distinct observed Transformers environments")
    return report


def validate_all(root, check_installed=None):
    root = Path(root)
    manifest = read_json(root / "reproducibility/external_artifacts.json")
    index = artifact_index(manifest)
    parser = parser_from_source(root / "config/args_config.py")
    paths = sorted((root / "experiments").glob("*.json"))
    configs = [read_json(p) for p in paths]
    require(len(configs) == 6 and {c["id"] for c in configs} == set(EXPECTED),
            "Expected exactly the six canonical configurations")
    return {"schema_version": 1, "validation": "STATIC_ONLY",
            "experiments": [validate_config(c, parser, index) for c in configs],
            "runners": [validate_runner_source(root / f"train_{task}.py", task)
                        for task in ("next_hop", "tte")],
            "environments": validate_environments(root, check_installed),
            "external_artifacts": len(index),
            "training_or_generation_executed": False,
            "unresolved_blockers": manifest["blockers"]}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    cli.add_argument("--check-environment", choices=("bigcity", "qwen", "analysis"))
    cli.add_argument("--print-command", choices=tuple(EXPECTED))
    cli.add_argument("--bind", action="append", default=[], metavar="KEY=PATH")
    options = cli.parse_args(argv)
    try:
        report = validate_all(options.root, options.check_environment)
        if options.print_command:
            bindings = {}
            for binding in options.bind:
                require("=" in binding, "Binding must be KEY=PATH")
                key, value = binding.split("=", 1)
                require(key not in bindings, "Duplicate binding")
                bindings[key] = value
            require(set(bindings) == PLACEHOLDERS, "Provide all four path bindings")
            config = read_json(options.root / f"experiments/{options.print_command}.json")
            report["printed_command_only"] = {
                "working_directory": bindings["BIGCITY_ROOT"],
                "argv": ["python", "-B", str(Path(bindings["BIGCITY_ROOT"]) / config["runner"]),
                         *argv_from_cli(config["cli"], bindings)],
                "note": "Activate the BIGCity environment before any separately authorized execution. This tool never launches the printed command.",
            }
        require(not options.bind or options.print_command, "--bind requires --print-command")
        print(json.dumps(report, indent=2))
        return 0
    except (ValueError, KeyError, OSError, StopIteration, SyntaxError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
