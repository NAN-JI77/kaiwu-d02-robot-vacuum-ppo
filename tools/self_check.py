"""Check candidate contents without loading the competition project."""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md", "README_EN.md", ".gitignore", "configs/evidence_snapshot.json",
    "docs/evidence.md", "docs/architecture.md", "docs/observation_action.md",
    "docs/reward.md", "docs/hybrid_decision.md", "docs/training.md", "docs/ppo.md",
    "docs/experiments.md", "docs/dependencies.md", "docs/reproducibility.md",
    "NOTICE.md", "docs/RELEASE_CHECKLIST.md", "docs/evaluation.md",
    "configs/source_manifest.json", "experiments/episodes.jsonl",
    "experiments/learner_metrics.jsonl", "experiments/window_metrics.json",
    "experiments/source_fingerprints.json", "scripts/summarize_experiments.py",
)
RESTRICTED_SUFFIXES = {
    ".pkl", ".pt", ".pth", ".pyc", ".zip", ".7z", ".mp4", ".jpg", ".png",
    ".log", ".pem", ".key", ".dat", ".onnx", ".safetensors", ".npz", ".npy",
    ".ckpt", ".bin", ".tar", ".gz", ".rar",
}
RESTRICTED_NAMES = {
    "kaiwu.json", "id_list", ".env", "license.dat",
}
RESTRICTED_DIRS = {
    "sdk", "third_party", "debug_logs", "video_analysis", ".codex_tmp", "src",
    "data", "datasets", "maps", ".codex_backups", "backups",
    ".private", "ckpt", "checkpoints", "history", "logs",
}


def validate_records(root):
    errors = []
    counts = {"W10": (53, 19), "W11": (40, 12), "W12": (82, 30), "W27": (33, 18)}
    for index, name in enumerate(("episodes", "learner_metrics")):
        seen = set()
        window_counts = {window: 0 for window in counts}
        try:
            rows = [json.loads(line) for line in (root / "experiments" / (name + ".jsonl")).read_text(encoding="utf-8").splitlines()]
            for row in rows:
                window = row["window_id"]
                alias = window + ("/episodes" if index == 0 else "/learner")
                identity = (row["source"], row["source_line"])
                if window not in counts or row["source"] != alias or identity in seen:
                    raise ValueError("invalid/duplicate source identity")
                if type(row["source_line"]) is not int or row["source_line"] <= 0:
                    raise ValueError("invalid source line")
                seen.add(identity)
                window_counts[window] += 1
                for key, value in row.items():
                    if key in ("window_id", "source"):
                        continue
                    if key == "result" and index == 0 and re.fullmatch(r"[A-Z_]+", value):
                        continue
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError("unexpected nonnumeric field")
            if window_counts != {window: pair[index] for window, pair in counts.items()}:
                raise ValueError("record counts changed")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append("experiment records invalid: " + name + ": " + type(exc).__name__)
    return errors


def validate_snapshot(data):
    errors = []
    observation = data.get("observation", {})
    local = observation.get("local_shape_hwc", [])
    global_shape = observation.get("global_shape_chw", [])
    if local != [7, 7, 12] or global_shape != [14, 16, 16]:
        errors.append("observation shape differs from evidence")
    if observation.get("base") != 73 or observation.get("scalar") != 32:
        errors.append("base/scalar dimensions differ from evidence")
    if observation.get("total") != 4277:
        errors.append("total observation length differs from evidence")
    expected_actions = [[1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1]]
    if data.get("actions_dx_dz") != expected_actions:
        errors.append("action mapping differs from evidence")
    if data.get("purpose") != "documentation_snapshot_not_runtime_configuration":
        errors.append("snapshot must identify its documentation purpose")
    if "final_submission_checkpoint" not in data or data["final_submission_checkpoint"] is not None:
        errors.append("final submission is unknown")
    if data.get("current_preload_checkpoint") != "73189" or data.get("source_code_label") != "202604282345":
        errors.append("current package identity differs from evidence")
    expected_ppo = {
        "gamma": .995, "lambda": .97, "learning_rate": 3e-5, "clip": .1,
        "value_coefficient": .7, "entropy_start": .008, "entropy_end": .0015,
        "entropy_decay_learner_calls": 150000, "gradient_clip_norm": .5,
    }
    if data.get("ppo") != expected_ppo:
        errors.append("PPO parameters differ from current evidence")
    expected_environment = {
        "map": list(range(1, 11)), "map_random": True, "robot_count": 4,
        "charger_count": 3, "max_step": 1000, "battery_max": 150,
    }
    if data.get("current_environment") != expected_environment:
        errors.append("environment differs from current evidence")
    return errors


def check_candidate(root):
    errors = []
    for name in REQUIRED:
        if not (root / name).is_file():
            errors.append("missing: " + name)
    try:
        data = json.loads((root / "configs/evidence_snapshot.json").read_text(encoding="utf-8"))
        errors.extend(validate_snapshot(data))
    except (OSError, ValueError, TypeError) as exc:
        errors.append("snapshot load failed: " + type(exc).__name__)
    manifest = []
    try:
        manifest = json.loads((root / "configs/source_manifest.json").read_text(encoding="utf-8"))["files"]
        if len(manifest) != 22 or len({entry["path"] for entry in manifest}) != 22:
            errors.append("source/config manifest must contain 22 unique files")
        if sum(entry["path"].endswith(".py") for entry in manifest) != 17:
            errors.append("source manifest must contain 17 Python files")
        for entry in manifest:
            path = root / entry["path"]
            if root.resolve() not in path.resolve().parents:
                errors.append("source manifest path escapes candidate")
            elif not path.is_file():
                errors.append("missing source/config: " + entry["path"])
            elif (hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
                  or path.stat().st_size != entry["bytes"]):
                errors.append("source hash/size mismatch: " + entry["path"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append("source manifest invalid: " + type(exc).__name__)
    errors.extend(validate_records(root))
    count = 0
    python_count = 0
    copyright_count = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if {".git", ".venv"}.intersection(relative.parts):
            continue
        if path.is_symlink():
            errors.append("symlink is not allowed: " + relative.as_posix())
        if any(part in RESTRICTED_DIRS for part in relative.parts):
            errors.append("restricted directory: " + relative.as_posix())
        if not path.is_file():
            continue
        count += 1
        if path.suffix.lower() in RESTRICTED_SUFFIXES or path.name in RESTRICTED_NAMES or path.name.startswith(".env."):
            errors.append("restricted artifact: " + relative.as_posix())
        if path.suffix == ".py":
            python_count += 1
            try:
                compile(path.read_text(encoding="utf-8"), relative.as_posix(), "exec")
            except (SyntaxError, UnicodeError) as exc:
                errors.append("compile failed: " + relative.as_posix() + ": " + type(exc).__name__)
        if path.suffix == ".md":
            content = path.read_text(encoding="utf-8")
            targets = re.findall(r"\[[^\]\r\n]+\]\(([^)\r\n]+)\)", content)
            for target in targets:
                if target.startswith(("https://", "http://", "#")):
                    continue
                try:
                    destination = (path.parent / target.split("#", 1)[0]).resolve()
                except OSError:
                    errors.append("invalid local link in " + relative.as_posix())
                    continue
                if root.resolve() not in destination.parents or not destination.exists():
                    errors.append("broken/external local link in " + relative.as_posix())
            if re.search(r"(?:MIGRATION_MANIFEST|P[0-3]_(?:TEST|COPYRIGHT|DEPENDENCY|MIGRATION)|infer_tensor\.py|summarize_logs\.py)", content):
                errors.append("internal/removed artifact reference in " + relative.as_posix())
        if re.match(r"P[0-3]_", path.name) or path.name in ("MIGRATION_MANIFEST.md", "OPEN_SOURCE_STATUS.md", "TEST_RESULTS.md"):
            errors.append("internal report in " + relative.as_posix())
        if path.suffix in (".py", ".toml", ".md", ".json", ".jsonl", ".txt"):
            content = path.read_text(encoding="utf-8-sig")
            if re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", content):
                errors.append("machine path in " + relative.as_posix())
            if re.search(r"\b(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)\b", content):
                errors.append("internal IP in " + relative.as_posix())
            if path.suffix == ".py" and re.search(r"^# Copyright .+Tencent\. All Rights Reserved\.$", content, re.MULTILINE):
                copyright_count += 1
            if re.search(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----", content):
                errors.append("private key in " + relative.as_posix())
            if re.search(r"(?i)(?:password|api[_-]?key|access[_-]?token)\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{16,}", content):
                errors.append("credential-shaped value in " + relative.as_posix())
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError):
                errors.append("invalid JSON in " + relative.as_posix())
    if copyright_count != 8:
        errors.append("expected eight preserved Tencent copyright headers")
    return {"status": "FAILED" if errors else "PASS", "public_files": count,
            "python_files_compiled": python_count, "source_config_files": len(manifest),
            "tencent_copyright_headers": copyright_count, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = check_candidate(args.root.resolve())
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

