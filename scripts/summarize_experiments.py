"""Recompute public experiment-window aggregates from numeric JSONL records."""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = {
    "W10": ("39171", 200, 4, 53), "W11": ("39171", 200, 4, 40),
    "W12": ("39171", 200, 4, 82), "W27": ("73189", 150, 3, 33),
}


def read_rows(name):
    path = ROOT / "experiments" / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize():
    episodes = read_rows("episodes.jsonl")
    learners = read_rows("learner_metrics.jsonl")
    result = []
    for window, (checkpoint, battery, chargers, expected) in WINDOWS.items():
        e = [row for row in episodes if row["window_id"] == window]
        l = [row for row in learners if row["window_id"] == window]
        assert len(e) == expected
        def avg(key, rows=e):
            values = [row[key] for row in rows if key in row]
            return {"n": len(values), "mean": round(mean(values), 6) if values else None}
        result.append({
            "id": window, "initial_checkpoint": checkpoint,
            "environment": {"battery_max": battery, "charger_count": chargers,
                            "max_step": 1000, "robot_count": 4,
                            "map": list(range(1, 11)), "map_random": True},
            "type": "continuing_training_window_not_frozen_evaluation",
            "official_evaluations_recovered": 0, "episode_count": len(e),
            "result_counts": dict(Counter(row.get("result", "UNKNOWN") for row in e)),
            "episode_metrics": {key: avg(key) for key in
                                ("total_score", "steps", "repeat_ratio", "clean_repeat_ratio",
                                 "charge_step_ratio", "rule_override_rate")},
            "learner_metrics": {key: avg(key, l) for key in ("approx_kl", "clip_frac")},
            "evidence": ["experiments/episodes.jsonl", "experiments/learner_metrics.jsonl"],
            "final_submission": "UNKNOWN", "causal_improvement": "UNVERIFIED",
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    actual = summarize()
    expected = json.loads((ROOT / "experiments/window_metrics.json").read_text(encoding="utf-8"))
    if args.check:
        comparable = [{key: row[key] for key in actual[0]} for row in expected]
        if actual != comparable:
            raise SystemExit("FAIL: public records disagree with saved window aggregates")
        print("PASS: 208 episode rows / 79 learner rows reproduce all four window aggregates")
        return 0
    if args.output:
        output = args.output.resolve()
        if ROOT not in output.parents:
            raise ValueError("output must be inside the candidate")
        args.output.write_text(json.dumps(actual, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(actual, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
