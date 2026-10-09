"""Meaningful checks for the candidate's evidence contract and exclusion gate."""

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("candidate_checks", ROOT / "tools/self_check.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "configs/evidence_snapshot.json").read_text(encoding="utf-8"))

    def test_dimensions_and_unknown_submission(self):
        self.assertEqual(checks.validate_snapshot(self.data), [])

    def test_dimension_drift_rejected(self):
        changed = copy.deepcopy(self.data)
        changed["observation"]["total"] = 73
        self.assertTrue(checks.validate_snapshot(changed))

    def test_action_axis_drift_rejected(self):
        changed = copy.deepcopy(self.data)
        changed["actions_dx_dz"][1] = [-1, 1]
        self.assertTrue(checks.validate_snapshot(changed))

    def test_guessed_final_model_rejected(self):
        changed = copy.deepcopy(self.data)
        changed["final_submission_checkpoint"] = "73189"
        self.assertTrue(checks.validate_snapshot(changed))

    def test_environment_or_ppo_drift_rejected(self):
        changed = copy.deepcopy(self.data)
        changed["current_environment"]["battery_max"] = 200
        changed["ppo"]["clip"] = .15
        self.assertEqual(len(checks.validate_snapshot(changed)), 2)

    def test_candidate_documents_links_and_assets(self):
        result = checks.check_candidate(ROOT)
        self.assertEqual(result["errors"], [])

    def test_restricted_artifact_gate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            # Only a temporary sentinel, never a copied checkpoint or credential.
            for suffix in (".pkl", ".pt", ".pth", ".ckpt", ".onnx", ".bin", ".safetensors"):
                (root / ("sentinel" + suffix)).touch()
            (root / ".private").mkdir()
            (root / ".private/hidden.pkl").touch()
            (root / "src").mkdir()
            errors = checks.check_candidate(root)["errors"]
            for suffix in (".pkl", ".pt", ".pth", ".ckpt", ".onnx", ".bin", ".safetensors"):
                self.assertIn("restricted artifact: sentinel" + suffix, errors)
            self.assertIn("restricted artifact: .private/hidden.pkl", errors)
            self.assertIn("restricted directory: src", errors)


if __name__ == "__main__":
    unittest.main()

