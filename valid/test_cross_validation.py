"""python -m unittest valid.test_cross_validation -v

設 RUN_CV_GPU_TEST=1 可額外執行真實 CUDA 單折單 epoch smoke test。
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from valid.ntu_cross_validation import (
    FoldDataset, calibrate, combined_scores, normalize_samples,
    scan_samples, split_records, subject_split,
)


class CrossValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_data(self):
        rng = np.random.default_rng(12)
        for subject in range(1, 41):
            for action in (1, 7):
                for repetition in (1, 2):
                    path = self.root / f"S001C001P{subject:03d}R{repetition:03d}A{action:03d}_rgb.npy"
                    np.save(path, rng.uniform(1, 100, (2, 12, 17)).astype(np.float32))
        return scan_samples(self.root)

    def test_subjects_are_disjoint_and_tested_once(self):
        tested = []
        for fold in range(1, 11):
            split = subject_split(fold)
            train, val, test = (set(split[key]) for key in ("train", "val", "test"))
            self.assertFalse(train & val or train & test or val & test)
            self.assertEqual(train | val | test, set(range(1, 41)))
            tested.extend(test)
        self.assertEqual(sorted(tested), list(range(1, 41)))

    def test_unknowns_only_in_test_and_complete_test_coverage(self):
        records = self.make_data()
        tested = []
        for fold in range(1, 11):
            splits = split_records(records, fold, [1])
            for split in ("train", "val"):
                self.assertTrue(all(r["action"] == 1 for r in splits[split]))
            tested.extend(r["path"] for r in splits["test"])
        self.assertEqual(sorted(tested), sorted(r["path"] for r in records))

    def test_duplicate_sample_rejected(self):
        self.make_data()
        np.save(self.root / "S001C001P001R001A001.npy", np.ones((2, 12, 17)))
        with self.assertRaisesRegex(ValueError, "重複"):
            scan_samples(self.root)

    def test_missing_known_class_rejected(self):
        records = self.make_data()
        with self.assertRaisesRegex(ValueError, "缺少已知"):
            split_records(records, 1, [1, 2])

    def test_padding_and_invalid_data(self):
        records = self.make_data()
        dataset = FoldDataset(records, [1], 300, 17)
        tensor, _, _ = dataset[0]
        self.assertEqual(tuple(tensor.shape), (2, 300, 17))
        self.assertEqual(torch.count_nonzero(tensor[:, 12:]).item(), 0)
        np.save(records[0]["path"], np.zeros((2, 12, 17)))
        with self.assertRaises(ValueError):
            dataset[0]

    def test_normalization_independent_of_batch(self):
        inputs = torch.rand(3, 2, 15, 17)
        alone = normalize_samples(inputs[:1], 11)
        grouped = normalize_samples(torch.cat((inputs[:1], inputs[1:] * 100)), 11)[:1]
        torch.testing.assert_close(alone, grouped)

    def test_threshold_ties_and_no_test_refitting(self):
        distances = np.linspace(0, 1, 101)
        mse = distances ** 2
        params = calibrate(distances, mse, 0.05, 0.6)
        self.assertLessEqual(params["validation_known_fpr"], 0.05)
        before = params.copy()
        ordinary = combined_scores([0.4], [0.16], params)[0]
        with_outlier = combined_scores([0.4, 1e9], [0.16, 1e9], params)[0]
        self.assertEqual(ordinary, with_outlier)
        self.assertEqual(params, before)
        tied = calibrate(np.ones(20), np.ones(20), 0.05, 0.6)
        self.assertEqual(tied["validation_known_fpr"], 0)

    @unittest.skipUnless(os.getenv("RUN_CV_GPU_TEST") == "1" and torch.cuda.is_available(), "opt-in CUDA smoke test")
    def test_gpu_training_calibration_and_evaluation(self):
        from valid.cross_validation_config import CV_CONFIG
        from valid.train_ntu_cross_validation import run_fold, summarize
        records = self.make_data()
        splits = split_records(records, 1, [1])
        splits["train"] = splits["train"][:4]
        config = dict(CV_CONFIG, known_actions=[1], num_epochs=1, batch_size=2, num_workers=0)
        result = run_fold(1, splits, config, self.root / "fold_01")
        summarize([result], self.root)
        self.assertEqual(result["test_samples"], 4)
        self.assertTrue(np.isfinite(result["metrics"]["unknown_auroc"]))
        for name in ("best_val.pth", "last.pth", "manifest.json", "calibration.json", "metrics.json", "test_predictions.npz"):
            self.assertTrue((self.root / "fold_01" / name).is_file())
        with self.assertRaises(FileExistsError):
            run_fold(1, splits, config, self.root / "fold_01")


if __name__ == "__main__":
    unittest.main()
