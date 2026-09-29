import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import real_time_detector as detector


class EventSubmissionTests(unittest.TestCase):
    def setUp(self):
        output = patch("builtins.print")
        output.start()
        self.addCleanup(output.stop)
        self.pending = detector._PendingVLMEvents()

    def submit(self, executor, partial=False):
        return detector._finish_and_submit_event(
            "event-1", ["frame"], [True], 30,
            executor, self.pending, "user", force_partial=partial,
        )

    def test_normal_and_partial_use_same_queue_and_complete(self):
        for partial in (False, True):
            with self.subTest(partial=partial), patch.object(detector, "finish_event_collection", return_value={"frame_paths": ["frame.jpg"]}) as finish, patch.object(detector, "_analyze_event_with_vlm") as analyze:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = self.submit(executor, partial)
                self.assertIsNotNone(future)
                self.assertEqual(self.pending.total(), 0)
                analyze.assert_called_once_with(["frame.jpg"], "user")
                self.assertEqual(finish.call_args.kwargs["force_partial"], partial)

    def test_discarded_event_is_not_submitted(self):
        executor = Mock()
        with patch.object(detector, "finish_event_collection", return_value=None):
            self.assertIsNone(self.submit(executor))
        executor.submit.assert_not_called()
        self.assertEqual(self.pending.total(), 0)

    def test_save_failure_is_not_submitted(self):
        executor = Mock()
        with patch.object(detector, "finish_event_collection", side_effect=OSError("disk")):
            self.assertIsNone(self.submit(executor))
        executor.submit.assert_not_called()
        self.assertEqual(self.pending.total(), 0)

    def test_submission_failure_rolls_back_count(self):
        executor = Mock()
        executor.submit.side_effect = RuntimeError("closed")
        with patch.object(detector, "finish_event_collection", return_value={"frame_paths": ["frame.jpg"]}):
            self.assertIsNone(self.submit(executor))
        self.assertEqual(self.pending.total(), 0)

    def test_failed_vlm_releases_count_and_next_event_runs(self):
        with patch.object(detector, "finish_event_collection", return_value={"frame_paths": ["frame.jpg"]}), patch.object(detector, "_analyze_event_with_vlm", side_effect=[RuntimeError("VLM"), None]) as analyze:
            with ThreadPoolExecutor(max_workers=1) as executor:
                first = self.submit(executor)
                second = self.submit(executor, True)
            self.assertIsInstance(first.exception(), RuntimeError)
            self.assertIsNone(second.exception())
            self.assertEqual(analyze.call_count, 2)
            self.assertEqual(self.pending.total(), 0)


if __name__ == "__main__":
    unittest.main()
