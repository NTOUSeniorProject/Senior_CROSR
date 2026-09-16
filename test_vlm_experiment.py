import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests
import VLM_check as vlm
import systemVsVLM as experiment


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.log = Path(self.directory.name) / "requests.jsonl"
        self.env = patch.dict(os.environ, {"VLM_EXPERIMENT_LOG": str(self.log),
                                         "VLM_EXPERIMENT_VIDEO": "sample.mp4",
                                         "VLM_EXPERIMENT_RUN_ID": "test"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def response(self, data):
        return Mock(status_code=200, json=Mock(return_value=data))

    def test_both_layers_are_counted(self):
        result = json.dumps(dict(is_abnormal=False, category="normal", confidence=0.8,
                                 description="normal", need_alert=False))
        responses = [self.response({"message": {"content": result}, "prompt_eval_count": 100,
                                    "eval_count": 20, "eval_duration": 2_000_000_000}),
                     self.response({"choices": [{"message": {"content": result}}],
                                    "usage": {"prompt_tokens": 200, "completion_tokens": 30}})]
        with patch.object(vlm, "EXPERIMENT_MODE", True), patch.object(vlm, "image_to_base64", return_value="image"), patch.object(experiment.requests, "post", side_effect=responses):
            output = vlm.analyze_frames_with_ollama(["frame.jpg"], double_vlm=1, vlm_78b_layer=2)
        self.assertTrue(output["double_vlm_triggered"])
        rows = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual([r["layer"] for r in rows], [1, 2])
        summary = experiment.summarize(self.log)[0]
        self.assertEqual(summary["total_tokens_known_sum"], 350)
        self.assertEqual(summary["thinking_seconds_missing_requests"], 2)

    def test_timeout_is_logged_and_propagated(self):
        with patch.object(experiment.requests, "post", side_effect=requests.Timeout):
            with self.assertRaises(requests.Timeout):
                experiment.request_vlm("http://test", json={"model": "test"})
        summary = experiment.summarize(self.log)[0]
        self.assertEqual(summary["failed_requests"], 1)
        self.assertIsNone(summary["total_tokens_known_sum"])

    def test_disabled_does_not_log(self):
        response = self.response({})
        with patch.object(vlm, "EXPERIMENT_MODE", False), patch.object(vlm.requests, "post", return_value=response) as post:
            self.assertIs(vlm._post_vlm("http://test", experiment_layer=1), response)
            post.assert_called_once_with("http://test")
        self.assertFalse(self.log.exists())

    def test_missing_usage_stays_unknown(self):
        with patch.object(experiment.requests, "post", return_value=self.response({"choices": []})):
            experiment.request_vlm("http://test")
        summary = experiment.summarize(self.log)[0]
        self.assertEqual(summary["total_tokens_missing_requests"], 1)
        self.assertIsNone(summary["total_tokens_known_sum"])

    def test_video_command_defaults_to_config(self):
        with patch.dict("sys.modules", {"constants": Mock(CONFIG={"video_path": "configured.mp4"})}), patch("sys.argv", ["systemVsVLM.py", "video"]), patch.object(experiment, "run_video") as run:
            experiment.main()
        self.assertEqual(run.call_args.args[0], "configured.mp4")

    def test_summary_uses_environment_log(self):
        with patch("sys.argv", ["systemVsVLM.py", "summary"]), patch.object(experiment, "summarize", return_value=[]) as summary, patch("builtins.print"):
            experiment.main()
        summary.assert_called_once_with(str(self.log))


if __name__ == "__main__":
    unittest.main()
