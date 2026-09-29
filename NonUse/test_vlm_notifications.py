import unittest
from unittest.mock import patch

import constants
import inference.line_notifier as notifier
class VLMNotificationTests(unittest.TestCase):
    def setUp(self):
        output = patch("builtins.print")
        output.start()
        self.addCleanup(output.stop)

    def result(self):
        return dict(is_abnormal=True, need_alert=True, confidence=0.75,
                    description="人物跌倒後未起身")

    def test_alert_conditions_and_threshold_boundary(self):
        cases = [
            ({}, True),
            ({"confidence": 0.749}, False),
            ({"confidence": 0.9}, True),
            ({"is_abnormal": False}, False),
            ({"need_alert": False}, False),
        ]
        with patch.object(constants, "VLM_ALERT_CONFIDENCE_THRESHOLD", 0.75):
            for changes, expected in cases:
                result = self.result()
                result.update(changes)
                with self.subTest(changes=changes), patch.object(notifier, "push_line_message") as push:
                    notifier.notify_vlm_result(result, "test-user")
                    self.assertEqual(push.call_count, int(expected))
                    if expected:
                        push.assert_called_once_with(
                            "test-user", "🚨 VLM 確認異常事件\n描述：人物跌倒後未起身\n")

    def test_threshold_comes_from_constants(self):
        with patch.object(constants, "VLM_ALERT_CONFIDENCE_THRESHOLD", 0.8):
            self.assertFalse(notifier.should_alert_vlm(self.result()))

    def test_missing_user_previews_without_network(self):
        with patch.object(notifier.requests, "post") as post, patch("builtins.print") as output:
            notifier.notify_vlm_result(self.result(), None)
        post.assert_not_called()
        self.assertTrue(any("人物跌倒後未起身" in str(call) for call in output.call_args_list))

    def test_missing_token_does_not_send(self):
        with patch.object(notifier, "LINE_CHANNEL_ACCESS_TOKEN", None), patch.object(notifier.requests, "post") as post:
            notifier.notify_vlm_result(self.result(), "test-user")
        post.assert_not_called()

    def test_vlm_failure_does_not_notify(self):
        import inference.real_time_detector as detector
        with patch.object(detector, "analyze_frames_with_ollama", side_effect=RuntimeError("unavailable")), patch.object(detector, "notify_vlm_result") as notify:
            with self.assertRaises(RuntimeError):
                detector._analyze_event_with_vlm(["frame.jpg"], "test-user")
        notify.assert_not_called()

    def test_event_analysis_delegates_notification_once(self):
        import inference.real_time_detector as detector
        result = self.result()
        with patch.object(detector, "analyze_frames_with_ollama", return_value=result), patch.object(detector, "notify_vlm_result") as notify:
            detector._analyze_event_with_vlm(["frame.jpg"], "test-user")
        notify.assert_called_once_with(result, "test-user")


if __name__ == "__main__":
    unittest.main()
