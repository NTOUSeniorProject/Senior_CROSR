import cv2
import numpy as np

class MovementDetector:
    def __init__(
        self,
        history=500,
        var_threshold=25,
        detect_shadows=True,
        min_motion_ratio=0.01,
        warmup_frames=60,
        confirm_frames=3,
        resize_width=320,
    ):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

        self.min_motion_ratio = min_motion_ratio
        self.warmup_frames = warmup_frames
        self.confirm_frames = confirm_frames
        self.resize_width = resize_width

        self.frame_count = 0
        self.motion_count = 0

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

    def check_whether_move(self, frame):
        height, width = frame.shape[:2]

        if width > self.resize_width:
            scale = self.resize_width / width

            frame_small = cv2.resize(
                frame,
                (
                    self.resize_width,
                    max(1, int(height * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )
        else:
            frame_small = frame

        fg_mask = self.mog2.apply(frame_small)
        self.frame_count += 1

        _, fg_mask = cv2.threshold(
            fg_mask,
            200,
            255,
            cv2.THRESH_BINARY,
        )

        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_OPEN,
            self.kernel,
            iterations=1,
        )

        fg_mask = cv2.dilate(
            fg_mask,
            self.kernel,
            iterations=2,
        )

        foreground_pixels = cv2.countNonZero(
            fg_mask
        )

        total_pixels = fg_mask.size

        motion_ratio = (
            foreground_pixels / total_pixels
            if total_pixels > 0
            else 0.0
        )

        if self.frame_count <= self.warmup_frames:
            self.motion_count = 0
            return False, motion_ratio, fg_mask

        raw_motion = (
            motion_ratio >= self.min_motion_ratio
        )

        if raw_motion:
            self.motion_count += 1
        else:
            self.motion_count = 0

        has_movement = (
            self.motion_count >= self.confirm_frames
        )

        return has_movement, motion_ratio, fg_mask
            