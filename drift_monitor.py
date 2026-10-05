import cv2
import numpy as np
import time
import json

LOG_PATH = "drift_log.jsonl"


class DriftMonitor:
    """Detects sudden brightness/contrast shifts in incoming frames (e.g. night or rain).

    If baseline_brightness/baseline_contrast are not provided, the monitor auto-calibrates
    from the first `calibration_frames` frames it sees. This makes it robust to different
    camera angles and setups: each stream establishes its own "normal" conditions instead
    of being compared against one fixed value taken from a different, unrelated camera.
    """

    def __init__(self, baseline_brightness=None, baseline_contrast=None,
                 brightness_threshold=35, contrast_threshold=20,
                 calibration_frames=30):
        self.baseline_brightness = baseline_brightness
        self.baseline_contrast = baseline_contrast
        self.brightness_threshold = brightness_threshold
        self.contrast_threshold = contrast_threshold
        self.calibration_frames = calibration_frames
        self._calib_brightness = []
        self._calib_contrast = []

    def _is_calibrated(self):
        return self.baseline_brightness is not None and self.baseline_contrast is not None

    def check(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        if not self._is_calibrated():
            # Still learning this stream's own "normal" brightness/contrast
            self._calib_brightness.append(brightness)
            self._calib_contrast.append(contrast)
            if len(self._calib_brightness) >= self.calibration_frames:
                self.baseline_brightness = float(np.mean(self._calib_brightness))
                self.baseline_contrast = float(np.mean(self._calib_contrast))
            return {
                "timestamp": time.time(),
                "brightness": round(brightness, 1),
                "contrast": round(contrast, 1),
                "alert": None,
                "calibrating": True,
            }

        alert = None
        if self.baseline_brightness - brightness > self.brightness_threshold:
            alert = "low_light"       # looks like nighttime
        elif self.baseline_contrast - contrast > self.contrast_threshold:
            alert = "low_contrast"    # looks like rain/fog

        result = {
            "timestamp": time.time(),
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "alert": alert,
        }

        if alert:
            self._log(result)

        return result

    def _log(self, result):
        print(f"⚠️  DRIFT ALERT: {result['alert']} "
              f"(brightness={result['brightness']}, contrast={result['contrast']})")
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(result) + "\n")