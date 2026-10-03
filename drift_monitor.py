import cv2
import numpy as np
import time
import json

LOG_PATH = "drift_log.jsonl"


class DriftMonitor:
    def __init__(self, baseline_brightness, baseline_contrast,
                 brightness_threshold=35, contrast_threshold=20):
        self.baseline_brightness = baseline_brightness
        self.baseline_contrast = baseline_contrast
        self.brightness_threshold = brightness_threshold
        self.contrast_threshold = contrast_threshold

    def check(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        alert = None
        if self.baseline_brightness - brightness > self.brightness_threshold:
            alert = "low_light"       # похоже на ночь
        elif self.baseline_contrast - contrast > self.contrast_threshold:
            alert = "low_contrast"    # похоже на дождь/туман

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