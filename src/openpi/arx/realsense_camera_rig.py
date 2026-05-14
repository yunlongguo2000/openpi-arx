"""RealSense camera rig for ARX inference: head + left wrist + right wrist.

Handles slow D405 camera initialization (observed in realsense-viewer too).
"""

from __future__ import annotations

import logging
import time

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

log = logging.getLogger(__name__)

CAMERA_KEYS = ("head_image", "left_wrist_image", "right_wrist_image")


class RealSenseCameraRig:
    """Connects to three RealSense D405 cameras with retry logic.

    Maps serial numbers to logical camera names based on calibration_params.yaml.
    """

    def __init__(
        self,
        head_serial: str,
        left_wrist_serial: str,
        right_wrist_serial: str,
        *,
        width: int = 424,
        height: int = 240,
        fps: int = 30,
        start_retries: int = 5,
    ):
        if rs is None:
            raise ImportError("pyrealsense2 is not installed")

        self._width = width
        self._height = height
        self._fps = fps
        self._start_retries = start_retries

        self._serial_to_key = {
            head_serial: "head_image",
            left_wrist_serial: "left_wrist_image",
            right_wrist_serial: "right_wrist_image",
        }
        self._serials = [head_serial, left_wrist_serial, right_wrist_serial]
        self._pipelines: dict[str, rs.pipeline] = {}

    def _start_one_camera(self, serial: str) -> rs.pipeline:
        key = self._serial_to_key[serial]

        for attempt in range(1, self._start_retries + 1):
            try:
                cfg = rs.config()
                cfg.enable_device(serial)
                cfg.enable_stream(rs.stream.color, self._width, self._height, rs.format.rgb8, self._fps)

                pipe = rs.pipeline()
                pipe.start(cfg)
                log.info("Camera %s (%s) pipeline started (attempt %d)", serial, key, attempt)

                # Warm up with retries
                for warmup in range(20):
                    ok, fs = pipe.try_wait_for_frames(timeout_ms=2000)
                    if ok:
                        color = fs.get_color_frame()
                        if color is not None:
                            img = np.asanyarray(color.get_data())
                            log.info("Camera %s (%s) warmed up: %dx%d", serial, key, img.shape[1], img.shape[0])
                            return pipe

                log.warning("Camera %s warmup frames not ready, retrying start...", serial)
                pipe.stop()

            except Exception as e:
                log.warning("Camera %s start attempt %d failed: %s", serial, attempt, e)

            time.sleep(1.0)

        raise RuntimeError(f"Camera {serial} ({key}) failed to start after {self._start_retries} attempts")

    def connect(self) -> None:
        # Verify all devices present
        ctx = rs.context()
        found = {
            d.get_info(rs.camera_info.serial_number): d
            for d in ctx.query_devices()
        }
        for serial in self._serials:
            if serial not in found:
                raise RuntimeError(f"Camera {serial} not found. Available: {list(found.keys())}")

        for serial in self._serials:
            pipe = self._start_one_camera(serial)
            self._pipelines[serial] = pipe

        log.info("All cameras connected and streaming")

    def read(self) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for serial, pipe in self._pipelines.items():
            success, fs = pipe.try_wait_for_frames(timeout_ms=5000)
            if not success:
                raise RuntimeError(f"Frame timeout from camera {serial}")
            color = fs.get_color_frame()
            if color is None:
                raise RuntimeError(f"No color frame from camera {serial}")
            img = np.asanyarray(color.get_data())
            key = self._serial_to_key[serial]
            frames[key] = img
        return frames

    def disconnect(self) -> None:
        for serial, pipe in self._pipelines.items():
            pipe.stop()
            log.info("Camera %s stopped", serial)
        self._pipelines.clear()
