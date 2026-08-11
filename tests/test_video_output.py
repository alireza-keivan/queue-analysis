"""Tests for the annotated-output sizing/codec logic.

Pure logic + OpenCV container handling - no model, no tracker, no GPU, so this
runs anywhere in a couple of seconds.

    python -m pytest tests/test_video_output.py -v
"""
import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.queue_management import annotated_frame_size, frame_stride, video_writer


class FakeCapture:
    """Stands in for cv2.VideoCapture - only the properties these functions read."""

    def __init__(self, width, height, fps):
        self._props = {
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_FPS: fps,
        }

    def get(self, prop):
        return self._props[prop]


# --- annotated_frame_size -------------------------------------------------

def test_downscales_1080p_to_max_width_preserving_aspect():
    w, h = annotated_frame_size(FakeCapture(1920, 1080, 60), max_width=960)
    assert (w, h) == (960, 540)


def test_never_upscales_a_small_source():
    # 640 wide is already under the cap - must be left exactly as-is.
    assert annotated_frame_size(FakeCapture(640, 480, 30), max_width=960) == (640, 480)


def test_dimensions_are_always_even():
    # Odd results are rejected outright by some encoders.
    for src_w, src_h in [(1919, 1079), (1280, 719), (1001, 667)]:
        w, h = annotated_frame_size(FakeCapture(src_w, src_h, 30), max_width=960)
        assert w % 2 == 0 and h % 2 == 0, f"{src_w}x{src_h} -> {w}x{h}"


def test_max_width_none_disables_downscaling():
    assert annotated_frame_size(FakeCapture(1920, 1080, 60), max_width=None) == (1920, 1080)


def test_aspect_ratio_is_preserved_within_a_pixel():
    src_w, src_h = 1920, 1080
    w, h = annotated_frame_size(FakeCapture(src_w, src_h, 60), max_width=960)
    assert abs((w / h) - (src_w / src_h)) < 0.01


# --- frame_stride ---------------------------------------------------------

def test_stride_matches_source_over_target_fps():
    stride, effective = frame_stride(FakeCapture(1920, 1080, 59.94), 10)
    assert stride == 6
    assert effective == pytest.approx(59.94 / 6, rel=1e-6)


def test_stride_never_below_one_when_target_exceeds_source():
    # Asking for more fps than the source has must not produce stride 0.
    stride, effective = frame_stride(FakeCapture(640, 480, 25), 60)
    assert stride == 1
    assert effective == pytest.approx(25.0)


def test_effective_fps_is_what_the_writer_should_use():
    # The writer's playback rate must equal the rate frames are processed at,
    # or the output plays at the wrong speed.
    stride, effective = frame_stride(FakeCapture(1920, 1080, 30), 10)
    assert effective == pytest.approx(30 / stride)


# --- video_writer ---------------------------------------------------------

def test_video_writer_opens_and_accepts_downscaled_frames():
    """The real failure this guards against: VideoWriter silently discards
    frames whose size differs from the size it was opened with, producing an
    empty file with no error raised."""
    cap = FakeCapture(1920, 1080, 59.94)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.mp4")
        writer = video_writer(cap, path, target_fps=10)
        assert writer is not None and writer.isOpened()

        size = annotated_frame_size(cap)
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        for _ in range(5):
            writer.write(frame)
        writer.release()

        assert os.path.getsize(path) > 0, "writer produced an empty file"


def test_downscaled_output_is_smaller_than_full_resolution():
    """The whole point of the downscale: fewer pixels to encode and upload."""
    cap = FakeCapture(1920, 1080, 59.94)
    rng = np.random.default_rng(0)
    sizes = {}
    with tempfile.TemporaryDirectory() as tmp:
        for label, max_width in (("small", 960), ("full", None)):
            path = os.path.join(tmp, f"{label}.mp4")
            w, h = annotated_frame_size(cap, max_width=max_width)
            writer = video_writer(cap, path, target_fps=10, max_width=max_width)
            assert writer is not None
            # Noise, not a flat colour - a constant frame compresses to almost
            # nothing and would make the comparison meaningless.
            for _ in range(15):
                writer.write(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
            writer.release()
            sizes[label] = os.path.getsize(path)

    assert sizes["small"] < sizes["full"], sizes
