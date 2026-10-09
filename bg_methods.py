"""Background-removal methods for the galvanizing waviness project.

Methods
-------
subtractBaselineMedian  — brightness flattening + global temporal median
subtractGlobalAverage   — global frame-average background
subtractSlidingAverage  — sliding-window frame-average background

Reuses helpers from bg_subtract_rolling.py.
"""
from pathlib import Path

import cv2
import numpy as np

from bg_subtract_rolling import (
    _read_gray_frames,
    _temporal_median,
    _diff_to_uint8,
    _open_writer,
)


# ── Brightness flattening ──────────────────────────────────────────

def _flattenBrightness(frames, refBg):
    """Normalise each frame's mean intensity to match the reference background.

    gain_i = mean(frame_i) / mean(refBg)
    flattened_i = frame_i / gain_i

    NOTE: uses ALL pixels as "valid" — no mask or ROI is defined yet.
    """
    refMean = float(np.mean(refBg))
    if refMean == 0.0:
        return frames.astype(np.float32)

    flattened = np.empty(frames.shape, dtype=np.float32)
    for i in range(frames.shape[0]):
        frameMean = float(np.mean(frames[i]))
        gain = frameMean / refMean if frameMean != 0.0 else 1.0
        flattened[i] = frames[i].astype(np.float32) / gain
    return flattened


# ── Baseline: flatten + global temporal median ──────────────────────

def subtractBaselineMedian(videoPath, outPath=None, diffRange=30,
                           fourcc="mp4v", chunkRows=50):
    """Brightness-flattened global temporal median background removal.

    1. Load all frames as grayscale.
    2. Compute an initial global median background from raw frames.
    3. Flatten each frame's brightness relative to that background.
    4. Recompute the global median from the flattened frames.
    5. Subtract the recomputed background from each flattened frame.

    Returns (bg, outPath).
    """
    videoPath = Path(videoPath)
    if outPath is None:
        outDir = Path("Results")
        outDir.mkdir(parents=True, exist_ok=True)
        outPath = outDir / f"{videoPath.stem}_baseline.mp4"
    outPath = Path(outPath)

    frames, fps = _read_gray_frames(videoPath)
    n, h, w = frames.shape

    # Initial background from raw frames
    rawBg = _temporal_median(frames, chunkRows)

    # Flatten brightness using the raw background as reference
    flatFrames = _flattenBrightness(frames, rawBg)

    # Recompute background from flattened frames
    bg = _temporal_median(flatFrames, chunkRows)

    # Write output video
    writer = _open_writer(outPath, fourcc, fps, w, h)
    for i in range(n):
        writer.write(_diff_to_uint8(flatFrames[i], bg, diffRange))
    writer.release()

    print(f"{videoPath.name}: {n} frames, {w}x{h}, {fps:.2f} fps")
    print(f"Baseline (flatten + global median), diff range +/-{diffRange}")
    print(f"Saved to {outPath.resolve()}")
    return bg, outPath


# ── Global frame-average ────────────────────────────────────────────

def subtractGlobalAverage(videoPath, outPath=None, diffRange=30,
                          fourcc="mp4v"):
    """Background = pixel-wise mean of all frames.

    Returns (bg, outPath).
    """
    videoPath = Path(videoPath)
    if outPath is None:
        outDir = Path("Results")
        outDir.mkdir(parents=True, exist_ok=True)
        outPath = outDir / f"{videoPath.stem}_global_avg.mp4"
    outPath = Path(outPath)

    frames, fps = _read_gray_frames(videoPath)
    n, h, w = frames.shape

    bg = np.mean(frames, axis=0).astype(np.float32)

    writer = _open_writer(outPath, fourcc, fps, w, h)
    for i in range(n):
        writer.write(_diff_to_uint8(frames[i], bg, diffRange))
    writer.release()

    print(f"{videoPath.name}: {n} frames, {w}x{h}, {fps:.2f} fps")
    print(f"Global average background, diff range +/-{diffRange}")
    print(f"Saved to {outPath.resolve()}")
    return bg, outPath


# ── Sliding-window frame-average ────────────────────────────────────

def subtractSlidingAverage(videoPath, outPath=None, win=187, step=30,
                           diffRange=30, fourcc="mp4v"):
    """Sliding-window frame-average background removal.

    For each block of `step` frames, compute the background as the
    mean of the `win` frames centred on that block.  Follows the same
    windowing logic as subtract_rolling_median in bg_subtract_rolling.py.

    Returns outPath (no single background image — it changes over time).
    """
    videoPath = Path(videoPath)
    if outPath is None:
        outDir = Path("Results")
        outDir.mkdir(parents=True, exist_ok=True)
        outPath = outDir / f"{videoPath.stem}_sliding_avg.mp4"
    outPath = Path(outPath)

    frames, fps = _read_gray_frames(videoPath)
    n, h, w = frames.shape
    win = min(win, n)
    half = win // 2

    writer = _open_writer(outPath, fourcc, fps, w, h)
    nBg = 0
    for start in range(0, n, step):
        stop = min(start + step, n)
        center = (start + stop) // 2
        lo = int(np.clip(center - half, 0, n - win))
        bg = np.mean(frames[lo:lo + win], axis=0).astype(np.float32)
        nBg += 1
        for t in range(start, stop):
            writer.write(_diff_to_uint8(frames[t], bg, diffRange))
    writer.release()

    print(f"{videoPath.name}: {n} frames, {w}x{h}, {fps:.2f} fps")
    print(f"Sliding average: window {win}, step {step}, {nBg} backgrounds")
    print(f"Diff range +/-{diffRange} -> saved to {outPath.resolve()}")
    return outPath
