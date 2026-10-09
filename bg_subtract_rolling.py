"""滑动中值背景减除：每一帧减去它附近一段时间的中值背景，输出成新视频（不删除重复帧）。

用法（notebook 和本文件放在同一个文件夹）:
    from bg_subtract_rolling import subtract_rolling_median
    out = subtract_rolling_median("../Data/gap-15mm/test-15-9psi-0_8mps.avi", win=187)
"""
from pathlib import Path

import cv2
import numpy as np

DEFAULT_OUT_DIR = Path("../Results/background_subtracted")


# ---------- 共用的小工具 ----------

def _read_gray_frames(video_path):
    """读入全部帧并转成灰度（uint8），保留重复帧。返回 (frames, fps)。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.stack(frames), fps


def _temporal_median(stack, chunk_rows=50):
    """逐像素求时间中值，按行分块以节省内存。"""
    out = np.empty(stack.shape[1:], np.float32)
    for r in range(0, stack.shape[1], chunk_rows):
        out[r:r + chunk_rows] = np.median(stack[:, r:r + chunk_rows], axis=0)
    return out


def _open_writer(out_path, fourcc, fps, w, h):
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc),
                             fps, (w, h), isColor=True)
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频（编码 {fourcc}）: {out_path}")
    return writer


def _diff_to_uint8(frame, bg, diff_range):
    """差值映射到 0–255：0 → 中灰，+diff_range → 白，-diff_range → 黑。"""
    diff = frame.astype(np.float32) - bg
    out8 = np.clip(127.5 + diff * (127.5 / diff_range), 0, 255).astype(np.uint8)
    return cv2.cvtColor(out8, cv2.COLOR_GRAY2BGR)


def _default_out(video_path, suffix):
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUT_DIR / f"{Path(video_path).stem}_{suffix}.mp4"


# ---------- 滑动中值背景 ----------

def subtract_rolling_median(video_path, out_path=None, win=187, step=30,
                            diff_range=30, fourcc="mp4v", chunk_rows=50):
    """每一帧用它附近 win 帧的时间中值作背景，减去背景后写成新视频。

    为了省时间，背景不是每帧都重新算：把视频按 step 帧分成小段，
    每段用以段中心为中心、长度为 win 的窗口算一张背景，段内所有帧共用。
    靠近开头和结尾时，窗口整体向内平移，保持始终是 win 帧。

    参数
    ----
    video_path : 输入视频路径
    out_path   : 输出路径；默认 ../Results/background_subtracted/<视频名>_minus_rollbg.mp4
    win        : 窗口长度（原始帧数，含重复帧）。
                 每个真实帧重复 6 次时，187 帧 ≈ 31 个真实帧
    step       : 每隔多少帧重新算一次背景；1 = 逐帧计算（很慢）
    diff_range : 差值的显示范围（灰度级）：0 → 中灰，±diff_range → 白 / 黑
    fourcc     : 视频编码，默认 "mp4v"
    chunk_rows : 求中值时每次处理的行数，内存紧张时调小

    返回 out_path：输出视频路径
    """
    out_path = Path(out_path) if out_path else _default_out(video_path, "minus_rollbg")
    frames, fps = _read_gray_frames(video_path)
    n, h, w = frames.shape
    win = min(win, n)
    half = win // 2

    writer = _open_writer(out_path, fourcc, fps, w, h)
    n_bg = 0
    for start in range(0, n, step):
        stop = min(start + step, n)
        center = (start + stop) // 2
        lo = int(np.clip(center - half, 0, n - win))     # 窗口保持 win 帧
        bg = _temporal_median(frames[lo:lo + win], chunk_rows)
        n_bg += 1
        for t in range(start, stop):
            writer.write(_diff_to_uint8(frames[t], bg, diff_range))
    writer.release()

    print(f"{Path(video_path).name}: {n} 帧，{w}x{h}，{fps:.2f} fps")
    print(f"滑动中值背景：窗口 {win} 帧，每 {step} 帧更新一次，共算了 {n_bg} 张背景")
    print(f"差值范围 ±{diff_range} 灰度级 → 已保存到 {out_path.resolve()}")
    return out_path
