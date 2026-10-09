"""全局中值背景减除：每一帧减去整段视频的时间中值背景，输出成新视频（不删除重复帧）。

用法（notebook 和本文件放在同一个文件夹）:
    from bg_subtract import subtract_global_median
    bg, out = subtract_global_median("../Data/gap-15mm/test-15-9psi-0_8mps.avi")
"""
from pathlib import Path

import cv2
import numpy as np


def subtract_global_median(video_path, out_path=None, diff_range=30,
                           fourcc="mp4v", chunk_rows=50):
    """用全局时间中值作背景，每一帧减去背景，写成新视频。

    参数
    ----
    video_path : 输入视频路径
    out_path   : 输出视频路径；默认存到 ../Results/background_subtracted/<视频名>_minus_bg.mp4
    diff_range : 差值的显示范围（灰度级）。输出视频中 0 → 中灰，
                 +diff_range 及以上 → 白，-diff_range 及以下 → 黑
    fourcc     : 视频编码，默认 "mp4v"（.mp4，QuickTime 可播放）
    chunk_rows : 求中值时每次处理的行数，内存紧张时调小

    返回
    ----
    bg       : 背景图（float32，和原视频同尺寸）
    out_path : 输出视频的路径
    """
    video_path = Path(video_path)
    if out_path is None:
        out_dir = Path("../Results/background_subtracted")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{video_path.stem}_minus_bg.mp4"
    out_path = Path(out_path)

    # 1. 读取全部帧（转成灰度），保留重复帧
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
    frames = np.stack(frames)
    n, h, w = frames.shape

    # 2. 全局时间中值背景（按行分块，节省内存）
    bg = np.empty((h, w), np.float32)
    for r in range(0, h, chunk_rows):
        bg[r:r + chunk_rows] = np.median(frames[:, r:r + chunk_rows], axis=0)

    # 3. 每一帧减去背景，映射到 0–255 后写入视频
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc),
                             fps, (w, h), isColor=True)
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频（编码 {fourcc}）: {out_path}")
    scale = 127.5 / diff_range
    for f in frames:
        diff = f.astype(np.float32) - bg
        out8 = np.clip(127.5 + diff * scale, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(out8, cv2.COLOR_GRAY2BGR))
    writer.release()

    print(f"{video_path.name}: {n} 帧，{w}x{h}，{fps:.2f} fps")
    print(f"差值范围 ±{diff_range} 灰度级 → 已保存到 {out_path.resolve()}")
    return bg, out_path
