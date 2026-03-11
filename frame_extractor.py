# -*- coding: utf-8 -*-
"""
从本地读取视频（滑窗提取帧）或图片，供后续上传 Gemini 打标。
"""
from pathlib import Path
from typing import Iterator, List, Tuple, Union
import cv2
import numpy as np


def extract_frames_from_video(
    video_path: Union[str, Path],
    step: int = 30,
    window_size: int = 1,
    max_frames: int = 0,
) -> Iterator[Tuple[int, bytes]]:
    """
    从视频中按滑窗方式提取帧，返回 (帧序号, 帧的 JPEG 字节)。

    :param video_path: 视频文件路径
    :param step: 每隔多少帧取一帧（步长）
    :param window_size: 每个窗口取几帧（通常为 1）
    :param max_frames: 最多提取帧数，0 表示不限制
    :yield: (frame_index, jpeg_bytes)
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    count = 0
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 按步长采样
            if frame_index % step < window_size:
                _, buf = cv2.imencode(".jpg", frame)
                yield frame_index, buf.tobytes()
                count += 1
                if max_frames and count >= max_frames:
                    break
            frame_index += 1
    finally:
        cap.release()


def load_image_as_bytes(image_path: Union[str, Path]) -> bytes:
    """
    读取单张图片并返回 JPEG 字节（若是其他格式会先解码再编码为 JPEG）。
    使用 Python 原生 open 读取，避免 Windows 下 cv2.imread 对中文/Unicode 路径的支持问题。

    :param image_path: 图片路径
    :return: 图片的 JPEG 字节
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    with open(image_path, "rb") as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"无法解码图片: {image_path}")

    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def collect_media(
    input_path: Union[str, Path],
    video_step: int = 30,
    video_window_size: int = 1,
    max_video_frames: int = 0,
) -> List[Tuple[str, bytes]]:
    """
    根据路径类型收集所有要打标的图片字节。
    - 若是视频：按滑窗提取帧，返回 [(f"frame_{idx}", bytes), ...]
    - 若是单张图：返回 [(filename, bytes)]
    - 若是目录：遍历目录下图片文件，返回 [(filename, bytes), ...]

    :return: [(名称, 图片字节), ...]
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"路径不存在: {input_path}")

    results: List[Tuple[str, bytes]] = []
    suffix_image = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if input_path.is_file():
        suf = input_path.suffix.lower()
        if suf in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"):
            for idx, jpeg_bytes in extract_frames_from_video(
                input_path,
                step=video_step,
                window_size=video_window_size,
                max_frames=max_video_frames,
            ):
                results.append((f"frame_{idx:06d}.jpg", jpeg_bytes))
        elif suf in suffix_image:
            results.append((input_path.name, load_image_as_bytes(input_path)))
        else:
            raise ValueError(f"不支持的文件类型: {input_path}")
        return results

    if input_path.is_dir():
        for p in sorted(input_path.iterdir()):
            if p.suffix.lower() in suffix_image:
                results.append((p.name, load_image_as_bytes(p)))
        return results

    raise ValueError(f"无效路径: {input_path}")
