# -*- coding: utf-8 -*-
"""
主流程：从本地读取视频（滑窗提取帧）或图片，上传到 Qwen 通义千问实现自动打标。
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Callable

from config import QWEN_API_KEY, DEFAULT_WINDOW_STEP, DEFAULT_WINDOW_SIZE
from frame_extractor import collect_media
from qwen_labeler import label_image_with_qwen, DEFAULT_ANNOTATION_PROMPT


def run_pipeline(
    input_path: str,
    output_dir: str = "output_labels",
    video_step: int = DEFAULT_WINDOW_STEP,
    video_window_size: int = DEFAULT_WINDOW_SIZE,
    max_video_frames: int = 0,
    model: str = "qwen3-vl-plus",
    api_key: str = "",
    prompt: str | None = None,
    only_image_names: list | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
):
    """
    完整流程：收集媒体 -> 逐张调用 Qwen 打标 -> 保存结果。
    若传入 prompt 则作为标注模型的提示词，否则使用默认提示词。
    only_image_names: 若提供则只处理这些图片名（用于跳过已验证的）。
    progress_callback(current, total, phase): 进度回调，phase 为 'qwen'。
    返回生成的 labels 文件路径。
    """
    if not QWEN_API_KEY and not api_key:
        raise ValueError("请设置 QWEN_API_KEY 或 DASHSCOPE_API_KEY（在 .env 中）或通过参数传入 api_key")

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    print(f"正在从本地读取: {input_path}")
    if path.is_file():
        suf = path.suffix.lower()
        if suf in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"):
            print(f"  视频滑窗参数: step={video_step}, window_size={video_window_size}, max_frames={max_video_frames or '不限制'}")
        else:
            print("  单张图片模式")
    else:
        print("  目录模式：将遍历目录下所有图片")

    items = collect_media(
        input_path,
        video_step=video_step,
        video_window_size=video_window_size,
        max_video_frames=max_video_frames,
    )
    if not items:
        print("未找到任何可处理的图片或帧，退出。")
        return None
    if only_image_names is not None:
        allow = set(only_image_names)
        items = [(n, b) for n, b in items if n in allow]
    if not items:
        print("过滤后无待处理图片，退出。")
        return None

    _prompt = prompt if prompt else DEFAULT_ANNOTATION_PROMPT
    total = len(items)
    print(f"共 {total} 张待打标")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    # 图片保存目录：与标签同目录下 images 子文件夹，便于人工验证时看图对标签
    images_dir = out_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    results = []
    key = api_key or QWEN_API_KEY

    for i, (name, image_bytes) in enumerate(items, 1):
        # 先保存图片到 images 文件夹，命名与标签中的 name 一致
        image_file = images_dir / name
        with open(image_file, "wb") as f:
            f.write(image_bytes)

        print(f"  [{i}/{total}] {name} ... ", end="", flush=True)
        try:
            label_text = label_image_with_qwen(
                image_bytes=image_bytes,
                prompt=_prompt,
                model=model,
                api_key=key,
            )
            results.append({"name": name, "label": label_text})
            print("OK")
        except Exception as e:
            print(f"失败: {e}")
            results.append({"name": name, "label": "", "error": str(e)})
        if progress_callback:
            progress_callback(i, total, "qwen")

    # 保存汇总结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = out_path / f"labels_{timestamp}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n打标结果已保存到: {summary_file}")

    # 同时保存一份按名称的文本摘要，便于查看
    txt_file = out_path / f"labels_{timestamp}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"=== {r['name']} ===\n{r.get('label', r.get('error', ''))}\n\n")
    print(f"文本摘要: {txt_file}")
    print(f"图片已保存到: {images_dir}（命名与标签中的 name 一致，便于人工验证）")
    return summary_file


def main():
    parser = argparse.ArgumentParser(
        description="从本地视频/图片读取并上传 Qwen 通义千问自动打标（流程第一步）"
    )
    parser.add_argument(
        "input",
        type=str,
        help="输入路径：视频文件、单张图片或图片所在目录",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output_labels",
        help="打标结果输出目录（默认: output_labels）",
    )
    parser.add_argument(
        "--video-step",
        type=int,
        default=DEFAULT_WINDOW_STEP,
        help=f"视频滑窗步长，每隔多少帧取一帧（默认: {DEFAULT_WINDOW_STEP}）",
    )
    parser.add_argument(
        "--video-window",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"滑窗大小（默认: {DEFAULT_WINDOW_SIZE}）",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="视频最多提取帧数，0 表示不限制",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-vl-plus",
        help="Qwen 视觉模型名（默认: qwen3-vl-plus，最新 VL）",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="Qwen/DashScope API Key（不填则使用 .env 中的 QWEN_API_KEY）",
    )
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        video_step=args.video_step,
        video_window_size=args.video_window,
        max_video_frames=args.max_frames,
        model=args.model,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
