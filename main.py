# -*- coding: utf-8 -*-
"""
主流程：从本地读取视频（滑窗提取帧）或图片，上传到 Qwen 通义千问实现自动打标。
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

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
):
    """
    完整流程：收集媒体 -> 逐张调用 Qwen 打标 -> 保存结果。
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
        return

    print(f"共 {len(items)} 张待打标")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []
    key = api_key or QWEN_API_KEY

    for i, (name, image_bytes) in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {name} ... ", end="", flush=True)
        try:
            label_text = label_image_with_qwen(
                image_bytes=image_bytes,
                prompt=DEFAULT_ANNOTATION_PROMPT,
                model=model,
                api_key=key,
            )
            results.append({"name": name, "label": label_text})
            print("OK")
        except Exception as e:
            print(f"失败: {e}")
            results.append({"name": name, "label": "", "error": str(e)})

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
    return results


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
