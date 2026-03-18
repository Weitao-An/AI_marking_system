# -*- coding: utf-8 -*-
"""
使用阿里云百炼批量推理 API 对多张图片进行 Qwen VL 打标，成本约为实时推理的 50%。
参考：https://help.aliyun.com/zh/model-studio/batch-inference
"""
import base64
import json
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from openai import OpenAI

from config import QWEN_API_KEY
from qwen_labeler import DASHSCOPE_BASE_URL, DEFAULT_ANNOTATION_PROMPT

# 批量推理单行 JSON 上限 6MB，预留给 prompt 等，图片 base64 后建议单条 < 5MB
MAX_JSONL_LINE_BYTES = 5 * 1024 * 1024
POLL_INTERVAL_SEC = 10
BATCH_ENDPOINT = "/v1/chat/completions"


def create_batch_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    key = api_key or QWEN_API_KEY
    if not key:
        raise ValueError("未设置 QWEN_API_KEY，请在 .env 中配置")
    return OpenAI(api_key=key, base_url=base_url or DASHSCOPE_BASE_URL)


def build_jsonl(
    items: list[tuple[str, bytes]],
    prompt: str = DEFAULT_ANNOTATION_PROMPT,
    model: str = "qwen3-vl-plus",
) -> tuple[Path, list[int]]:
    """
    生成批量推理用 JSONL 文件，每行一个请求。
    :param items: [(name, image_bytes), ...]
    :return: (jsonl_path, indices_of_oversized) 超出单行体积的索引，需后续用实时接口补跑
    """
    suf_to_mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    oversized: list[int] = []
    lines: list[str] = []

    for i, (name, image_bytes) in enumerate(items):
        suf = Path(name).suffix.lower()
        mime = suf_to_mime.get(suf, "image/jpeg")
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"

        body = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        }
        line = json.dumps(
            {"custom_id": str(i), "method": "POST", "url": BATCH_ENDPOINT, "body": body},
            ensure_ascii=False,
        )
        if len(line.encode("utf-8")) > MAX_JSONL_LINE_BYTES:
            oversized.append(i)
            continue
        lines.append(line)

    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", encoding="utf-8", delete=False)
    for ln in lines:
        f.write(ln + "\n")
    f.close()
    return Path(f.name), oversized


def _parse_batch_result_lines(lines: list[str]) -> dict[str, str]:
    """
    解析批量推理结果（每行一条 JSON，与阿里云 success.jsonl 格式一致）。
    :return: { custom_id (str) -> label_text }
    """
    result_by_id: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        custom_id = row.get("custom_id")
        if custom_id is None:
            continue
        resp = row.get("response") or {}
        if resp.get("status_code") != 200:
            result_by_id[str(custom_id)] = ""
            continue
        body = resp.get("body") or {}
        choices = body.get("choices") or []
        if not choices:
            result_by_id[str(custom_id)] = ""
            continue
        msg = choices[0].get("message") or {}
        result_by_id[str(custom_id)] = (msg.get("content") or "").strip()
    return result_by_id


def parse_batch_result_file(path: Path) -> dict[str, str]:
    """
    从本地 success.jsonl 文件解析批量推理结果（阿里云下载的格式）。
    :return: { custom_id (index 字符串 "0","1",...) -> label_text }
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    return _parse_batch_result_lines(text.strip().splitlines())


def merge_batch_result_to_labels(
    success_jsonl_path: Path,
    image_names: list[str],
    output_path: Path,
    merge_into_existing: bool = True,
) -> int:
    """
    将 success.jsonl 中的结果按 image_names 顺序合并进 labels 文件。
    custom_id "0" 对应 image_names[0]，"1" 对应 image_names[1]，以此类推。
    :param success_jsonl_path: 阿里云下载的 success.jsonl 路径
    :param image_names: 与批量提交时顺序一致的图片文件名列表
    :param output_path: 输出的 labels JSON 路径（若存在且 merge_into_existing 则在此基础上更新）
    :param merge_into_existing: 若 output_path 已存在，是否合并进现有条目（否则覆盖）
    :return: 写入的条目数量
    """
    result_by_id = parse_batch_result_file(success_jsonl_path)
    n_results = len(result_by_id)
    if n_results == 0:
        raise ValueError(f"未能从 {success_jsonl_path} 解析出任何结果，请检查文件格式")

    if output_path.exists() and merge_into_existing:
        raw = output_path.read_text(encoding="utf-8", errors="replace")
        try:
            entries = json.loads(raw)
        except json.JSONDecodeError:
            entries = []
        if not isinstance(entries, list):
            entries = []
    else:
        entries = []

    by_name = {e.get("name"): e for e in entries if e.get("name")}
    for i, name in enumerate(image_names):
        label = result_by_id.get(str(i), "")
        if name in by_name:
            by_name[name]["label"] = label
        else:
            entry = {"name": name, "label": label, "verified": False}
            by_name[name] = entry
            entries.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    return len(entries)


def run_batch_and_collect(
    jsonl_path: Path,
    api_key: Optional[str] = None,
    completion_window: str = "24h",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict[str, str]:
    """
    上传 JSONL、创建批量任务、轮询直至完成、下载结果并解析。
    :return: { custom_id (index str) -> label_text }
    """
    def log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    client = create_batch_client(api_key=api_key)
    log("正在上传 JSONL 文件…")
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    input_file_id = file_obj.id
    try:
        jsonl_path.unlink(missing_ok=True)
    except OSError:
        pass

    log("正在创建批量推理任务…")
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint=BATCH_ENDPOINT,
        completion_window=completion_window,
    )
    batch_id = batch.id

    while True:
        batch = client.batches.retrieve(batch_id=batch_id)
        status = batch.status
        log(f"批量任务状态: {status}")
        if status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(POLL_INTERVAL_SEC)

    if status != "completed":
        err = getattr(batch, "errors", None) or status
        raise RuntimeError(f"批量任务未成功: {err}")

    output_file_id = batch.output_file_id
    if not output_file_id:
        return {}

    log("正在下载结果文件…")
    content = client.files.content(output_file_id)
    text = getattr(content, "text", None) or (content.read().decode("utf-8", errors="replace") if hasattr(content, "read") else str(content))
    return _parse_batch_result_lines(text.strip().splitlines())


def label_images_with_batch(
    items: list[tuple[str, bytes]],
    prompt: str = DEFAULT_ANNOTATION_PROMPT,
    model: str = "qwen3-vl-plus",
    api_key: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> list[dict]:
    """
    使用批量推理 + 必要时实时补跑，返回与 items 同序的 [{"name", "label", "error"?}]。
    """
    if not items:
        return []

    total = len(items)
    # 先尝试批量
    jsonl_path, oversized_indices = build_jsonl(items, prompt=prompt, model=model)
    batch_indices = [i for i in range(total) if i not in oversized_indices]

    results: list[dict] = [{"name": name, "label": ""} for name, _ in items]
    batch_ok = False

    def batch_progress(msg: str) -> None:
        if progress_callback:
            progress_callback(0, total, msg)

    try:
        if batch_indices:
            result_by_id = run_batch_and_collect(
                jsonl_path,
                api_key=api_key,
                progress_callback=batch_progress,
            )
            for i in batch_indices:
                label = result_by_id.get(str(i), "")
                results[i]["label"] = label
            if progress_callback:
                progress_callback(len(batch_indices), total, "qwen_batch")
            batch_ok = True
    except Exception as e:
        for i in batch_indices:
            results[i]["label"] = ""
            results[i]["error"] = str(e)

    # 超出单行体积的用实时接口补跑；批量失败时全部改实时
    if batch_ok:
        to_realtime = list(oversized_indices)
    else:
        to_realtime = list(range(total))
    if to_realtime:
        from qwen_labeler import label_image_with_qwen
        key = api_key or QWEN_API_KEY
        for idx, i in enumerate(to_realtime):
            name, image_bytes = items[i]
            if progress_callback:
                progress_callback(i + 1, total, "qwen")
            try:
                suf = Path(name).suffix.lower()
                mime = "image/png" if suf == ".png" else "image/jpeg"
                label = label_image_with_qwen(
                    image_bytes=image_bytes,
                    prompt=prompt,
                    model=model,
                    api_key=key,
                    mime_type=mime,
                )
                results[i]["label"] = label
                if "error" in results[i]:
                    del results[i]["error"]
            except Exception as e:
                results[i]["label"] = ""
                results[i]["error"] = str(e)

    return results


def _main_merge() -> None:
    """CLI：将下载的 success.jsonl 合并进 labels，需提供与批量提交时同序的图片名列表。"""
    import argparse
    parser = argparse.ArgumentParser(
        description="将阿里云批量推理下载的 success.jsonl 按图片名顺序合并进 labels 文件"
    )
    parser.add_argument(
        "success_jsonl",
        type=Path,
        help="下载的 success.jsonl 文件路径（如 output_labels/xxx_success.jsonl）",
    )
    parser.add_argument(
        "-n", "--names-file",
        type=Path,
        default=None,
        help="文本文件：每行一个图片文件名，顺序与批量提交时一致",
    )
    parser.add_argument(
        "--names-from-labels",
        type=Path,
        default=None,
        help="从已有 labels JSON 中取前 N 个 name 作为顺序（N=success.jsonl 行数），慎用",
    )
    parser.add_argument(
        "--names-from-dir",
        type=Path,
        default=None,
        help="从目录中按文件名排序取前 N 张图名作为顺序（N=success.jsonl 行数），慎用",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="输出 labels 文件路径（默认：与 success_jsonl 同目录下 labels_batch_merge.json）",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="不合并进已有文件，只根据图片名列表生成新 labels",
    )
    args = parser.parse_args()

    path = args.success_jsonl.resolve()
    if not path.is_file():
        raise SystemExit(f"文件不存在: {path}")

    result_by_id = parse_batch_result_file(path)
    n = len(result_by_id)
    print(f"已解析 {n} 条批量结果（custom_id 0..{n-1}）")

    image_names: list[str] = []
    if args.names_file:
        image_names = [
            line.strip() for line in args.names_file.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        print(f"从 --names-file 读取 {len(image_names)} 个图片名")
    elif args.names_from_labels:
        labels_path = args.names_from_labels.resolve()
        if not labels_path.is_file():
            raise SystemExit(f"文件不存在: {labels_path}")
        raw = labels_path.read_text(encoding="utf-8", errors="replace")
        entries = json.loads(raw)
        if not isinstance(entries, list):
            raise SystemExit("--names-from-labels 需为 JSON 数组")
        image_names = [e.get("name") for e in entries[:n] if e.get("name")]
        print(f"从 --names-from-labels 取前 {n} 个 name，得到 {len(image_names)} 个")
    elif args.names_from_dir:
        d = args.names_from_dir.resolve()
        if not d.is_dir():
            raise SystemExit(f"目录不存在: {d}")
        allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        all_names = sorted(f.name for f in d.iterdir() if f.suffix.lower() in allowed)
        image_names = all_names[:n]
        print(f"从 --names-from-dir 按排序取前 {n} 个，得到 {len(image_names)} 个（请确认与批量提交顺序一致）")
    else:
        raise SystemExit(
            "必须指定图片名顺序之一: --names-file <每行一个文件名> | --names-from-labels <已有 labels.json> | --names-from-dir <images 目录>"
        )

    if len(image_names) < n:
        print(f"警告: 图片名数量 {len(image_names)} < 结果数 {n}，不足的 custom_id 将无对应 name")

    out = args.output or path.parent / "labels_batch_merge.json"
    out = out.resolve()
    merge = not args.no_merge
    written = merge_batch_result_to_labels(path, image_names, out, merge_into_existing=merge)
    print(f"已写入 {written} 条到: {out}")


if __name__ == "__main__":
    _main_merge()
