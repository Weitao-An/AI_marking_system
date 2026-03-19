# -*- coding: utf-8 -*-
"""
使用 SAM 3 (Segment Anything Model 3) 对标注结果做背景分离，提取每个目标的外轮廓坐标并写回 labels JSON。
输入：output_labels/labels_*.json 中的条目 + 对应图片；
输出：同一 JSON 中为每条目标增加 contour（归一化外轮廓坐标）。
参考：https://github.com/facebookresearch/sam3
ModelScope 镜像：https://modelscope.cn/models/facebook/sam3
"""
from __future__ import annotations

import json
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple

# 若指定了 --device cpu 且当前环境是 .venv_gpu（CUDA 版 PyTorch），先尝试导入 torch；
# 若因 torch_cuda.dll 失败，直接提示先退出 .venv_gpu 再用默认 Python 运行，避免进入冗长报错。
def _early_check_cpu_in_cuda_env():
    argv = sys.argv
    i = argv.index("--device") if "--device" in argv else -1
    want_cpu = "--device=cpu" in argv or (i >= 0 and i + 1 < len(argv) and argv[i + 1] == "cpu")
    if not want_cpu:
        return
    try:
        import torch  # noqa: F401
    except OSError as e:
        err = str(e)
        if "torch_cuda" in err or "127" in err:
            venv = sys.prefix
            print("当前在 GPU 虚拟环境中，且无法加载 CUDA 依赖，无法使用 --device cpu。")
            print("请先退出当前环境，再用默认 Python 运行：")
            print("  deactivate")
            print("  python sam_segment.py --modelscope --device cpu")
            sys.exit(1)
        raise


_early_check_cpu_in_cuda_env()

import cv2
import numpy as np

from label_parser import parse_label


def _imread_unicode(path: Path):
    """读取图片，支持含中文等 Unicode 的路径（Windows 下 cv2.imread 会失败）。"""
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ModelScope 镜像模型 ID（与 modelscope.cn 一致）
MODELSCOPE_SAM3_MODEL_ID = "facebook/sam3"
MODELSCOPE_SAM3_CKPT_NAME = "sam3.pt"


def download_ckpt_from_modelscope(cache_dir: Optional[Path] = None) -> str:
    """从 ModelScope 镜像下载 SAM 3 权重，返回本地 checkpoint 文件路径。"""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        raise ImportError(
            "使用 ModelScope 镜像需安装：pip install modelscope\n"
            "镜像地址：https://modelscope.cn/models/facebook/sam3"
        )
    root = cache_dir or Path.home() / ".cache" / "modelscope" / "hub"
    root.mkdir(parents=True, exist_ok=True)
    model_dir = snapshot_download(MODELSCOPE_SAM3_MODEL_ID, cache_dir=str(root))
    model_path = Path(model_dir)
    ckpt = model_path / MODELSCOPE_SAM3_CKPT_NAME
    if not ckpt.is_file():
        for f in model_path.rglob(MODELSCOPE_SAM3_CKPT_NAME):
            return str(f)
        raise FileNotFoundError(
            f"在 {model_dir} 下未找到 {MODELSCOPE_SAM3_CKPT_NAME}，请检查 ModelScope 仓库：https://modelscope.cn/models/facebook/sam3"
        )
    return str(ckpt)


def _patch_torch_accelerator_for_cpu():
    """PyTorch 2.7+ 在纯 CPU 下会因 torch.accelerator 报错，此处做兼容：令 is_available=False、device_count=0。"""
    import torch
    if not hasattr(torch, "accelerator"):
        return
    acc = torch.accelerator
    if getattr(acc, "_sam3_cpu_patched", False):
        return
    _orig_is_available = getattr(acc, "is_available", None)
    _orig_device_count = getattr(acc, "device_count", None)

    def _is_available():
        return False

    def _device_count():
        return 0

    try:
        acc.is_available = _is_available
        acc.device_count = _device_count
        acc._sam3_cpu_patched = True
    except Exception:
        pass


def get_sam3_processor(
    device: str = "cuda",
    checkpoint_path: Optional[str] = None,
    load_from_hf: bool = True,
    use_modelscope: bool = False,
):
    """加载 SAM 3 模型并返回 Processor（用于图像 + box 提示）。"""
    # 优先使用项目内的 sam3_repo（即克隆的 sam3 仓库）
    _script_dir = Path(__file__).resolve().parent
    _sam3_repo = _script_dir / "sam3_repo"
    if _sam3_repo.is_dir() and str(_sam3_repo) not in sys.path:
        sys.path.insert(0, str(_sam3_repo))
    try:
        import torch
        # PyTorch 2.7+ 在 CPU 下会触发 "Cannot access accelerator device when none is available"
        if device == "cpu" or not torch.cuda.is_available():
            _patch_torch_accelerator_for_cpu()
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as e:
        raise ImportError(
            "请先安装 SAM 3：\n"
            "  方式一：使用项目内 sam3_repo（推荐）—— 确保项目根目录下有 sam3_repo 文件夹（即克隆的 sam3 仓库）。\n"
            "  方式二：pip install torch torchvision && pip install 'git+https://github.com/facebookresearch/sam3.git'\n"
            "并确保 Python >= 3.12、PyTorch >= 2.7。"
        ) from e
    except OSError as e:
        if "torch_cuda" in str(e) or "127" in str(e):
            raise RuntimeError(
                "当前环境的 PyTorch 为 CUDA 版，但无法加载 GPU 依赖（torch_cuda.dll）。\n"
                "可选方案：\n"
                "  1) 用 CPU 跑：退出 .venv_gpu，在默认环境下执行\n"
                "     python sam_segment.py --modelscope --device cpu\n"
                "  2) 修复 GPU 环境：安装 VC++ 运行库与（可选）CUDA Toolkit 12.4，见 INSTALL.md「torch_cuda.dll」说明。"
            ) from e
        raise
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("未检测到 CUDA，使用 CPU 运行 SAM 3。")

    # 优先从 ModelScope 镜像下载（国内更稳定）
    if use_modelscope and checkpoint_path is None:
        print("正在从 ModelScope 镜像下载 SAM 3 权重：https://modelscope.cn/models/facebook/sam3")
        checkpoint_path = download_ckpt_from_modelscope()
        load_from_hf = False

    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_hf,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    return Sam3Processor(model, device=device)


def mask_to_contour_normalized(
    mask: np.ndarray,
    height: int,
    width: int,
    simplify_epsilon: float = 0.002,
) -> List[List[float]]:
    """
    从二值 mask 提取外轮廓，并归一化到 [0,1]。
    :param mask: (H, W) 二值 mask
    :param height, width: 原图尺寸，用于归一化
    :param simplify_epsilon: 轮廓简化比例 (相对周长)，0 表示不简化
    :return: [[x_norm, y_norm], ...]，归一化坐标
    """
    if mask.dtype != np.uint8:
        mask = (mask.astype(np.float32) > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    contour = contour.astype(np.float64).reshape(-1, 2)
    if simplify_epsilon > 0 and len(contour) > 4:
        peri = cv2.arcLength(contour.astype(np.float32), True)
        contour = cv2.approxPolyDP(
            contour.astype(np.float32), simplify_epsilon * peri, True
        )
        contour = contour.reshape(-1, 2)
    h, w = mask.shape[:2]
    if (h, w) != (height, width):
        contour[:, 0] *= width / w
        contour[:, 1] *= height / h
    contour[:, 0] = np.clip(contour[:, 0] / width, 0.0, 1.0)
    contour[:, 1] = np.clip(contour[:, 1] / height, 0.0, 1.0)
    return contour.round(4).tolist()


def run_sam_on_image(
    image_path: Path,
    entries: List[Dict[str, Any]],
    processor,
    images_dir: Path,
) -> List[Dict[str, Any]]:
    """
    对一张图对应的所有 label 条目运行 SAM 3，为每个目标生成 contour 并写回 entry。
    SAM 3 使用 add_geometric_prompt(box=[cx, cy, w, h], label=True)，box 为归一化 [0,1]。
    每张图处理完会释放 GPU 张量引用，避免几十张后因显存碎片/缓存膨胀变慢。
    """
    from PIL import Image

    image = _imread_unicode(image_path)
    if image is None:
        return entries
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    pil_image = Image.fromarray(image_rgb)

    for entry in entries:
        label_text = entry.get("label") or ""
        parsed = parse_label(label_text)
        objects = parsed.get("objects") or []
        if not objects:
            entry["segmentation"] = []
            continue

        segmentation_list: List[Dict[str, Any]] = []
        inference_state = processor.set_image(pil_image)
        try:
            _run_sam_objects(
                entry, objects, processor, inference_state, height, width, segmentation_list
            )
        finally:
            # 及时释放本图 inference 状态中的 GPU 张量，减轻显存碎片与缓存膨胀
            _clear_inference_state(inference_state)

    return entries


def _clear_inference_state(inference_state: Any) -> None:
    """释放 inference_state 内可能持有的 GPU 张量引用，便于 GC 回收。"""
    try:
        if hasattr(inference_state, "clear") and callable(inference_state.clear):
            inference_state.clear()
    except Exception:
        pass


def _periodic_cuda_cleanup(image_index: int, interval: int = 20) -> None:
    """每 interval 张图清理一次 CUDA 缓存与 Python GC，缓解几十张后变慢。"""
    if image_index <= 0 or (image_index % interval) != 0:
        return
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _run_sam_objects(
    entry: Dict[str, Any],
    objects: List[Dict[str, Any]],
    processor: Any,
    inference_state: Any,
    height: int,
    width: int,
    segmentation_list: List[Dict[str, Any]],
) -> None:
    for obj in objects:
        category = obj.get("category", "").strip()
        x_min = obj.get("x_min", 0)
        y_min = obj.get("y_min", 0)
        x_max = obj.get("x_max", 0)
        y_max = obj.get("y_max", 0)
        x_min, x_max = max(0, min(x_min, x_max)), min(1, max(x_min, x_max))
        y_min, y_max = max(0, min(y_min, y_max)), min(1, max(y_min, y_max))
        if x_max <= x_min or y_max <= y_min:
            segmentation_list.append({
                "category": category,
                "bbox": [x_min, y_min, x_max, y_max],
                "contour": [],
                "error": "invalid_bbox",
            })
            continue

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min
        box_normalized = [cx, cy, w, h]

        try:
            processor.reset_all_prompts(inference_state)
            processor.add_geometric_prompt(
                box=box_normalized,
                label=True,
                state=inference_state,
            )
        except Exception as e:
            err_msg = str(e)
            if "accelerator device when none is available" in err_msg:
                err_msg += " （当前在 CPU 下运行时 PyTorch 2.7+ 的已知问题，建议使用可用 GPU 环境或见 INSTALL.md）"
            segmentation_list.append({
                "category": category,
                "bbox": [round(x_min, 4), round(y_min, 4), round(x_max, 4), round(y_max, 4)],
                "contour": [],
                "error": err_msg,
            })
            continue

        masks = inference_state.get("masks")
        if masks is None or masks.numel() == 0:
            segmentation_list.append({
                "category": category,
                "bbox": [round(x_min, 4), round(y_min, 4), round(x_max, 4), round(y_max, 4)],
                "contour": [],
                "error": "no_mask",
            })
            continue

        mask = masks[0]
        if mask.dim() == 3:
            mask = mask[0]
        mask_np = mask.cpu().numpy().astype(np.uint8)
        del mask  # 尽早释放 GPU 张量
        contour_norm = mask_to_contour_normalized(
            mask_np, height, width, simplify_epsilon=0.002
        )
        segmentation_list.append({
            "category": category,
            "bbox": [round(x_min, 4), round(y_min, 4), round(x_max, 4), round(y_max, 4)],
            "contour": contour_norm,
        })

    entry["segmentation"] = segmentation_list


def _process_chunk(
    chunk: List[Tuple[str, List[Tuple[int, Dict[str, Any]]]]],
    processor: Any,
    images_dir: Path,
) -> List[Tuple[str, List[Tuple[int, Dict[str, Any]]]]]:
    """对一批 (name, [(idx, entry), ...]) 用同一 processor 跑 SAM，原地修改 entry 并返回。"""
    for name, indices_entries in chunk:
        image_path = images_dir / name
        if not image_path.is_file():
            for _, entry in indices_entries:
                entry["segmentation"] = []
            continue
        entries = [e for _, e in indices_entries]
        run_sam_on_image(image_path, entries, processor, images_dir)
    return chunk


def _read_jsonl_lines(path: Path, indices: List[int]) -> List[Dict[str, Any]]:
    """仅读取 JSONL 中指定行号的条目，其他行不解析，节省内存。返回顺序与 indices 一致。"""
    if not indices:
        return []
    idx_set = set(indices)
    by_idx: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in idx_set:
                line = line.rstrip("\n\r")
                if line:
                    by_idx[i] = json.loads(line)
            if i > max(indices):
                break
    return [by_idx[i] for i in indices if i in by_idx]


def _write_jsonl_lines(
    path: Path, indices: List[int], new_entries: List[Dict[str, Any]]
) -> None:
    """仅更新 JSONL 中指定行的内容，其他行按原样流式拷贝，不读入内存。"""
    if not indices or len(indices) != len(new_entries):
        return
    idx_to_entry = dict(zip(indices, new_entries))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(path, "r", encoding="utf-8") as fr, open(tmp, "w", encoding="utf-8") as fw:
        for i, line in enumerate(fr):
            if i in idx_to_entry:
                fw.write(json.dumps(idx_to_entry[i], ensure_ascii=False) + "\n")
            else:
                fw.write(line)
    tmp.replace(path)


def _jsonl_to_json(path_jsonl: Path, path_json: Path) -> None:
    """将 JSONL 流式转回单个 JSON 数组，不一次性加载全部。"""
    with open(path_jsonl, "r", encoding="utf-8") as f, open(
        path_json, "w", encoding="utf-8"
    ) as out:
        out.write("[\n")
        first = True
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if not first:
                out.write(",\n")
            out.write("  " + line + "\n")
            first = False
        out.write("]\n")


def _merge_batch_results_into_full(
    jsonl_path: Path,
    batch_paths: List[Path],
    chunk_indices: List[List[int]],
    out_path: Path,
) -> None:
    """用各 batch 的结果按索引填回全量条目，写出 out_path。"""
    full: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line:
                full.append(json.loads(line))
    for batch_path, indices in zip(
        sorted(batch_paths, key=lambda x: x.name), chunk_indices
    ):
        with open(batch_path, "r", encoding="utf-8") as f:
            batch_entries = json.load(f)
        for i, entry in zip(indices, batch_entries):
            if i < len(full):
                full[i] = entry
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)


def process_labels_file(
    labels_path: Path,
    images_dir: Path,
    device: str = "cuda",
    checkpoint_path: Optional[str] = None,
    load_from_hf: bool = True,
    use_modelscope: bool = False,
    image_name: Optional[str] = None,
    skip_verified: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    num_workers: int = 1,
    batch_size: int = 0,
    merge_batches: bool = True,
    image_order: Optional[List[str]] = None,
) -> None:
    """
    读取 labels JSON，对每条条目对应的图片用 SAM 3 生成轮廓，写回同一文件。
    读写仅针对当前图片对应的行（JSONL 暂存），其他内容不读不写，节省内存、避免冲突。
    skip_verified: 为 True 时跳过 verified=True 的条目。
    progress_callback(current, total, phase): 进度回调，phase 为 'sam3'。
    num_workers: 并行线程数；batch_size>0 时强制为 1。
    batch_size: 每连续 N 张图单独写一个 labels 文件（如 20），每批结束后做彻底 GPU 清理；0 表示不分批。
    merge_batches: batch_size>0 时，是否将各批合并回一个大文件；False 则保留多个小文件不合并（人工打标加载更快）。
    """
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("标签文件为空，无需处理。")
        return

    by_name: Dict[str, list] = {}
    for i, entry in enumerate(data):
        name = (entry.get("name") or "").strip()
        if not name:
            continue
        if image_name and name != image_name:
            continue
        by_name.setdefault(name, []).append((i, entry))

    # 待处理顺序：若提供 image_order，则严格按该顺序；否则按 labels 中出现的顺序（dict 插入序）。
    to_process: List[Tuple[str, List[int]]] = []
    if image_order:
        for name in image_order:
            indices_entries = by_name.get(name)
            if not indices_entries:
                continue
            if skip_verified and all((e.get("verified") for _, e in indices_entries)):
                continue
            to_process.append((name, [x[0] for x in indices_entries]))
    else:
        for name, indices_entries in by_name.items():
            if skip_verified and all((e.get("verified") for _, e in indices_entries)):
                continue
            to_process.append((name, [x[0] for x in indices_entries]))
    total = len(to_process)
    if not to_process:
        print("无待处理条目。")
        return

    jsonl_path = labels_path.with_suffix(".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    del data

    use_batch_files = batch_size > 0
    if use_batch_files:
        num_workers = 1
        chunk_size = batch_size
        chunks: List[List[Tuple[str, List[int]]]] = [
            to_process[i : i + chunk_size]
            for i in range(0, len(to_process), chunk_size)
        ]
        stem = labels_path.stem
        parent = labels_path.parent
        batch_paths: List[Path] = []
        chunk_indices: List[List[int]] = []
        processor = get_sam3_processor(
            device=device,
            checkpoint_path=checkpoint_path,
            load_from_hf=load_from_hf,
            use_modelscope=use_modelscope,
        )
        global_idx = 0
        for batch_idx, chunk in enumerate(chunks):
            all_indices = [idx for (_, inds) in chunk for idx in inds]
            entries_chunk = _read_jsonl_lines(jsonl_path, all_indices)
            if not entries_chunk:
                continue
            pos = 0
            batch_entries: List[Dict[str, Any]] = []
            for name, indices in chunk:
                n = len(indices)
                entries_this = entries_chunk[pos : pos + n]
                pos += n
                image_path = images_dir / name
                if not image_path.is_file():
                    for e in entries_this:
                        e["segmentation"] = []
                    batch_entries.extend(entries_this)
                    global_idx += 1
                    continue
                run_sam_on_image(image_path, entries_this, processor, images_dir)
                batch_entries.extend(entries_this)
                global_idx += 1
                if progress_callback:
                    progress_callback(global_idx, total, "sam3")
                print(f"已处理: {name}")
            batch_path = parent / f"{stem}_batch{batch_idx + 1:03d}.json"
            with open(batch_path, "w", encoding="utf-8") as f:
                json.dump(batch_entries, f, ensure_ascii=False, indent=2)
            batch_paths.append(batch_path)
            chunk_indices.append(all_indices)
            print(f"已写回本批: {batch_path.name}（{len(batch_entries)} 条）")
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        if merge_batches:
            _merge_batch_results_into_full(
                jsonl_path, batch_paths, chunk_indices, labels_path
            )
            for p in batch_paths:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
            print(f"已合并写回: {labels_path}")
        else:
            print(f"已保留 {len(batch_paths)} 个小 labels 文件，未合并")
        try:
            jsonl_path.unlink(missing_ok=True)
        except OSError:
            pass
        return

    num_workers = max(1, min(num_workers, total))
    if num_workers == 1:
        processor = get_sam3_processor(
            device=device,
            checkpoint_path=checkpoint_path,
            load_from_hf=load_from_hf,
            use_modelscope=use_modelscope,
        )
        for current, (name, indices) in enumerate(to_process, 1):
            image_path = images_dir / name
            if not image_path.is_file():
                print(f"跳过（图片不存在）: {name}")
                if progress_callback:
                    progress_callback(current, total, "sam3")
                continue
            entries = _read_jsonl_lines(jsonl_path, indices)
            if not entries:
                continue
            run_sam_on_image(image_path, entries, processor, images_dir)
            _write_jsonl_lines(jsonl_path, indices, entries)
            print(f"已处理并写回: {name}")
            if progress_callback:
                progress_callback(current, total, "sam3")
            _periodic_cuda_cleanup(current, interval=20)
    else:
        import threading

        def process_one_image(
            name: str,
            indices: List[int],
            proc: Any,
            entries_copy: List[Dict[str, Any]],
        ) -> Tuple[str, List[int], List[Dict[str, Any]]]:
            image_path = images_dir / name
            if not image_path.is_file():
                for e in entries_copy:
                    e["segmentation"] = []
                return name, indices, entries_copy
            run_sam_on_image(image_path, entries_copy, proc, images_dir)
            return name, indices, entries_copy

        processors: List[Any] = []
        for _ in range(num_workers):
            processors.append(
                get_sam3_processor(
                    device=device,
                    checkpoint_path=checkpoint_path,
                    load_from_hf=load_from_hf,
                    use_modelscope=use_modelscope,
                )
            )
        write_lock = threading.Lock()
        next_idx = 0
        done = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            pending: set = set()
            for _ in range(min(num_workers, total)):
                if next_idx >= total:
                    break
                name, indices = to_process[next_idx]
                entries_copy = _read_jsonl_lines(jsonl_path, indices)
                if not entries_copy:
                    next_idx += 1
                    continue
                fut = executor.submit(
                    process_one_image,
                    name,
                    indices,
                    processors[next_idx % num_workers],
                    entries_copy,
                )
                pending.add(fut)
                next_idx += 1

            while pending:
                future = next(iter(as_completed(pending)))
                pending.discard(future)
                name, indices, updated_entries = future.result()
                with write_lock:
                    _write_jsonl_lines(jsonl_path, indices, updated_entries)
                    done += 1
                    print(f"已处理并写回: {name}")
                    if progress_callback:
                        progress_callback(done, total, "sam3")
                    _periodic_cuda_cleanup(done, interval=20)
                if next_idx < total:
                    name, indices = to_process[next_idx]
                    entries_copy = _read_jsonl_lines(jsonl_path, indices)
                    if entries_copy:
                        fut = executor.submit(
                            process_one_image,
                            name,
                            indices,
                            processors[next_idx % num_workers],
                            entries_copy,
                        )
                        pending.add(fut)
                    next_idx += 1

    _jsonl_to_json(jsonl_path, labels_path)
    try:
        jsonl_path.unlink(missing_ok=True)
    except OSError:
        pass
    print(f"已写回: {labels_path}")


def main():
    parser = argparse.ArgumentParser(
        description="用 SAM 3 对标注结果做背景分离，提取目标外轮廓并写回 labels JSON"
    )
    parser.add_argument(
        "labels_json",
        type=Path,
        nargs="?",
        default=None,
        help="labels JSON 路径，例如 output_labels/labels_20260311_142146.json",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="图片目录，默认与 labels 同目录下的 images",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="可选：本地 SAM 3 权重路径；不指定则从 Hugging Face 下载（需先 hf auth login 并申请权重访问）",
    )
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="使用本地 --checkpoint，不从 Hugging Face 下载",
    )
    parser.add_argument(
        "--modelscope",
        action="store_true",
        help="从 ModelScope 镜像下载 SAM 3 权重（国内推荐）：https://modelscope.cn/models/facebook/sam3",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="运行设备",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="仅处理该图片名（如 9e39ce28-9c3b-40f8-9843-d6e868f2e962.png）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        metavar="N",
        help="并行线程数（每线程独立加载模型，显存占用约 N 倍，默认 2）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        metavar="N",
        help="每 N 张图单独写一个 labels，每批后清显存；0 表示不分批（默认 0）",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="与 --batch-size 同用：保留多个小 labels 文件不合并，人工打标加载更快",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    default_output = project_root / "output_labels"

    if args.labels_json is None:
        candidates = sorted(
            default_output.glob("labels_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            parser.error("未找到 output_labels/labels_*.json，请指定 labels_json 路径")
        labels_path = candidates[0]
        print(f"使用最新标签文件: {labels_path}")
    else:
        labels_path = args.labels_json.resolve()

    images_dir = args.images_dir or (labels_path.parent / "images")
    if not images_dir.is_dir():
        parser.error(f"图片目录不存在: {images_dir}")

    use_modelscope = bool(args.modelscope)
    if use_modelscope:
        load_from_hf = False  # 使用 ModelScope 时不从 HF 下载
    else:
        load_from_hf = args.checkpoint is None and not args.no_hf
    if args.checkpoint is not None and not Path(args.checkpoint).is_file():
        parser.error(f"权重文件不存在: {args.checkpoint}")
    if args.checkpoint is None and args.no_hf and not use_modelscope:
        parser.error("未指定 --checkpoint 时请使用 --modelscope 或去掉 --no-hf（从镜像或 Hugging Face 下载）")

    process_labels_file(
        labels_path,
        images_dir,
        device=args.device,
        checkpoint_path=args.checkpoint,
        load_from_hf=load_from_hf,
        use_modelscope=use_modelscope,
        image_name=args.image,
        num_workers=args.workers,
        batch_size=getattr(args, "batch_size", 0),
        merge_batches=not getattr(args, "no_merge", False),
    )


if __name__ == "__main__":
    main()
