# -*- coding: utf-8 -*-
"""
人工验证标注：Web UI，展示 images 内图片与对应标签，支持编辑并保存。
并提供素材导入页：上传图片、设置提示词、一键生成标注并跑 SAM3。
"""
import json
import os
import re
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from flask import Flask, send_from_directory, request, jsonify, Response
from label_parser import parse_label, format_label

app = Flask(__name__, static_folder=None)

# 默认使用项目下的 output_labels；可通过环境变量 REVIEW_OUTPUT_DIR 覆盖
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output_labels"


def get_output_dir() -> Path:
    p = os.environ.get("REVIEW_OUTPUT_DIR", "")
    if p:
        return Path(p)
    return DEFAULT_OUTPUT_DIR


def get_images_dir() -> Path:
    return get_output_dir() / "images"


def get_all_labels_paths():
    """返回所有 labels_*.json；当存在 _batch 小文件时，优先用 batch 内容（先列 batch 再列主文件）。"""
    out = get_output_dir()
    files = list(out.glob("labels_*.json"))
    # batch 文件在前，合并时其条目优先生效；主文件在后用于补全未入 batch 的条目（如已验证跳过）
    return sorted(files, key=lambda f: (("_batch" not in f.name), f.name))


def get_latest_labels_path():
    files = get_all_labels_paths()
    return files[0] if files else None


def get_merged_entries():
    """合并所有 labels_*.json：同名以最新文件为准。返回 (by_name, name_to_path)。"""
    by_name = {}
    name_to_path = {}
    for path in get_all_labels_paths():
        if not path.is_file():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        for e in data:
            if "verified" not in e:
                e["verified"] = False
            name = (e.get("name") or "").strip()
            if not name:
                continue
            if name not in by_name:
                by_name[name] = e
                name_to_path[name] = path
    return by_name, name_to_path


def load_labels(path=None) -> tuple[list, Path]:
    """返回 (entries, path)。path 为空时用最新文件（单文件加载，供 save 用）。"""
    if path is None:
        path = get_latest_labels_path()
    if not path or not path.is_file():
        return [], path
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for e in data:
        if "verified" not in e:
            e["verified"] = False
    return data, path


def save_labels(entries: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    # 同步更新同时间戳的 .txt
    txt_path = path.with_suffix(".txt")
    if txt_path.exists():
        with open(txt_path, "w", encoding="utf-8") as f:
            for r in entries:
                f.write(f"=== {r.get('name', '')} ===\n{r.get('label', '')}\n\n")


def _safe_filename(name: str) -> str:
    """只保留安全字符，避免路径遍历。"""
    name = os.path.basename(name)
    return re.sub(r"[^\w.\-]", "_", name) or "image"


@app.route("/")
def index():
    return send_from_directory(Path(__file__).parent, "review_index.html")


@app.route("/import")
def import_page():
    return send_from_directory(Path(__file__).parent, "import_index.html")


@app.route("/augment")
def augment_page():
    return send_from_directory(Path(__file__).parent, "augment_index.html")


@app.route("/api/config")
def api_config():
    out = get_output_dir()
    images_dir = get_images_dir()
    names = []
    if images_dir.is_dir():
        for f in sorted(images_dir.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                names.append(f.name)
    all_paths = get_all_labels_paths()
    label_files = [p.name for p in all_paths if p.is_file()]
    return jsonify({
        "outputDir": str(out),
        "labelFiles": label_files,
        "imageNames": names,
    })


@app.route("/api/entry/<path:name>")
def api_entry(name):
    by_name, name_to_path = get_merged_entries()
    name_stripped = name.strip()
    entry = by_name.get(name_stripped)
    label_text = entry.get("label", "") if entry else ""
    parsed = parse_label(label_text)
    source_path = name_to_path.get(name_stripped)
    result = {
        "name": name_stripped,
        "label": label_text,
        "verified": entry.get("verified", False) if entry else False,
        "parsed": parsed,
        "sourceFile": source_path.name if source_path else None,
    }
    if entry and "segmentation" in entry:
        result["segmentation"] = entry["segmentation"]
    else:
        result["segmentation"] = []
    return jsonify(result)


@app.route("/api/image/<path:name>")
def api_image(name):
    images_dir = get_images_dir()
    return send_from_directory(images_dir, name)


@app.route("/api/default-prompt", methods=["GET"])
def api_default_prompt():
    from qwen_labeler import DEFAULT_ANNOTATION_PROMPT
    return jsonify({"prompt": DEFAULT_ANNOTATION_PROMPT})


# ---------- 边缘情形增强：仅针对已验证图片 ----------
@app.route("/api/augment/edge-images", methods=["GET"])
def api_augment_edge_images():
    """返回某张原图及其已存在的 edge 变体文件名列表（原图在前，其次 _edge、_edge1、_edge2…）。"""
    import re
    name = (request.args.get("name") or "").strip()
    if not name:
        return jsonify({"images": []})
    images_dir = get_images_dir()
    if not images_dir.is_dir():
        return jsonify({"images": []})
    stem = Path(name).stem
    ext = Path(name).suffix or ".jpg"
    base = re.sub(r"_edge\d*$", "", stem)
    ordered = [base + ext]
    for i in range(0, 500):
        candidate = f"{base}_edge{i}{ext}" if i > 0 else f"{base}_edge{ext}"
        if candidate not in ordered:
            ordered.append(candidate)
    existing = [n for n in ordered if (images_dir / n).is_file()]
    return jsonify({"images": existing})


@app.route("/api/augment/verified-images")
def api_augment_verified_images():
    """返回已通过人工验证的图片列表及简要信息，供边缘情形页使用。"""
    by_name, _ = get_merged_entries()
    images_dir = get_images_dir()
    if not images_dir.is_dir():
        return jsonify({"images": []})
    names = sorted(
        f.name for f in images_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )
    verified = []
    for name in names:
        entry = by_name.get(name)
        if not entry or not entry.get("verified"):
            continue
        label_text = entry.get("label", "")
        parsed = parse_label(label_text)
        verified.append({
            "name": name,
            "scene": parsed.get("scene", ""),
            "objects": parsed.get("objects", []),
            "categories": parsed.get("categories", ""),
            "label": label_text,
        })
    return jsonify({"images": verified})


@app.route("/api/augment/prompts", methods=["POST"])
def api_augment_prompts():
    """
    根据图片与标注生成边缘情形提示词，或与 Qwen 多轮讨论。
    Body: { "imageName": str, "chatHistory": [{role, content}], "userMessage": str }.
    若 chatHistory 与 userMessage 存在则为多轮对话；否则为首次生成。
    """
    data = request.get_json() or {}
    name = (data.get("imageName") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "缺少 imageName"}), 400
    images_dir = get_images_dir()
    path = images_dir / name
    if not path.is_file():
        return jsonify({"ok": False, "error": "图片不存在"}), 404
    by_name, _ = get_merged_entries()
    entry = by_name.get(name)
    if not entry or not entry.get("verified"):
        return jsonify({"ok": False, "error": "该图片未通过人工验证"}), 400
    label_text = entry.get("label", "")
    parsed = parse_label(label_text)
    scene = parsed.get("scene", "")
    objects = parsed.get("objects", [])
    categories = parsed.get("categories", "")

    with open(path, "rb") as f:
        image_bytes = f.read()
    suf = path.suffix.lower()
    mime = "image/png" if suf == ".png" else "image/jpeg"

    try:
        from edge_augment import generate_edge_prompts, chat_edge_prompts
        chat_history = data.get("chatHistory") or []
        user_message = (data.get("userMessage") or "").strip()
        if chat_history and user_message:
            text = chat_edge_prompts(
                image_bytes=image_bytes,
                scene=scene,
                objects=objects,
                categories=categories,
                user_message=user_message,
                chat_history=chat_history,
                mime_type=mime,
            )
        else:
            text = generate_edge_prompts(
                image_bytes=image_bytes,
                scene=scene,
                objects=objects,
                categories=categories,
                user_extra=user_message or None,
                chat_history=chat_history if chat_history else None,
                mime_type=mime,
            )
        return jsonify({"ok": True, "text": text})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/augment/generate", methods=["POST"])
def api_augment_generate():
    """
    用 qwen-image-edit 根据编辑提示词生成新图，保存到 images 并写入 labels。
    会根据编辑提示词用 Qwen 适配场景描述，物体列表等沿用原图。
    Body: { "imageName": str, "editPrompt": str }.
    """
    data = request.get_json() or {}
    name = (data.get("imageName") or "").strip()
    edit_prompt = (data.get("editPrompt") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "缺少 imageName"}), 400
    if not edit_prompt:
        return jsonify({"ok": False, "error": "缺少 editPrompt"}), 400
    images_dir = get_images_dir()
    path = images_dir / name
    if not path.is_file():
        return jsonify({"ok": False, "error": "图片不存在"}), 404
    by_name, name_to_path = get_merged_entries()
    entry = by_name.get(name)
    if not entry or not entry.get("verified"):
        return jsonify({"ok": False, "error": "该图片未通过人工验证"}), 400
    label_text = entry.get("label", "")

    try:
        from edge_augment import image_edit, download_image_url, suggest_new_filename, adapt_label_for_edge
        urls = image_edit(path, edit_prompt, keep_position=True)
        if not urls:
            return jsonify({"ok": False, "error": "未返回生成图片"}), 500
        existing = [f.name for f in images_dir.iterdir() if f.is_file()]
        new_name = suggest_new_filename(name, existing_names=existing)
        save_path = images_dir / new_name
        download_image_url(urls[0], save_path)

        # 根据编辑提示词适配场景描述，生成边缘情形对应的 label（物体列表等保持不变）
        adapted_label = adapt_label_for_edge(label_text, edit_prompt)

        # 写入 labels：与主流程一致，使用最新 labels 文件并追加
        labels_path = get_latest_labels_path()
        if not labels_path or not labels_path.is_file():
            labels_path = get_output_dir() / "labels_manual.json"
        entries, _ = load_labels(labels_path)
        # 新条目：使用适配后的 label，视为未验证（可再去验证页勾选）
        new_entry = {"name": new_name, "label": adapted_label, "verified": False}
        if "segmentation" in entry:
            new_entry["segmentation"] = entry["segmentation"]
        entries.append(new_entry)
        save_labels(entries, labels_path)
        return jsonify({
            "ok": True,
            "newName": new_name,
            "labelFile": labels_path.name,
            "message": f"已生成并保存为 {new_name}，label 已按边缘情形适配场景描述",
        })
    except Exception as e:
        import traceback
        print("ERROR: /api/augment/generate failed")
        print(traceback.format_exc())
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/augment/delete-edge-image", methods=["POST"])
def api_augment_delete_edge_image():
    """
    删除边缘情形生成的图片文件，并从所有 labels 中移除对应条目。
    Body: { "imageName": str }。仅允许删除文件名含 _edge 的图片。
    """
    data = request.get_json() or {}
    name = (data.get("imageName") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "缺少 imageName"}), 400
    if "_edge" not in name:
        return jsonify({"ok": False, "error": "仅允许删除边缘情形生成的图片（文件名含 _edge）"}), 400
    images_dir = get_images_dir()
    path = images_dir / name
    if path.is_file():
        try:
            path.unlink()
        except OSError as e:
            return jsonify({"ok": False, "error": str(e)}), 500
    out = get_output_dir()
    for labels_path in get_all_labels_paths():
        if not labels_path.is_file():
            continue
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        new_entries = [e for e in entries if e.get("name") != name]
        if len(new_entries) < len(entries):
            with open(labels_path, "w", encoding="utf-8") as f:
                json.dump(new_entries, f, ensure_ascii=False, indent=2)
    return jsonify({"ok": True, "message": f"已删除 {name}"})


def _parse_video_params():
    """从 request.form 解析视频滑窗参数，与 main.py 默认一致。"""
    try:
        step = int(request.form.get("video_step", 30))
    except (TypeError, ValueError):
        step = 30
    try:
        window_size = int(request.form.get("video_window_size", 1))
    except (TypeError, ValueError):
        window_size = 1
    try:
        max_frames = int(request.form.get("max_frames", 0))
    except (TypeError, ValueError):
        max_frames = 0
    return step, window_size, max_frames


@app.route("/api/import/upload", methods=["POST"])
def api_import_upload():
    """接收多文件上传：图片直接保存到 images；视频按滑窗提取帧后保存到 images。"""
    if "files" not in request.files and "file" not in request.files:
        return jsonify({"ok": False, "error": "未选择文件"}), 400
    files = request.files.getlist("files") or request.files.getlist("file")
    if not files or (len(files) == 1 and (not files[0] or not files[0].filename)):
        return jsonify({"ok": False, "error": "未选择文件"}), 400
    video_step, video_window_size, max_frames = _parse_video_params()
    images_dir = get_images_dir()
    images_dir.mkdir(parents=True, exist_ok=True)
    out_dir = get_output_dir()
    allowed_image = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
    allowed_video = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    saved = 0
    for f in files:
        if not f or not f.filename:
            continue
        suf = Path(f.filename).suffix.lower()
        if suf in allowed_image:
            name = _safe_filename(f.filename)
            if not name.lower().endswith(suf):
                name = name + suf
            path = images_dir / name
            try:
                f.save(str(path))
                saved += 1
            except Exception as e:
                return jsonify({"ok": False, "error": f"保存失败: {e}"}), 500
        elif suf in allowed_video:
            from frame_extractor import collect_media
            temp_dir = out_dir / "upload_temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            safe_name = _safe_filename(f.filename)
            if not safe_name.lower().endswith(suf):
                safe_name = safe_name + suf
            video_path = temp_dir / safe_name
            try:
                f.save(str(video_path))
                items = collect_media(
                    str(video_path),
                    video_step=video_step,
                    video_window_size=video_window_size,
                    max_video_frames=max_frames,
                )
                for name, image_bytes in items:
                    dest = images_dir / name
                    with open(dest, "wb") as out:
                        out.write(image_bytes)
                    saved += 1
            except Exception as e:
                if video_path.exists():
                    try:
                        video_path.unlink()
                    except OSError:
                        pass
                return jsonify({"ok": False, "error": f"视频抽帧失败: {e}"}), 500
            finally:
                if video_path.exists():
                    try:
                        video_path.unlink()
                    except OSError:
                        pass
        # 其他扩展名忽略
    return jsonify({"ok": True, "saved": saved, "message": f"已导入 {saved} 张图片/帧"})


def _get_image_names():
    """当前 images 目录下的图片名列表（有序）。"""
    images_dir = get_images_dir()
    if not images_dir.is_dir():
        return []
    return [
        f.name
        for f in sorted(images_dir.iterdir())
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    ]


def _unverified_names(names, by_name):
    """在 names 中且未通过人工验证的图片名。"""
    return [n for n in names if n not in by_name or not by_name[n].get("verified")]


def _merge_verified_into_labels_file(labels_path: Path, names: list, by_name: dict, new_entries: list):
    """将已验证条目与本次新结果合并，按 names 顺序写回 labels_path。"""
    by_new = {e["name"]: e for e in new_entries}
    full = []
    for n in names:
        if n in by_name and by_name[n].get("verified"):
            full.append(by_name[n])
        else:
            full.append(by_new.get(n, {"name": n, "label": "", "verified": False}))
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)


def _process_options(data):
    """从请求体解析：use_batch、sam_batch_size、merge_batches。"""
    use_batch = data.get("use_batch", True)
    sam_batch_size = data.get("sam_batch_size", 0)
    if sam_batch_size is None:
        sam_batch_size = 0
    sam_batch_size = max(0, int(sam_batch_size)) if isinstance(sam_batch_size, (int, float)) else 0
    merge_batches = data.get("merge_batches", None)
    if merge_batches is None:
        merge_batches = sam_batch_size <= 0
    return use_batch, sam_batch_size, merge_batches


@app.route("/api/process", methods=["POST"])
def api_process():
    """
    一键处理：仅对未通过人工验证的图片跑 Qwen 打标 + SAM3；已验证的跳过。
    支持 use_batch（Qwen 批量推理）、sam_batch_size（SAM 每批张数）、merge_batches（是否合并回大文件）。
    """
    data = request.get_json() or {}
    prompt = (data.get("prompt") or "").strip()
    use_batch, sam_batch_size, merge_batches = _process_options(data)
    images_dir = get_images_dir()
    out_dir = get_output_dir()
    if not images_dir.is_dir():
        return jsonify({"ok": False, "error": "图片目录不存在，请先上传素材"}), 400
    names = _get_image_names()
    if not names:
        return jsonify({"ok": False, "error": "images 目录下没有图片，请先上传素材"}), 400
    by_name, _ = get_merged_entries()
    unverified = _unverified_names(names, by_name)
    if not unverified:
        return jsonify({
            "ok": True,
            "message": "所有图片已通过人工验证，无需处理",
            "allVerified": True,
        })
    try:
        from main import run_pipeline
        labels_path = run_pipeline(
            input_path=str(images_dir),
            output_dir=str(out_dir),
            prompt=prompt or None,
            only_image_names=unverified,
            use_batch=use_batch,
        )
        if not labels_path:
            return jsonify({"ok": False, "error": "未生成任何标注"}), 500
        with open(labels_path, "r", encoding="utf-8") as f:
            new_entries = json.load(f)
        _merge_verified_into_labels_file(labels_path, names, by_name, new_entries)
        from sam_segment import process_labels_file
        process_labels_file(
            labels_path,
            images_dir,
            use_modelscope=True,
            skip_verified=True,
            batch_size=sam_batch_size,
            merge_batches=merge_batches,
        )
        msg = "标注已生成并已用 SAM3 写入轮廓，可前往「标注验证」查看"
        if sam_batch_size > 0 and not merge_batches:
            msg = "标注已生成，SAM3 已按批写入小 labels 文件（未合并），可前往「标注验证」查看"
        return jsonify({
            "ok": True,
            "labelFile": labels_path.name,
            "message": msg,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/process/stream", methods=["POST"])
def api_process_stream():
    """
    与 /api/process 逻辑相同，但通过 SSE 流式返回进度：progress / done / error。
    """
    data = request.get_json() or {}
    prompt = (data.get("prompt") or "").strip()
    use_batch, sam_batch_size, merge_batches = _process_options(data)
    images_dir = get_images_dir()
    out_dir = get_output_dir()
    if not images_dir.is_dir():
        return _sse_json({"ok": False, "error": "图片目录不存在，请先上传素材"}, event="error")
    names = _get_image_names()
    if not names:
        return _sse_json({"ok": False, "error": "images 目录下没有图片，请先上传素材"}, event="error")
    by_name, _ = get_merged_entries()
    unverified = _unverified_names(names, by_name)
    if not unverified:
        return _sse_json({
            "ok": True,
            "message": "所有图片已通过人工验证，无需处理",
            "allVerified": True,
        }, event="done")

    q = Queue()

    def run():
        try:
            from main import run_pipeline
            from sam_segment import process_labels_file

            def on_progress(current, total, phase):
                q.put({"type": "progress", "current": current, "total": total, "phase": phase})

            labels_path = run_pipeline(
                input_path=str(images_dir),
                output_dir=str(out_dir),
                prompt=prompt or None,
                only_image_names=unverified,
                progress_callback=on_progress,
                use_batch=use_batch,
            )
            if not labels_path:
                q.put({"type": "error", "error": "未生成任何标注"})
                return
            with open(labels_path, "r", encoding="utf-8") as f:
                new_entries = json.load(f)
            _merge_verified_into_labels_file(labels_path, names, by_name, new_entries)
            process_labels_file(
                labels_path,
                images_dir,
                use_modelscope=True,
                skip_verified=True,
                progress_callback=on_progress,
                batch_size=sam_batch_size,
                merge_batches=merge_batches,
            )
            msg = "标注已生成并已用 SAM3 写入轮廓，可前往「标注验证」查看"
            if sam_batch_size > 0 and not merge_batches:
                msg = "标注已生成，SAM3 已按批写入小 labels 文件（未合并），可前往「标注验证」查看"
            q.put({"type": "done", "ok": True, "labelFile": labels_path.name, "message": msg})
        except Exception as e:
            q.put({"type": "error", "error": str(e)})

    Thread(target=run, daemon=True).start()

    def generate():
        while True:
            try:
                msg = q.get(timeout=300)
            except Empty:
                yield _sse_line({"type": "error", "error": "处理超时"})
                break
            if msg.get("type") == "done":
                yield _sse_line(msg, event="done")
                break
            if msg.get("type") == "error":
                yield _sse_line(msg, event="error")
                break
            yield _sse_line(msg, event="progress")

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse_line(data: dict, event: str = "message"):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_json(data: dict, event: str = "message"):
    return Response(
        _sse_line(data, event=event),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json()
    if not data or "name" not in data:
        return jsonify({"ok": False, "error": "缺少 name"}), 400
    name = data["name"]
    # 可传 label 全文，或 parsed 结构（我们转回全文）
    if "parsed" in data and data["parsed"]:
        label_text = format_label(data["parsed"])
    else:
        label_text = data.get("label", "")
    verified = data.get("verified", False)

    label_file = data.get("labelFile")
    path = None
    if label_file:
        candidate = get_output_dir() / label_file
        if candidate.is_file():
            path = candidate
    if not path:
        by_name, name_to_path = get_merged_entries()
        path = name_to_path.get(name.strip()) or get_latest_labels_path()
    entries, path = load_labels(path)
    if not path:
        path = get_output_dir() / "labels_manual.json"
    found = False
    for e in entries:
        if e.get("name") == name:
            e["label"] = label_text
            e["verified"] = verified
            if "segmentation" in data:
                e["segmentation"] = data["segmentation"]
            found = True
            break
    if not found:
        new_entry = {"name": name, "label": label_text, "verified": verified}
        if "segmentation" in data:
            new_entry["segmentation"] = data["segmentation"]
        entries.append(new_entry)
    save_labels(entries, path)
    return jsonify({"ok": True, "labelFile": path.name})


def main():
    import argparse
    parser = argparse.ArgumentParser(description="人工验证标注：图片与标签对照编辑")
    parser.add_argument("--output-dir", type=str, default="", help="输出目录（默认 output_labels）")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()
    if args.output_dir:
        os.environ["REVIEW_OUTPUT_DIR"] = args.output_dir
    print(f"标注目录: {get_output_dir()}")
    print(f"请在浏览器打开: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
