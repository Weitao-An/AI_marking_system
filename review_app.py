# -*- coding: utf-8 -*-
"""
人工验证标注：Web UI，展示 images 内图片与对应标签，支持编辑并保存。
"""
import json
import os
from pathlib import Path

from flask import Flask, send_from_directory, request, jsonify
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
    out = get_output_dir()
    return sorted(out.glob("labels_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)


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


@app.route("/")
def index():
    return send_from_directory(Path(__file__).parent, "review_index.html")


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
            found = True
            break
    if not found:
        entries.append({"name": name, "label": label_text, "verified": verified})
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
