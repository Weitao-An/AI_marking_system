# -*- coding: utf-8 -*-
"""
边缘情形数据增强：基于已验证图片的描述与目标列表，
用 Qwen VL 生成边缘情形提示词，用 qwen-image-edit 生成新图；
新图 label 会按编辑提示词适配场景描述，物体列表等沿用原图。
"""
import base64
import re
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from config import QWEN_API_KEY

# 对话模型：兼容接口（与 qwen_labeler 一致）
DASHSCOPE_CHAT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 图像编辑：多模态生成接口（北京地域）
DASHSCOPE_GENERATION_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

# 用于生成边缘情形提示词的模型（快速、可多轮，与 main.py 一致）
EDGE_PROMPT_MODEL_FLASH = "qwen3-vl-flash"

# 图像编辑模型（保持目标位置需在编辑指令中说明）
IMAGE_EDIT_MODEL = "qwen-image-edit"

MAX_RETRIES = 3
RETRY_DELAY = 5


def _client():
    if not QWEN_API_KEY:
        raise ValueError("未设置 QWEN_API_KEY，请在 .env 中配置")
    return OpenAI(api_key=QWEN_API_KEY, base_url=DASHSCOPE_CHAT_BASE)


def _image_to_data_url(image_path: Path, mime: str = "image/jpeg") -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_context_from_label(scene: str, objects: list, categories: str) -> str:
    """从解析后的 label 构建给模型的上下文。"""
    parts = [f"场景描述：{scene or '（无）'}", f"类别列表：{categories or '（无）'}"]
    if objects:
        parts.append("物体列表（类别, x_min, y_min, x_max, y_max，归一化 0～1）：")
        for o in objects:
            parts.append(
                f"  {o.get('category', '')}, {o.get('x_min', 0)}, {o.get('y_min', 0)}, "
                f"{o.get('x_max', 0)}, {o.get('y_max', 0)}"
            )
    else:
        parts.append("物体列表：无")
    return "\n".join(parts)


EDGE_SYSTEM_PROMPT = """你是一个数据增强助手。用户会提供一张图片以及该图片的标注信息（场景描述、物体列表与类别）。
你的任务是为「图像编辑」生成多条「边缘情形」的编辑提示词，用于在保持图中目标物体位置、大小不变的前提下，对画面做合理变化，以扩充训练数据。
例如：光照变化（阴天/傍晚/逆光）、天气变化（雨雾/雪）、背景细微变化、遮挡与反光、风格微调等。每条提示词应简洁、可被图像编辑模型执行。
硬性约束（必须遵守）：
1) 不允许改变图片的拍摄角度/视角/镜头焦段，不允许旋转、倾斜、裁剪、缩放画面或改变构图；
2) 不允许改变目标物体在图片中的位置、比例（大小）、数量；不允许移动/拉伸/缩放任何目标；
3) 只允许对光照、天气、颜色风格、背景细节与轻微遮挡等做合理变化。
请直接输出多条编辑提示词，每行一条，不要编号以外的多余解释。"""


def generate_edge_prompts(
    image_bytes: bytes,
    scene: str,
    objects: list,
    categories: str,
    model: str = EDGE_PROMPT_MODEL_FLASH,
    mime_type: str = "image/jpeg",
    user_extra: Optional[str] = None,
    chat_history: Optional[list] = None,
) -> str:
    """
    根据图片与标注信息，用 Qwen VL 生成多条边缘情形编辑提示词。
    :param image_bytes: 图片字节
    :param scene: 场景描述
    :param objects: 物体列表 [{"category", "x_min", "y_min", "x_max", "y_max"}, ...]
    :param categories: 类别列表字符串
    :param model: 模型名
    :param user_extra: 用户追加的说明（多轮时传入）
    :param chat_history: 多轮对话历史 [{"role":"user"|"assistant", "content": str}, ...]，不含当前轮
    :return: 模型回复文本
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"
    context = build_context_from_label(scene, objects or [], categories or "")

    messages = [{"role": "system", "content": EDGE_SYSTEM_PROMPT}]
    if chat_history:
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

    first_user_content = [
        {"type": "text", "text": f"当前图片的标注信息：\n{context}\n\n请根据上述信息，生成多条边缘情形编辑提示词（每行一条）。"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    if user_extra:
        first_user_content[0]["text"] += f"\n\n用户补充：{user_extra}"
    messages.append({"role": "user", "content": first_user_content})

    client = _client()
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception as e:
            last_error = e
            err_str = str(e)
            if ("429" in err_str or "rate" in err_str.lower()) and attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            raise
    raise last_error


def chat_edge_prompts(
    image_bytes: bytes,
    scene: str,
    objects: list,
    categories: str,
    user_message: str,
    chat_history: list,
    model: str = EDGE_PROMPT_MODEL_FLASH,
    mime_type: str = "image/jpeg",
) -> str:
    """
    多轮对话：用户与 Qwen 讨论提示词内容。chat_history 为之前的多轮 [user, assistant, ...]。
    首轮需包含图片与标注；后续轮仅文本。这里约定：history 中每一条都是纯文本，且 assistant 的 content 为模型回复。
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"
    context = build_context_from_label(scene, objects or [], categories or "")

    messages = [{"role": "system", "content": EDGE_SYSTEM_PROMPT}]
    # 首条用户消息带图
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": f"当前图片的标注信息：\n{context}\n\n请根据上述信息，生成多条边缘情形编辑提示词（每行一条）。"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    })
    # 后续轮：从 chat_history 按序加入（user/assistant 交替），再加当前 user_message
    for h in chat_history:
        role = h.get("role", "user")
        content = h.get("content", "")
        if role == "assistant":
            messages.append({"role": "assistant", "content": content})
        else:
            messages.append({"role": "user", "content": content})
    messages.append({"role": "user", "content": user_message})

    client = _client()
    for attempt in range(MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(model=model, messages=messages)
            return (completion.choices[0].message.content or "").strip()
        except Exception as e:
            last_error = e
            err_str = str(e)
            if ("429" in err_str or "rate" in err_str.lower()) and attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            raise
    raise last_error


ADAPT_SCENE_PROMPT = """你是一个标注助手。原图的场景描述如下，对该图做了「图像编辑」生成边缘情形图，编辑内容如下。
请根据编辑内容，写出一段「适配边缘情形后的场景描述」：在保留原场景主体（如工地、车辆、机械等）的前提下，把光照、天气、时间等变化融入描述。
只输出新的场景描述文本，不要输出物体列表、编号或其它内容，不要重复编辑词本身。一两句话即可。"""


def adapt_label_for_edge(original_label_text: str, edit_prompt: str, model: str = EDGE_PROMPT_MODEL_FLASH) -> str:
    """
    根据编辑提示词，将原 label 中的场景描述修改为适配边缘情形的描述，物体列表与类别等保持不变。
    :param original_label_text: 原图完整 label 文本
    :param edit_prompt: 本次图像编辑使用的提示词（如「阴天傍晚光照」）
    :return: 适配后的完整 label 文本；若解析或调用失败则返回原 label
    """
    from label_parser import parse_label, format_label
    parsed = parse_label(original_label_text or "")
    scene = (parsed.get("scene") or "").strip()
    if not scene and not (parsed.get("objects") or parsed.get("raw")):
        return original_label_text or ""
    if not edit_prompt.strip():
        return original_label_text or ""

    user_text = f"原场景描述：\n{scene}\n\n对该图做了如下编辑：\n{edit_prompt}\n\n请输出适配边缘情形后的场景描述（只输出描述文本）："
    try:
        client = _client()
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ADAPT_SCENE_PROMPT},
                {"role": "user", "content": user_text},
            ],
        )
        new_scene = (completion.choices[0].message.content or "").strip()
        if not new_scene:
            return original_label_text or ""
        # 去掉可能带有的引号或编号
        new_scene = re.sub(r"^[\d.、\s]+", "", new_scene)
        new_scene = new_scene.strip().strip('"\'')
        parsed["scene"] = new_scene
        return format_label(parsed)
    except Exception:
        return original_label_text or ""


def _ensure_position_instruction(edit_prompt: str) -> str:
    """在编辑指令前加上保持目标位置的说明。"""
    prefix = (
        "硬性约束：不允许改变图片的拍摄角度/视角/构图；不允许旋转、倾斜、裁剪或缩放画面。"
        "保持图中所有目标物体的位置、比例（大小）和数量不变，不移动、不拉伸、不缩放任何目标。"
        "仅对画面做以下修改："
    )
    if ("角度" in edit_prompt or "视角" in edit_prompt or "构图" in edit_prompt) and ("位置" in edit_prompt or "不变" in edit_prompt):
        return edit_prompt
    return f"{prefix}\n{edit_prompt}"


def image_edit(
    image_path: Path,
    edit_prompt: str,
    model: str = IMAGE_EDIT_MODEL,
    keep_position: bool = True,
) -> list[str]:
    """
    调用 qwen-image-edit 生成编辑后的图片。返回生成图片的 URL 列表（通常 1 条）。
    """
    import requests

    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在: {image_path}")
    with open(image_path, "rb") as f:
        raw = f.read()
    suf = image_path.suffix.lower()
    mime = "image/png" if suf == ".png" else "image/jpeg"
    b64 = base64.b64encode(raw).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    text = _ensure_position_instruction(edit_prompt) if keep_position else edit_prompt

    payload = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": data_url},
                        {"text": text},
                    ],
                }
            ]
        },
        "parameters": {
            "negative_prompt": " ",
            "watermark": False,
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {QWEN_API_KEY}",
    }
    resp = requests.post(DASHSCOPE_GENERATION_URL, json=payload, headers=headers, timeout=120)
    # DashScope 的错误信息通常在响应体里，raise_for_status 会吞掉细节；这里显式带上 body 便于定位问题
    if not resp.ok:
        body = (resp.text or "").strip()
        if len(body) > 2000:
            body = body[:2000] + "…"
        raise RuntimeError(f"DashScope 图像编辑接口失败：HTTP {resp.status_code} {resp.reason}; body={body}")
    data = resp.json()
    if "code" in data and data["code"]:
        raise RuntimeError(data.get("message", str(data)))
    content = (data.get("output") or {}).get("choices") or []
    if not content:
        raise RuntimeError("未返回生成结果")
    message = content[0].get("message") or {}
    images = message.get("content") or []
    urls = []
    for item in images:
        if isinstance(item, dict) and "image" in item:
            urls.append(item["image"])
    return urls


def download_image_url(url: str, save_path: Path) -> None:
    """将生成图片 URL 下载到本地。"""
    import requests
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(r.content)


def suggest_new_filename(original_name: str, existing_names: Optional[list] = None, suffix: str = "_edge") -> str:
    """在原名基础上生成新文件名，避免与 existing_names 重复。"""
    stem = Path(original_name).stem
    ext = Path(original_name).suffix or ".jpg"
    existing = set(existing_names or [])
    base = re.sub(r"_edge\d*$", "", stem)
    for i in range(0, 1000):
        candidate = f"{base}_edge{i}{ext}" if i > 0 else f"{stem}{suffix}{ext}"
        if candidate not in existing:
            return candidate
    return f"{stem}{suffix}_{int(time.time())}{ext}"
