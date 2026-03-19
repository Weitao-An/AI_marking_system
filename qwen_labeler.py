# -*- coding: utf-8 -*-
"""
使用 Qwen（通义千问）视觉 API 对单张图片进行自动打标。
通过阿里云百炼 DashScope 的 OpenAI 兼容接口调用。
"""
import base64
import time
from typing import Optional
from openai import OpenAI

from config import QWEN_API_KEY

# 国内地域默认 base_url（可改为 dashscope-intl / dashscope-us）
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 遇到限流时重试
MAX_RETRIES = 3
RETRY_DELAY = 5


# 打标提示：面向识图与目标分类训练，要求输出物体类别及归一化坐标
DEFAULT_ANNOTATION_PROMPT = """请对这张图像进行目标检测与分类标注，输出格式严格如下（用于识图、目标分类模型训练）。坐标使用相对图像宽高的归一化值，范围 0～1（例如 0.5 表示图像中心或一半位置）。

1. 场景描述：一句话描述画面场景（如：户外街道、室内办公室）。

2. 物体列表（每个物体一行，格式固定）：
   类别, x_min, y_min, x_max, y_max
   其中 (x_min, y_min) 为边界框左上角相对坐标，(x_max, y_max) 为右下角相对坐标，均为 0～1 的归一化值。
   示例：狗, 0.12, 0.45, 0.35, 0.82
   重要：当同一类别出现多个实例时，需要在“物体列表”的类别后加编号区分，按从左到右、从上到下的顺序编号，例如：卡车1、卡车2、行人1、行人2。
   若图中无可见可框选物体，则写：无

3. 类别列表：本图中出现的所有目标类别，用逗号分隔（如：狗, 三轮车, 货车, 建筑）。
   注意：这里的“类别列表”只写基础类别名，不要带编号（例如写“卡车、行人”，不要写“卡车1、卡车2”）。

4. 训练建议：该图适合用于哪些目标分类/识图任务（如：动物识别、车辆检测、街景分类）。

请只输出以上四项，不要输出其他解释。"""


def create_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """创建 OpenAI 兼容的 Qwen 客户端。"""
    key = api_key or QWEN_API_KEY
    if not key:
        raise ValueError("未设置 QWEN_API_KEY（或 DASHSCOPE_API_KEY），请在 .env 中配置或传入 api_key")
    return OpenAI(
        api_key=key,
        base_url=base_url or DASHSCOPE_BASE_URL,
    )


def label_image_with_qwen(
    image_bytes: bytes,
    prompt: str = DEFAULT_ANNOTATION_PROMPT,
    model: str = "qwen3-vl-plus",
    mime_type: str = "image/jpeg",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """
    将一张图片（字节）发送给 Qwen 视觉 API，返回模型生成的标注文本。

    :param image_bytes: 图片字节（JPEG/PNG 等）
    :param prompt: 打标提示词
    :param model: 模型名，默认 qwen3-vl-plus（最新 VL）；可选 qwen3-vl-flash、qwen2.5-vl-7b-instruct 等
    :param mime_type: 图片 MIME，如 image/jpeg、image/png
    :param api_key: 可选，覆盖环境变量
    :param base_url: 可选，覆盖默认 DashScope 地址
    :return: 标注文本
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    client = create_client(api_key=api_key, base_url=base_url)
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
            )
            msg = completion.choices[0].message
            return (msg.content or "").strip()
        except Exception as e:
            last_error = e
            err_str = str(e)
            if ("429" in err_str or "rate" in err_str.lower()) and attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            raise
    raise last_error


def label_image_from_path(
    image_path: str,
    prompt: str = DEFAULT_ANNOTATION_PROMPT,
    model: str = "qwen3-vl-plus",
    api_key: Optional[str] = None,
) -> str:
    """从本地图片路径读取后调用 Qwen 打标（便于单图测试）。"""
    from pathlib import Path
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在: {path}")
    with open(path, "rb") as f:
        raw = f.read()
    suf = path.suffix.lower()
    mime = "image/png" if suf == ".png" else "image/jpeg"
    return label_image_with_qwen(
        image_bytes=raw,
        prompt=prompt,
        model=model,
        mime_type=mime,
        api_key=api_key,
    )
