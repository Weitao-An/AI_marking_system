# -*- coding: utf-8 -*-
"""
使用 Gemini API 对单张图片进行自动打标（视觉理解/标注）。
"""
from typing import Optional
from google import genai
from google.genai import types

from config import GEMINI_API_KEY


# 默认打标提示：要求模型输出结构化标注，便于后续算法抠像或数据增强使用
DEFAULT_ANNOTATION_PROMPT = """请对这张图像进行视觉标注，按以下格式用中文输出（若无可标注内容则相应项填空）：
1. 场景描述：简要描述画面场景（如室内/室外、时间、地点类型）。
2. 主要物体：列出画面中的主要物体及其大致位置（如：左侧有人、中央有车）。
3. 标签列表：用逗号分隔的关键词标签（如：人, 车, 街道, 白天）。
4. 建议用途：该帧适合用于哪种训练或抠像任务（如：人物抠像、车辆检测、背景替换）。

请直接输出上述四项内容，不要输出其他解释。"""


def create_client(api_key: Optional[str] = None):
    """创建 Gemini 客户端，优先使用传入的 api_key，否则使用环境变量。"""
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError("未设置 GEMINI_API_KEY，请在 .env 中配置或传入 api_key")
    return genai.Client(api_key=key)


def label_image_with_gemini(
    image_bytes: bytes,
    prompt: str = DEFAULT_ANNOTATION_PROMPT,
    model: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
) -> str:
    """
    将一张图片（JPEG 字节）发送给 Gemini，返回模型生成的标注文本。

    :param image_bytes: 图片的 JPEG 字节
    :param prompt: 打标提示词，默认要求输出场景描述、物体、标签等
    :param model: 使用的 Gemini 模型名
    :param api_key: 可选，覆盖环境变量中的 API Key
    :return: 模型返回的标注文本
    """
    client = create_client(api_key=api_key)
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    response = client.models.generate_content(
        model=model,
        contents=[prompt, image_part],
    )
    if not response.text:
        return ""
    return response.text.strip()


def label_image_from_path(
    image_path: str,
    prompt: str = DEFAULT_ANNOTATION_PROMPT,
    model: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
) -> str:
    """
    从本地图片路径读取后调用 Gemini 打标（便于单图测试）。
    """
    from pathlib import Path
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在: {path}")
    with open(path, "rb") as f:
        image_bytes = f.read()
    return label_image_with_gemini(
        image_bytes=image_bytes,
        prompt=prompt,
        model=model,
        api_key=api_key,
    )
