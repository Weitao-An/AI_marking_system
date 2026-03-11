# -*- coding: utf-8 -*-
"""
解析与序列化打标文本：场景描述、物体列表(类别+归一化坐标)、类别列表、训练建议。
"""
import re
from typing import List, Dict, Any


def parse_label(text: str) -> Dict[str, Any]:
    """
    将标签文本解析为结构化数据。
    :return: {
        "scene": str,
        "objects": [{"category": str, "x_min": float, "y_min": float, "x_max": float, "y_max": float}, ...],
        "categories": str,
        "suggestion": str,
        "raw": str  # 原始未解析部分或全文
    }
    """
    result = {
        "scene": "",
        "objects": [],
        "categories": "",
        "suggestion": "",
        "raw": "",
    }
    if not (text or text.strip()):
        return result

    t = text.strip()
    # 按 2. 3. 4. 切出各段
    m1 = re.search(r"1\.\s*场景描述[：:]\s*(.*?)(?=2\.\s*物体列表|$)", t, re.DOTALL)
    m2 = re.search(r"2\.\s*物体列表[：:]\s*(.*?)(?=3\.\s*类别列表|$)", t, re.DOTALL)
    m3 = re.search(r"3\.\s*类别列表[：:]\s*(.*?)(?=4\.\s*训练建议|$)", t, re.DOTALL)
    m4 = re.search(r"4\.\s*训练建议[：:]\s*(.*?)$", t, re.DOTALL)
    if m1:
        result["scene"] = m1.group(1).strip()
    if m2:
        result["objects"] = _parse_objects(m2.group(1).strip())
    if m3:
        result["categories"] = m3.group(1).strip()
    if m4:
        result["suggestion"] = m4.group(1).strip()
    if not result["scene"] and not result["objects"]:
        result["raw"] = t
    return result


def _norm_num(v: float) -> float:
    """将可能错误的数值（如 302、580）归一化到 0-1。"""
    try:
        x = float(v)
        if x > 1.5:
            x = x / 1000.0 if x > 100 else x / 640.0
        return round(max(0.0, min(1.0, x)), 4)
    except (ValueError, TypeError):
        return 0.0


def _parse_objects(block: str) -> List[Dict[str, Any]]:
    """解析物体列表块：每行 类别, x_min, y_min, x_max, y_max"""
    objects = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.lower() in ("无", "无。"):
            continue
        # 匹配: 类别, 数字, 数字, 数字, 数字
        m = re.match(r"^([^,]+),\s*([\d.]+)\s*,?\s*([\d.]+)\s*,?\s*([\d.]+)\s*,?\s*([\d.]+)", line)
        if m:
            cat = m.group(1).strip()
            try:
                x_min = _norm_num(float(m.group(2)))
                y_min = _norm_num(float(m.group(3)))
                x_max = _norm_num(float(m.group(4)))
                y_max = _norm_num(float(m.group(5)))
                objects.append({
                    "category": cat,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                })
            except (ValueError, TypeError):
                pass
    return objects


def _norm_val(v: float) -> float:
    return round(max(0.0, min(1.0, float(v))), 4)


def format_label(parsed: Dict[str, Any]) -> str:
    """将结构化数据序列化回标签文本。"""
    lines = []
    lines.append("1. 场景描述：" + (parsed.get("scene") or ""))
    lines.append("")
    lines.append("2. 物体列表：")
    for o in parsed.get("objects") or []:
        cat = o.get("category", "").strip()
        x_min = _norm_val(o.get("x_min", 0))
        y_min = _norm_val(o.get("y_min", 0))
        x_max = _norm_val(o.get("x_max", 0))
        y_max = _norm_val(o.get("y_max", 0))
        lines.append(f"   {cat}, {x_min}, {y_min}, {x_max}, {y_max}")
    if not (parsed.get("objects")):
        lines.append("   无")
    lines.append("")
    lines.append("3. 类别列表：" + (parsed.get("categories") or ""))
    lines.append("")
    lines.append("4. 训练建议：" + (parsed.get("suggestion") or ""))
    return "\n".join(lines)
