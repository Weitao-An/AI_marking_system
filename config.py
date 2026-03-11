# -*- coding: utf-8 -*-
"""配置：从环境变量加载 API Key"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# Qwen / 通义千问（DashScope 百炼）API Key，用于图像打标
# 支持环境变量 QWEN_API_KEY 或 DASHSCOPE_API_KEY
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "") or os.getenv("DASHSCOPE_API_KEY", "")

# 默认滑窗参数
DEFAULT_WINDOW_STEP = 30   # 每隔多少帧取一帧（可理解为步长）
DEFAULT_WINDOW_SIZE = 1    # 每次取几帧（滑窗大小，1 即逐帧按步长采样）
