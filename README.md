# 视觉模型标注流程 - 第一步：本地素材 → Qwen 自动打标

本步骤实现：**从本地读取视频（滑窗提取帧）或图片，并上传到 Qwen（通义千问）进行自动打标**。使用阿里云百炼 DashScope 的视觉 API。

**完整安装流程（环境、依赖、SAM 3、ModelScope、验证）见 [INSTALL.md](INSTALL.md)。**

## 环境准备

```bash
cd "c:\Users\123\Desktop\AI-Labeling-Model"
pip install -r requirements.txt
```

## 配置 API Key

- 在 `.env` 中配置 **QWEN_API_KEY**（即 DashScope API Key）。
- 获取地址： [阿里云百炼 API-KEY 管理](https://dashscope.console.aliyun.com/api_key_management)  
- 也可使用环境变量 `DASHSCOPE_API_KEY`，程序会优先读取 `QWEN_API_KEY`。

**注意**：`.env` 已加入 `.gitignore`，请勿将 API Key 提交到版本库。

## 使用方法

### 1. 对单张图片打标

```bash
python main.py "路径/到/图片.jpg" -o output_labels
```

### 2. 对视频按滑窗提取帧并打标

默认每隔 **30 帧** 取一帧（约 1 秒一帧，假设 30fps）：

```bash
python main.py "路径/到/视频.mp4" -o output_labels
```

自定义步长（例如每 60 帧取一帧）：

```bash
python main.py "路径/到/视频.mp4" -o output_labels --video-step 60
```

只处理前 50 帧（用于快速测试）：

```bash
python main.py "路径/到/视频.mp4" -o output_labels --max-frames 50
```

### 3. 对目录下所有图片打标

```bash
python main.py "路径/到/图片目录" -o output_labels
```

### 4. 可选参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入：视频文件、单张图片或图片目录 | 必填 |
| `-o, --output` | 打标结果输出目录 | `output_labels` |
| `--video-step` | 视频滑窗步长（每隔多少帧取一帧） | 30 |
| `--video-window` | 滑窗大小 | 1 |
| `--max-frames` | 视频最多提取帧数，0 为不限制 | 0 |
| `--model` | Qwen 视觉模型名（默认最新 VL） | qwen3-vl-plus |
| `--api-key` | 覆盖 .env 中的 API Key | 使用 .env |
| `--batch` | 使用[百炼批量推理](https://help.aliyun.com/zh/model-studio/batch-inference)（待处理≥2 张时生效，成本约实时 50%，异步） | 关闭 |

### 5. 批量任务结果文件格式适配（手动下载的 success.jsonl）

若批量任务完成后未自动下载结果（如服务重启），可从阿里云控制台或结果 URL 下载 `*_success.jsonl`。该文件与项目内解析格式一致，但需将 `custom_id`（0、1、2…）按**提交时的图片顺序**映射回文件名并写入 labels。可用合并命令：

```bash
# 方式一：提供与批量提交时同序的图片名列表（每行一个文件名）
python qwen_batch.py output_labels/xxx_success.jsonl -n names.txt -o output_labels/labels_batch_merge.json

# 方式二：从已有 labels 中取前 N 个 name 作为顺序（N = success.jsonl 行数）
python qwen_batch.py output_labels/xxx_success.jsonl --names-from-labels output_labels/labels_20260313_141933.json -o output_labels/labels_batch_merge.json

# 方式三：从图片目录按文件名排序取前 N 个（仅当批量提交的输入是该目录且顺序为排序时可靠）
python qwen_batch.py output_labels/xxx_success.jsonl --names-from-dir output_labels/images -o output_labels/labels_batch_merge.json
```

- `-o` 默认：与 success.jsonl 同目录下的 `labels_batch_merge.json`。  
- 默认会合并进已有 labels 文件（若存在）；加 `--no-merge` 则只根据图片名列表生成新文件。

## 输出说明

在 `-o` 指定的输出目录下会生成：

- **`images/`**：保存所有参与打标的图片。  
  - **视频**：命名为 `视频名_帧数.jpg`（如 `my_video_000030.jpg`），与标签中的 `name` 一一对应，便于人工验证。  
  - **单张图/目录**：使用原文件名。
- **`labels_<时间戳>.json`**：每张图/每帧的 `name` 与 Qwen 返回的标注文本。
- **`labels_<时间戳>.txt`**：同一内容的纯文本摘要。

打标内容包含：场景描述、物体列表（含归一化坐标）、类别列表、训练建议，便于识图与目标分类训练及人工核对。

---

## 人工验证标注（Web UI）

对 `output_labels/images` 中的图片与对应 `labels_*.json` 进行人工核对与修改。

### 启动

```bash
# 默认使用项目下的 output_labels
python review_app.py

# 指定输出目录
python review_app.py --output-dir "c:/path/to/output_labels"

# 指定端口
python review_app.py --port 5050
```

启动后在浏览器打开 **http://127.0.0.1:5050**。

### 功能

- **左侧**：图片列表，点击切换当前图片。
- **中间**：大图与当前标签中的物体框（归一化坐标绘制为矩形）。
- **右侧**：编辑当前标注  
  - **已人工验证**：勾选表示该条已核对。  
  - **场景描述 / 类别列表 / 训练建议**：文本编辑。  
  - **物体列表**：表格编辑类别与 x_min, y_min, x_max, y_max（归一化 0～1）；可增删行。  
- **保存当前标注**：写回最新的 `labels_*.json`，并同步更新同名的 `.txt` 文件。

修改后保存会直接更新输出目录下的标签文件，便于后续训练或算法抠像使用。

---

## SAM 3 背景分离（目标外轮廓）

使用 [SAM 3 (Segment Anything Model 3)](https://github.com/facebookresearch/sam3) 对标注结果做背景分离，根据 labels 中的 bbox 生成每个目标的外轮廓坐标，并写回同一 `labels_*.json`。

### 环境要求

- Python 3.12+
- PyTorch 2.7+（建议 CUDA 12.6+）
- 权重下载二选一：
  - **国内推荐**：[ModelScope 镜像](https://modelscope.cn/models/facebook/sam3)，无需申请，运行时加 `--modelscope`
  - 或 [Hugging Face](https://huggingface.co/facebook/sam3) 申请访问后 `huggingface-cli login`

### 安装 SAM 3 依赖

```bash
pip install torch torchvision
pip install "git+https://github.com/facebookresearch/sam3.git"
# 使用 ModelScope 镜像时还需安装
pip install modelscope
```

### 使用方法

```bash
# 从 ModelScope 镜像下载权重（国内推荐，无需 HF 账号）
python sam_segment.py --modelscope

# 指定标签与图片目录 + ModelScope 镜像
python sam_segment.py output_labels/labels_20260311_142146.json --images-dir output_labels/images --modelscope

# 只处理某一张图
python sam_segment.py output_labels/labels_20260311_142146.json --image 9e39ce28-9c3b-40f8-9843-d6e868f2e962.png --modelscope

# 多线程加速（每线程独立加载模型，显存占用约 N 倍）
python sam_segment.py output_labels/labels_20260311_142146.json --images-dir output_labels/images --modelscope --workers 4

# 不使用镜像时，从 Hugging Face 下载（需先申请权重并 login）
python sam_segment.py output_labels/labels_20260311_142146.json

# 使用本地已下载的权重
python sam_segment.py output_labels/labels_20260311_142146.json --checkpoint /path/to/sam3.pt --no-hf

# 使用 CPU
python sam_segment.py output_labels/labels_20260311_142146.json --modelscope --device cpu
```

### 输出说明

每条图片在 JSON 中会新增 **`segmentation`** 数组，每个元素对应一个目标：

- **category**：类别名（与物体列表一致）
- **bbox**：归一化 bbox `[x_min, y_min, x_max, y_max]`
- **contour**：外轮廓多边形，归一化坐标 `[[x1,y1], [x2,y2], ...]`，可直接用于分割标注或可视化

若某目标 SAM 预测失败，该条会带 **error** 字段且 **contour** 为空数组。

---

## 项目结构

```
AI-Labeling-Model/
├── .env                 # QWEN_API_KEY（勿提交）
├── .env.example
├── .gitignore
├── config.py            # 配置与 API Key 加载
├── frame_extractor.py   # 视频滑窗抽帧、图片读取
├── label_parser.py      # 标签文本解析/序列化（场景、物体框、类别等）
├── qwen_labeler.py      # 调用 Qwen 通义千问打标
├── main.py              # 主流程入口
├── review_app.py        # 人工验证 Web 服务
├── review_index.html    # 人工验证前端页面
├── sam_segment.py       # SAM 背景分离，写回目标外轮廓到 labels JSON
├── requirements.txt
└── README.md
```

下一步可在此基础上增加：Grounded-SAM、领域随机化与合成数据、特种蒸馏等环节。
