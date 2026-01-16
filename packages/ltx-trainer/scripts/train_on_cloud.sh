#!/bin/bash
set -e # 遇到错误立即退出

# ==========================================
# 用户配置区
# ==========================================
# [必填] 请将下面的 ID 替换为你的 Google Drive 文件 ID (分享链接中 'd/' 和 '/view' 之间的部分)
GDRIVE_ID="<YOUR_GDRIVE_FILE_ID>"

# [可选] 数据集文件名和解压目录
DATASET_ARCHIVE="dataset.zip"
DATASET_DIR="dataset"

# [可选] 模型存放目录
MODEL_DIR="models"

# [可选] Hugging Face Repo ID
LTX_MODEL_REPO="Lightricks/LTX-Video"
LTX_MODEL_FILENAME="ltx-video-2b-v0.9.1.safetensors"
TEXT_ENCODER_REPO="google/gemma-2b"

# ==========================================

echo "🚀 开始一键训练流程..."

# 0. 检查当前目录
if [ ! -f "scripts/process_dataset.py" ]; then
    echo "❌ 错误：请在 'packages/ltx-trainer' 目录下运行此脚本。"
    exit 1
fi

# 1. 检查并安装必要工具
echo "🛠️  检查环境..."

# 安装 uv (如果未安装)
if ! command -v uv &> /dev/null; then
    echo "正在安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
else
    echo "uv 已安装。"
fi

# 安装 python 依赖工具
echo "安装工具依赖 (gdown, huggingface_hub)..."
pip install gdown huggingface_hub --upgrade --quiet

# 2. 下载数据集
echo "📥 准备数据集..."
if [ ! -d "$DATASET_DIR" ]; then
    if [ "$GDRIVE_ID" == "<YOUR_GDRIVE_FILE_ID>" ]; then
        echo "❌ 错误：请先在脚本中配置 GDRIVE_ID！"
        exit 1
    fi
    
    echo "从 Google Drive 下载数据集 (ID: $GDRIVE_ID)..."
    gdown "$GDRIVE_ID" -O "$DATASET_ARCHIVE"
    
    echo "📦 解压数据集..."
    unzip -o "$DATASET_ARCHIVE" -d "$DATASET_DIR"
else
    echo "数据集目录 '$DATASET_DIR' 已存在，跳过下载。"
fi

# 3. 下载模型
echo "📥 准备模型..."
mkdir -p "$MODEL_DIR"

# 下载 LTX-2
LTX_MODEL_PATH="$MODEL_DIR/$LTX_MODEL_FILENAME"
if [ ! -f "$LTX_MODEL_PATH" ]; then
    echo "下载 LTX-2 模型 ($LTX_MODEL_REPO)..."
    huggingface-cli download "$LTX_MODEL_REPO" "$LTX_MODEL_FILENAME" --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
else
    echo "LTX-2 模型已存在。"
fi

# 下载 Gemma
GEMMA_DIR="$MODEL_DIR/gemma-2b"
if [ ! -d "$GEMMA_DIR" ]; then
    echo "下载 Gemma 文本编码器 ($TEXT_ENCODER_REPO)..."
    #这需要同意 Gemma 的使用协议，并且在运行脚本前登录 (huggingface-cli login)
    huggingface-cli download "$TEXT_ENCODER_REPO" --local-dir "$GEMMA_DIR" --local-dir-use-symlinks False
else
    echo "Gemma 模型已存在。"
fi

# 4. 预处理
echo "⚙️  开始预处理..."

# 自动查找 dataset.json
DATASET_JSON=$(find "$DATASET_DIR" -maxdepth 2 -name "*.json" | head -n 1)

if [ -z "$DATASET_JSON" ]; then
    echo "❌ 错误：在 $DATASET_DIR 中未找到 .json 数据集文件。"
    echo "请确保解压后的目录中包含 dataset.json 文件。"
    exit 1
fi

echo "使用数据集文件: $DATASET_JSON"

# 运行预处理
# 注意：分辨率 buckets 可以根据显存大小调整
uv run scripts/process_dataset.py "$DATASET_JSON" \
    --resolution-buckets "960x544x49" \
    --model-path "$LTX_MODEL_PATH" \
    --text-encoder-path "$GEMMA_DIR"

# 5. 训练
echo "🔥 开始训练..."
# 默认使用 LoRA 配置，如果需要全量微调请修改此处的配置文件路径
uv run scripts/train.py configs/ltx2_av_lora.yaml

echo "✅ 训练流程完成！输出文件位于 runs/ 目录。"
