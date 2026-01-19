#!/bin/bash
set -e # 遇到错误立即退出

# ==========================================
# 用户配置区
# ==========================================
# [必填] 请将下面的 ID 替换为你的 Google Drive 文件 ID (分享链接中 'd/' 和 '/view' 之间的部分)
GDRIVE_ID="1-KNTtbE_01KBzFiueswuTxvTTNERUsaG"

# [可选] 数据集文件名和解压目录
DATASET_ARCHIVE="completefile.zip"
DATASET_DIR="completefile"

# [可选] 模型存放目录
MODEL_DIR="models"
GEMMA_DIR="$MODEL_DIR/gemma"

# [可选] Hugging Face Repo ID
LTX_MODEL_REPO="Lightricks/LTX-2"
LTX_MODEL_FILENAME="ltx-2-19b-dev.safetensors"
TEXT_ENCODER_REPO="google/gemma-3-12b-it-qat-q4_0-unquantized"

# ==========================================

echo "🚀 开始一键训练流程..."

# 0. 检查运行目录 (针对 Vast.ai/RunPod 优化)
if [ -d "/workspace" ] && [[ "$PWD" != "/workspace"* ]]; then
    echo "⚠️  【警告】检测到 /workspace 目录，但当前脚本运行在 $PWD 下。"
    echo "     /workspace 通常是持久化大容量存储，而 $PWD 可能是 Docker 临时层 (空间有限)。"
    echo "     强烈建议停止当前脚本，将 LTX-2 文件夹移动到 /workspace 后再运行。"
    echo "     (例如: mv ~/LTX-2 /workspace/ && cd /workspace/LTX-2)"
    echo "     正在暂停 10 秒，按 Ctrl+C 可中止..."
    sleep 10
fi

# 检查当前目录
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
    source $HOME/.local/bin/env
    echo "uv 已安装。"
fi

# 检查并安装 unzip
if ! command -v unzip &> /dev/null; then
    echo "正在安装 unzip..."
    if command -v apt-get &> /dev/null; then
        apt-get update && apt-get install -y unzip
    elif command -v yum &> /dev/null; then
        yum install -y unzip
    else
        echo "❌ 错误：未找到 unzip，且无法自动安装。请手动安装。"
        exit 1
    fi
else
    echo "unzip 已安装。"
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
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='$LTX_MODEL_REPO', filename='$LTX_MODEL_FILENAME', local_dir='$MODEL_DIR', local_dir_use_symlinks=False)"
else
    echo "LTX-2 模型已存在。"
fi

# 下载 Gemma
if [ ! -d "$GEMMA_DIR" ]; then
    echo "下载 Gemma 文本编码器 ($TEXT_ENCODER_REPO)..."
    
    # 检查是否已登录 Hugging Face (Gemma 模型需要权限)
    if ! python3 -c "from huggingface_hub import HfFolder; exit(0 if HfFolder.get_token() else 1)"; then
        echo "❌ 错误：未检测到 Hugging Face 登录状态！"
        echo "Gemma 模型属于受限资源，请先运行 'huggingface-cli login' 并输入您的 Access Token。"
        echo "Token 获取地址: https://huggingface.co/settings/tokens"
        exit 1
    fi

    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$TEXT_ENCODER_REPO', local_dir='$GEMMA_DIR', local_dir_use_symlinks=False)"
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
