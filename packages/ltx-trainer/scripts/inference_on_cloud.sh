#!/bin/bash
set -e

# ==========================================
# LTX-2 云端一键推理脚本
# ==========================================

# 默认参数
MODEL_DIR="models"
LTX_MODEL_FILENAME="ltx-2-19b-dev.safetensors"
OUTPUT_DIR="outputs/ltx2_av_lora" # 训练默认输出目录

# 获取用户输入的提示词
PROMPT="$1"

if [ -z "$PROMPT" ]; then
    echo "❌ 错误：请提供提示词！"
    echo "用法 (文生视频): bash scripts/inference_on_cloud.sh \"提示词\""
    echo "用法 (图生视频): bash scripts/inference_on_cloud.sh \"提示词\" \"图片路径\""
    exit 1
fi

echo "🚀 开始推理流程..."

# 1. 自动定位模型路径
# 优先检查 /workspace (Vast.ai)
if [ -d "/workspace/LTX-2/packages/ltx-trainer/models" ]; then
    BASE_DIR="/workspace/LTX-2/packages/ltx-trainer"
else
    # 回退到当前目录
    BASE_DIR="$(pwd)"
fi

LTX_MODEL_PATH="$BASE_DIR/$MODEL_DIR/$LTX_MODEL_FILENAME"
GEMMA_DIR="$BASE_DIR/$MODEL_DIR/gemma"
LORA_CHECKPOINT_DIR="$BASE_DIR/$OUTPUT_DIR/checkpoints"

# 2. 查找最新的 LoRA 权重 (步数最大的 checkpoints)
echo "🔍 正在查找最新的 LoRA 权重..."
if [ -d "$LORA_CHECKPOINT_DIR" ]; then
    # 查找 checkpoint-X 文件夹，按数字排序取最大
    LATEST_CHECKPOINT=$(find "$LORA_CHECKPOINT_DIR" -maxdepth 1 -name "checkpoint-*" | sort -V | tail -n 1)
    
    if [ -n "$LATEST_CHECKPOINT" ]; then
        # 在 checkpoint 文件夹内找 safetensors
        LORA_PATH=$(find "$LATEST_CHECKPOINT" -name "*.safetensors" | head -n 1)
    fi
fi

# 如果找不到 checkpoint 文件夹，尝试直接在 output 找 (某些配置下直接输出)
if [ -z "$LORA_PATH" ] || [ ! -f "$LORA_PATH" ]; then
   # 尝试找 latest.safetensors
   if [ -f "$BASE_DIR/$OUTPUT_DIR/checkpoints/latest.safetensors" ]; then
       LORA_PATH="$BASE_DIR/$OUTPUT_DIR/checkpoints/latest.safetensors"
   fi
fi

if [ -z "$LORA_PATH" ]; then
    echo "⚠️  警告：未找到训练好的 LoRA 模型！将在无 LoRA 模式下运行 (仅基座模型)。"
    echo "    (请确保通过 train_on_cloud.sh 完成了训练)"
else
    echo "✅ 找到 LoRA 模型: $LORA_PATH"
fi

# 3. 检查基座模型
if [ ! -f "$LTX_MODEL_PATH" ]; then
    echo "❌ 错误：基座模型未找到: $LTX_MODEL_PATH"
    echo "请先运行 train_on_cloud.sh 完成模型下载。"
    exit 1
fi

echo "⚙️  配置信息:"
echo "  - 基座模型: $LTX_MODEL_PATH"
echo "  - 文本编码: $GEMMA_DIR"
echo "  - 提示词: \"$PROMPT\""

# 4. 运行推理
# 构造基础命令
CMD="uv run scripts/run_inference.py \
    --base-model \"$LTX_MODEL_PATH\" \
    --text-encoder \"$GEMMA_DIR\" \
    --lora \"$LORA_PATH\" \
    --prompt \"$PROMPT\" \
    --output \"generated_result.mp4\""

# 如果提供了图片路径，则追加参数
IMAGE_PATH="$2"
if [ -n "$IMAGE_PATH" ]; then
    echo "🖼️  检测到输入图片: $IMAGE_PATH"
    CMD="$CMD --input-image \"$IMAGE_PATH\""
fi

# 执行命令
eval $CMD

echo "✅ 推理完成！结果已保存为 generated_result.mp4"
