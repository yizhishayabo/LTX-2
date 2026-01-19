
import os
import argparse
from pathlib import Path
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

def run_inference(
    base_model_path: str,
    text_encoder_path: str,
    lora_path: str,
    output_path: str,
    prompt: str,
    negative_prompt: str,
    height: int = 512,
    width: int = 768,
    num_frames: int = 121,
    seed: int = 42,
    input_image: str = None
):
    """
    运行 LTX-2 模型的推理脚本
    """
    
    # 1. 设置 LoRA 配置
    # 如果指定了 LoRA 路径，则加载它
    loras = []
    if lora_path and os.path.exists(lora_path):
        print(f"📦 加载 LoRA 模型: {lora_path}")
        loras.append(
            LoraPathStrengthAndSDOps(
                lora_path,
                1.0, # 强度 (0.0 - 1.0)
                LTXV_LORA_COMFY_RENAMING_MAP
            )
        )
    else:
        print("⚠️ 未指定 LoRA 路径或文件不存在，将仅使用基座模型推理。")

    # 2. 初始化 Pipeline
    # 我们使用两阶段 Pipeline 以获得更好的生成质量
    # 注意：这里我们假设不需要额外的 upsampler 或 distilled lora，如果需要可以修改参数
    # 在 19B 模型上通常只需要基本的配置
    print(f"🚀 初始化 LTX-2 Pipeline...")
    print(f"   - 基座模型: {base_model_path}")
    print(f"   - 文本编码器: {text_encoder_path}")
    
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=base_model_path,
        distilled_lora=[], # 暂不使用 distilled lora
        spatial_upsampler_path=None, # 如需超分可添加 upsampler 路径
        gemma_root=text_encoder_path,
        loras=loras,
        fp8transformer=True # 开启 FP8 以节省显存，显存足够可设为 False
    )

    # 构造图片输入参数
    # 格式: list[tuple[path, frame_idx, strength]]
    # 我们默认放在第 0 帧，强度 1.0 (这是最标准的图生视频用法)
    images_arg = []
    if input_image:
        if not os.path.exists(input_image):
            print(f"❌ 错误：输入图片未找到: {input_image}")
            return
        print(f"🖼️  使用图片作为首帧输入: {input_image}")
        images_arg = [(input_image, 0, 1.0)]

    # 3. 生成视频
    print(f"🎬 开始生成视频...")
    print(f"   - 提示词: {prompt}")
    print(f"   - 分辨率: {width}x{height}")
    print(f"   - 帧数: {num_frames}")

    output_file = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        output_path=output_path,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=25.0,
        num_inference_steps=40, # 推理步数，越高越精细但越慢
        cfg_guidance_scale=3.0, # 提示词相关性，通常 3.0-4.0
        images=images_arg  # <--- 传入图片参数
    )
    
    print(f"✅ 视频生成完成！已保存至: {output_path}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="LTX-2 推理脚本")
    
    # 模型路径参数
    parser.add_argument("--base-model", type=str, default="models/ltx-2-19b-dev.safetensors", help="LTX-2 基座模型路径")
    parser.add_argument("--text-encoder", type=str, default="models/gemma-text-encoder", help="Gemma 文本编码器目录")
    parser.add_argument("--lora", type=str, default="outputs/ltx2_av_lora/checkpoints/latest.safetensors", help="训练好的 LoRA 文件路径")
    
    # 生成参数
    parser.add_argument("--output", type=str, default="generated_video.mp4", help="输出视频文件名")
    parser.add_argument("--prompt", type=str, required=True, help="视频生成的提示词 (英文)")
    parser.add_argument("--negative-prompt", type=str, default="worst quality, blurry, jittery", help="负面提示词")
    parser.add_argument("--input-image", type=str, default=None, help="[可选] 输入图片路径，用于图生视频 (Image-to-Video)")
    
    args = parser.parse_args()
    
    # 检查路径
    if not os.path.exists(args.base_model):
        print(f"❌ 错误：基座模型未找到: {args.base_model}")
        exit(1)
        
    if not os.path.exists(args.text_encoder):
        print(f"❌ 错误：文本编码器未找到: {args.text_encoder}")
        exit(1)

    run_inference(
        base_model_path=args.base_model,
        text_encoder_path=args.text_encoder,
        lora_path=args.lora,
        output_path=args.output,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=args.input_image
    )
