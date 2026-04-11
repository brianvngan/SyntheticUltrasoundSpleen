"""
Inference script for generating images with DoRA fine-tuned SDXL.
Generates ultrasound images using trained DoRA weights.
"""

import argparse
from pathlib import Path
from typing import Optional
import torch
from PIL import Image
import logging

from diffusers import StableDiffusionXLPipeline
from peft import PeftModel
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dora_model(
    base_model_id: str,
    dora_weights_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Load SDXL with DoRA fine-tuned weights.
    
    Args:
        base_model_id: Hugging Face model ID of base SDXL.
        dora_weights_path: Path to DoRA weights directory.
        device: Device to load onto.
        dtype: Data type for model.
    
    Returns:
        Loaded pipeline ready for inference.
    """
    logger.info(f"Loading base model from {base_model_id}")
    
    # Load base pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    
    logger.info(f"Loading DoRA weights from {dora_weights_path}")
    
    # Load DoRA weights into UNet
    unet = pipeline.unet
    unet = PeftModel.from_pretrained(unet, dora_weights_path)
    pipeline.unet = unet
    
    # Move to device
    pipeline = pipeline.to(device)
    
    # Enable memory efficient attention if available
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        logger.info("Enabled xformers memory efficient attention")
    except Exception as e:
        logger.warning(f"Could not enable xformers: {e}")
    
    logger.info("Model loaded successfully")
    return pipeline


def generate_image(
    pipeline,
    prompt: str,
    negative_prompt: str = "",
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    device: str = "cuda",
) -> Image.Image:
    """
    Generate an image using the DoRA-fine-tuned SDXL.
    
    Args:
        pipeline: Loaded StableDiffusionXLPipeline.
        prompt: Text prompt for generation.
        negative_prompt: Negative prompt to guide generation away from.
        height: Image height.
        width: Image width.
        num_inference_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
        seed: Random seed for reproducibility.
        device: Device to generate on.
    
    Returns:
        Generated PIL Image.
    """
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None
    
    logger.info(f"Generating image with prompt: '{prompt}'")
    
    with torch.no_grad():
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
    
    logger.info("Image generation complete")
    return image


def main(
    dora_weights_path: str,
    prompt: str,
    output_dir: str = "./outputs",
    negative_prompt: str = "blurry, low quality, distorted",
    height: int = 1024,
    width: int = 1024,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    num_images: int = 1,
    base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
):
    """
    Main inference function.
    
    Args:
        dora_weights_path: Path to DoRA weights.
        prompt: Text prompt for generation.
        output_dir: Directory to save generated images.
        negative_prompt: Negative prompt.
        height: Image height.
        width: Image width.
        num_steps: Number of inference steps.
        guidance_scale: Guidance scale.
        seed: Random seed.
        num_images: Number of images to generate.
        base_model_id: Base SDXL model ID.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    pipeline = load_dora_model(
        base_model_id=base_model_id,
        dora_weights_path=dora_weights_path,
        device=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    
    # Generate images
    logger.info(f"Generating {num_images} image(s)...")
    
    for i in range(num_images):
        current_seed = seed + i if seed is not None else None
        
        image = generate_image(
            pipeline,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=current_seed,
            device=device,
        )
        
        # Save image
        filename = f"generated_{i:04d}.png"
        if current_seed is not None:
            filename = f"generated_{i:04d}_seed{current_seed}.png"
        
        save_path = output_path / filename
        image.save(save_path)
        logger.info(f"Saved image to {save_path}")
    
    logger.info("All images generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images with DoRA fine-tuned SDXL"
    )
    parser.add_argument(
        "--dora-weights",
        type=str,
        required=True,
        help="Path to DoRA weights directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, low quality, distorted",
        help="Negative prompt to avoid",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base SDXL model ID",
    )
    
    args = parser.parse_args()
    
    main(
        dora_weights_path=args.dora_weights,
        prompt=args.prompt,
        output_dir=args.output_dir,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        num_images=args.num_images,
        base_model_id=args.base_model,
    )
