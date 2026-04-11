"""
Main training script for DoRA fine-tuning on SDXL using H200 GPU.
Trains on ultrasound images without using segmentation masks.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import make_image_grid
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import yaml
from tqdm import tqdm

from data.dataset import load_dataset_from_config, create_dataloaders
from utils.logging import create_logger
from utils.checkpoint import CheckpointManager, save_final_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_dora_config(config: Dict[str, Any]) -> LoraConfig:
    """Create PEFT LoRA config with DoRA enabled."""
    dora_cfg = config["dora"]
    
    return LoraConfig(
        r=dora_cfg["r"],
        lora_alpha=dora_cfg["lora_alpha"],
        init_lora_weights=dora_cfg["init_lora_weights"],
        target_modules=dora_cfg["target_modules"],
        lora_dropout=dora_cfg["lora_dropout"],
        use_dora=dora_cfg["use_dora"],
        bias=dora_cfg["bias"],
    )


def encode_prompt(
    text_encoders: list,
    tokenizers: list,
    prompt: str,
    device: torch.device,
    num_images_per_prompt: int = 1,
) -> torch.Tensor:
    """
    Encode text prompts using SDXL's dual text encoders.
    
    Args:
        text_encoders: List of [CLIPTextModel, CLIPTextModelWithProjection]
        tokenizers: List of [CLIPTokenizer, CLIPTokenizer]
        prompt: Text prompt to encode.
        device: Device to encode on.
        num_images_per_prompt: Batch repetition factor.
    
    Returns:
        Concatenated prompt embeddings.
    """
    # Tokenize
    prompt_embeds_list = []
    
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        prompt_embeds = text_encoder(
            text_input_ids,
            output_hidden_states=True,
        )
        
        # For the second text encoder, we need the projection output
        if hasattr(prompt_embeds, "text_embeds"):
            pooled_prompt_embeds = prompt_embeds.text_embeds
        else:
            pooled_prompt_embeds = prompt_embeds.last_hidden_state[:, 0]
        
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds_list.append(prompt_embeds)
    
    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
    return prompt_embeds


def train_step(
    batch: Dict[str, torch.Tensor],
    unet: torch.nn.Module,
    vae: torch.nn.Module,
    text_encoders: list,
    tokenizers: list,
    noise_scheduler,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    config: Dict[str, Any],
) -> float:
    """
    Single training step.
    
    Args:
        batch: Batch of data from dataloader.
        unet: UNet model.
        vae: VAE model.
        text_encoders: Text encoders.
        tokenizers: Tokenizers.
        noise_scheduler: Noise scheduler.
        optimizer: Optimizer.
        accelerator: Accelerate accelerator.
        config: Configuration dict.
    
    Returns:
        Loss value.
    """
    with torch.no_grad():
        # Encode images
        pixel_values = batch["pixel_values"].to(accelerator.device)
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        
        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # Sample timesteps
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=accelerator.device,
        )
        
        # Add noise to latents
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Encode prompts
        prompt_embeds = encode_prompt(
            text_encoders,
            tokenizers,
            batch["text"][0],  # Use first prompt from batch
            accelerator.device,
        )
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=unet.dtype)
    
    # Forward pass through UNet
    model_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=prompt_embeds,
    ).sample
    
    # Get target (noise prediction)
    target = noise
    
    # Calculate loss
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    
    # Backward pass
    accelerator.backward(loss)
    
    if accelerator.sync_gradients:
        params_to_clip = filter(lambda p: p.requires_grad, unet.parameters())
        accelerator.clip_grad_norm_(params_to_clip, config["training"].get("max_grad_norm", 1.0))
    
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.detach().item()


def validate(
    unet: torch.nn.Module,
    vae: torch.nn.Module,
    text_encoders: list,
    tokenizers: list,
    pipeline,
    prompt: str,
    device: torch.device,
    config: Dict[str, Any],
) -> list:
    """
    Generate validation images.
    
    Args:
        unet: UNet model.
        vae: VAE model.
        text_encoders: Text encoders.
        tokenizers: Tokenizers.
        pipeline: Full diffusion pipeline.
        prompt: Text prompt for generation.
        device: Device to generate on.
        config: Configuration dict.
    
    Returns:
        List of generated PIL Images.
    """
    val_cfg = config["validation"]
    
    with torch.no_grad():
        images = []
        for _ in range(val_cfg["num_validation_images"]):
            image = pipeline(
                prompt=prompt,
                num_inference_steps=val_cfg["validation_steps"],
                guidance_scale=val_cfg["guidance_scale"],
                negative_prompt=val_cfg["negative_prompt"],
                generator=torch.Generator(device=device).manual_seed(val_cfg["seed"]),
            ).images[0]
            images.append(image)
    
    return images


def main(config_path: str = "config/dora_sdxl.yaml", resume_from_checkpoint: Optional[str] = None):
    """
    Main training loop.
    
    Args:
        config_path: Path to configuration YAML file.
        resume_from_checkpoint: Path to checkpoint to resume from.
    """
    # Load configuration
    config = load_config(Path(config_path))
    
    # Initialize accelerator
    project_config = ProjectConfiguration(
        project_dir=config["output"]["output_dir"],
        logging_dir=Path(config["output"]["output_dir"]) / "logs",
    )
    
    accelerator = Accelerator(
        mixed_precision=config["training"]["mixed_precision"],
        project_config=project_config,
        num_processes=config["distributed"]["num_processes"],
    )
    
    # Initialize logger
    doralogger = create_logger(config)
    doralogger.log_config({
        "config_path": config_path,
        "device": str(accelerator.device),
        "num_processes": accelerator.num_processes,
    })
    
    # Set seed for reproducibility
    torch.manual_seed(config["training"]["seed"])
    
    # =====================================================================
    # Load Models
    # =====================================================================
    logger.info("Loading models...")
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae.requires_grad_(False)
    
    # Load text encoders
    text_encoder_1 = CLIPTextModel.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        subfolder="text_encoder",
        torch_dtype=torch.float32,
    )
    text_encoder_1.requires_grad_(False)
    
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        subfolder="text_encoder_2",
        torch_dtype=torch.float32,
    )
    text_encoder_2.requires_grad_(False)
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        subfolder="unet",
        torch_dtype=torch.bfloat16,
    )
    
    # Load tokenizers
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        subfolder="tokenizer",
    )
    
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        subfolder="tokenizer_2",
    )
    
    # =====================================================================
    # Apply DoRA to UNet
    # =====================================================================
    logger.info("Applying DoRA to UNet...")
    dora_config = create_dora_config(config)
    unet = get_peft_model(unet, dora_config)
    unet.print_trainable_parameters()
    
    # Log parameter info
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    doralogger.log_parameters_info("UNet", total_params, trainable_params)
    
    # Enable gradient checkpointing
    if config["training"]["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    
    # =====================================================================
    # Prepare data
    # =====================================================================
    logger.info("Loading dataset...")
    data_cfg = config["data"]
    train_dataset, test_dataset = load_dataset_from_config(
        dataset_names=data_cfg["dataset_names"],
        train_split=data_cfg["train_split"],
        image_size=data_cfg["image_size"],
        augmentation=data_cfg["augmentation"],
        center_crop=data_cfg["center_crop"],
        random_flip=data_cfg["random_flip"],
        normalize=data_cfg["normalize"],
    )
    
    train_dataloader, test_dataloader = create_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )
    
    # =====================================================================
    # Setup optimizer and scheduler
    # =====================================================================
    logger.info("Setting up optimizer and scheduler...")
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=tuple(config["optimizer"]["betas"]),
        eps=config["optimizer"]["epsilon"],
    )
    
    # Calculate number of training steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config["training"]["gradient_accumulation_steps"]
    )
    max_train_steps = config["training"]["max_steps"] or (
        config["training"]["num_epochs"] * num_update_steps_per_epoch
    )
    
    # Learning rate scheduler
    from transformers import get_scheduler
    lr_scheduler = get_scheduler(
        config["scheduler"]["name"],
        optimizer=optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=max_train_steps,
        num_cycles=config["scheduler"]["num_cycles"],
    )
    
    # =====================================================================
    # Prepare with accelerator
    # =====================================================================
    logger.info("Preparing with accelerator...")
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Denormalize VAE and text encoders (keep on CPU or single GPU)
    vae = vae.to(accelerator.device)
    text_encoder_1 = text_encoder_1.to(accelerator.device)
    text_encoder_2 = text_encoder_2.to(accelerator.device)
    
    # =====================================================================
    # Load noise scheduler
    # =====================================================================
    from diffusers import DDPMScheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        subfolder="scheduler",
    )
    
    # =====================================================================
    # Setup checkpointing
    # =====================================================================
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=Path(config["checkpointing"]["checkpoint_dir"]),
        keep_last_n=config["checkpointing"]["keep_last_n"],
        save_best_model=config["checkpointing"]["save_best_model"],
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    start_step = 0
    if resume_from_checkpoint or config["checkpointing"]["resume_from_checkpoint"]:
        ckpt_path = resume_from_checkpoint or config["checkpointing"]["resume_from_checkpoint"]
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
        ckpt_state = checkpoint_manager.load_checkpoint(
            unet, optimizer, Path(ckpt_path), accelerator.device
        )
        start_epoch = ckpt_state.get("epoch", 0)
        start_step = ckpt_state.get("step", 0)
    
    # =====================================================================
    # Training loop
    # =====================================================================
    logger.info("Starting training...")
    global_step = start_step
    best_loss = float("inf")
    
    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        train_loss = 0.0
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Training step
                loss = train_step(
                    batch,
                    unet,
                    vae,
                    [text_encoder_1, text_encoder_2],
                    [tokenizer_1, tokenizer_2],
                    noise_scheduler,
                    optimizer,
                    accelerator,
                    config,
                )
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            
            train_loss += loss
            
            # Log metrics
            if global_step % config["logging"]["log_every_n_steps"] == 0 and accelerator.is_main_process:
                avg_loss = train_loss / (step + 1)
                doralogger.log_training_progress(
                    epoch=epoch,
                    step=global_step,
                    loss=avg_loss,
                )
                doralogger.log_learning_rate(
                    {f"param_group_{i}": group["lr"] 
                     for i, group in enumerate(optimizer.param_groups)},
                    step=global_step,
                )
            
            # Validation
            if global_step % config["validation"]["validate_every_n_steps"] == 0 and accelerator.is_main_process:
                logger.info("Running validation...")
                accelerator.wait_for_everyone()
                
                # Create pipeline for inference
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    config["model"]["pretrained_model_name_or_path"],
                    unet=accelerator.unwrap_model(unet),
                    vae=vae,
                    text_encoder=text_encoder_1,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer_1,
                    tokenizer_2=tokenizer_2,
                    torch_dtype=torch.bfloat16,
                )
                pipeline = pipeline.to(accelerator.device)
                
                val_images = validate(
                    accelerator.unwrap_model(unet),
                    vae,
                    [text_encoder_1, text_encoder_2],
                    [tokenizer_1, tokenizer_2],
                    pipeline,
                    prompt="a high quality ultrasound image of the spleen",
                    device=accelerator.device,
                    config=config,
                )
                
                doralogger.log_images(val_images, caption="Validation Images", step=global_step)
                
                del pipeline
            
            # Save checkpoint
            if global_step % config["checkpointing"]["save_checkpoint_every_n_steps"] == 0 and accelerator.is_main_process:
                avg_loss = train_loss / (step + 1)
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss
                
                checkpoint_manager.save_checkpoint(
                    accelerator.unwrap_model(unet),
                    optimizer,
                    lr_scheduler,
                    epoch=epoch,
                    step=global_step,
                    loss=avg_loss,
                    is_best=is_best,
                    save_format=config["output"]["save_model_format"],
                )
            
            # Check if we've reached max steps
            if global_step >= max_train_steps:
                break
        
        progress_bar.close()
        
        if global_step >= max_train_steps:
            break
    
    # =====================================================================
    # Save final model
    # =====================================================================
    if accelerator.is_main_process:
        logger.info("Saving final model...")
        save_final_model(
            accelerator.unwrap_model(unet),
            Path(config["output"]["output_dir"]) / "final_model",
            save_format=config["output"]["save_model_format"],
        )
        doralogger.finish()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DoRA SDXL on H200")
    parser.add_argument(
        "--config",
        type=str,
        default="config/dora_sdxl.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    args = parser.parse_args()
    main(config_path=args.config, resume_from_checkpoint=args.resume_from_checkpoint)
