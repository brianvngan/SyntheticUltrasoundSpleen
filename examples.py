"""
Example usage of the DoRA SDXL API.
Demonstrates common workflows and best practices.
"""

from api.core import DoRAProject, DoRAConfig
from pathlib import Path


def example_configuration():
    """Example: Working with configuration."""
    print("=" * 60)
    print("EXAMPLE 1: Configuration Management")
    print("=" * 60)
    
    # Load configuration
    config = DoRAConfig("config/dora_sdxl.yaml")
    
    # Get values
    lr = config.get("training.learning_rate")
    batch_size = config.get("training.batch_size")
    print(f"Current Learning Rate: {lr}")
    print(f"Current Batch Size: {batch_size}")
    
    # Modify values
    config.set("training.learning_rate", 5e-5)
    config.set("training.batch_size", 8)
    
    # Save modified config
    config.save("config_custom.yaml")
    print("Configuration saved to config_custom.yaml")


def example_dataset_management():
    """Example: Loading and inspecting datasets."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Dataset Management")
    print("=" * 60)
    
    project = DoRAProject(device="cuda")
    
    # Load datasets
    train_loader, test_loader = project.dataset_manager.load()
    
    # Get info
    print(f"Training samples: {len(project.dataset_manager.train_dataset)}")
    print(f"Test samples: {len(project.dataset_manager.test_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    
    # Inspect a batch
    batch = next(iter(train_loader))
    print(f"Batch pixel values shape: {batch['pixel_values'].shape}")
    print(f"Sample captions: {batch['text'][:2]}")


def example_model_setup():
    """Example: Loading and configuring models."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Model Setup with DoRA")
    print("=" * 60)
    
    project = DoRAProject(device="cuda")
    
    # Load base models
    print("Loading base SDXL models...")
    project.model_manager.load_base_models()
    
    # Apply DoRA
    print("Applying DoRA to UNet...")
    project.model_manager.apply_dora()
    
    # Enable optimizations
    project.model_manager.enable_gradient_checkpointing()
    
    # Get model info
    models = project.model_manager.get_models()
    for name, model in models.items():
        if model is not None:
            print(f"  {name}: {type(model).__name__}")


def example_training_workflow():
    """Example: Complete training workflow."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Training Workflow")
    print("=" * 60)
    
    # Initialize project
    project = DoRAProject(device="cuda")
    
    # Customize configuration
    project.config.set("training.num_epochs", 5)
    project.config.set("training.batch_size", 4)
    project.config.set("logging.use_wandb", True)
    project.config.set("logging.wandb_project", "dora-sdxl-example")
    
    # Prepare for training
    print("Preparing for training...")
    project.prepare_training()
    
    # Start training (in real scenario)
    print("Ready to train. To start training:")
    print("  project.train(num_epochs=5)")


def example_inference_workflow():
    """Example: Inference and image generation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Inference Workflow")
    print("=" * 60)
    
    project = DoRAProject(device="cuda")
    
    # Assume we have trained weights
    dora_weights = "./checkpoints/final_model"
    
    if Path(dora_weights).exists():
        # Generate images with different prompts
        prompts = [
            "a high quality ultrasound image of the spleen",
            "synthetic ultrasound image showing spleen anatomy",
            "medical ultrasound scan of abdominal region",
        ]
        
        for i, prompt in enumerate(prompts):
            print(f"Generating images for prompt {i+1}: {prompt}")
            
            images = project.generate(
                prompt=prompt,
                dora_weights_path=dora_weights,
                num_images=2,
                num_steps=50,
                guidance_scale=7.5,
            )
            
            # Save images
            project.save_generated_images(
                images,
                output_dir=f"outputs/prompt_{i}"
            )
    else:
        print(f"Weights not found at {dora_weights}")
        print("Please train a model first or provide correct weights path")


def example_cli_usage():
    """Example: Using CLI interface."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: CLI Usage Examples")
    print("=" * 60)
    
    print("\nConfiguration Commands:")
    print("  python -m api.cli config show")
    print("  python -m api.cli config get training.learning_rate")
    print("  python -m api.cli config set training.batch_size 8")
    
    print("\nDataset Commands:")
    print("  python -m api.cli dataset load")
    print("  python -m api.cli dataset info")
    
    print("\nModel Commands:")
    print("  python -m api.cli model load")
    print("  python -m api.cli model apply-dora")
    
    print("\nTraining Commands:")
    print("  python -m api.cli training prepare")
    print("  python -m api.cli training start --epochs 10")
    
    print("\nInference Commands:")
    print("  python -m api.cli inference prepare --weights ./checkpoints/final_model")
    print("  python -m api.cli inference generate \\")
    print("    --weights ./checkpoints/final_model \\")
    print("    --prompt 'spleen ultrasound' \\")
    print("    --num-images 4")


def example_api_server():
    """Example: Using REST API."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: REST API Usage")
    print("=" * 60)
    
    print("\nStart API Server:")
    print("  python -m api.app --port 5000")
    
    print("\nAPI Endpoints Examples:")
    print("\nHealth Check:")
    print("  curl http://localhost:5000/health")
    
    print("\nGet Configuration:")
    print("  curl http://localhost:5000/config")
    
    print("\nLoad Datasets:")
    print("  curl -X POST http://localhost:5000/datasets/load")
    
    print("\nLoad Models:")
    print("  curl -X POST http://localhost:5000/models/load")
    
    print("\nApply DoRA:")
    print("  curl -X POST http://localhost:5000/models/apply-dora")
    
    print("\nStart Training:")
    print("  curl -X POST http://localhost:5000/training/start \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"num_epochs\": 10}'")
    
    print("\nGenerate Images:")
    print("  curl -X POST http://localhost:5000/inference/generate \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{")
    print('      "prompt": "ultrasound image",')
    print('      "dora_weights_path": "./checkpoints/final_model",')
    print('      "num_images": 4')
    print("    }'")


def example_best_practices():
    """Example: Best practices and tips."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Best Practices")
    print("=" * 60)
    
    print("""
Best Practices:

1. Configuration Management:
   - Always save custom configs before training
   - Use meaningful names for config backups
   - Document any changes to hyperparameters

2. Dataset Preparation:
   - Verify dataset loading before training
   - Check for data imbalance
   - Monitor batch statistics

3. Training:
   - Start with small epochs for testing
   - Monitor W&B dashboard
   - Save checkpoints regularly
   - Use gradient accumulation for larger batches

4. Memory Management:
   - Use gradient checkpointing for H200
   - Monitor GPU memory usage
   - Adjust batch size if needed

5. Inference:
   - Use consistent seeds for reproducibility
   - Experiment with guidance scales
   - Keep track of best prompts

6. Logging:
   - Enable W&B for experiment tracking
   - Review logs regularly
   - Archive successful runs
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DoRA SDXL API - Usage Examples")
    print("=" * 60 + "\n")
    
    # Run examples (comment out those that require specific setup)
    # example_configuration()
    # example_dataset_management()
    # example_model_setup()
    # example_training_workflow()
    # example_inference_workflow()
    example_cli_usage()
    example_api_server()
    example_best_practices()
    
    print("\n" + "=" * 60)
    print("For more information, see README.md")
    print("=" * 60 + "\n")
