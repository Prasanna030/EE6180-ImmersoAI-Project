#!/usr/bin/env python
# coding=utf-8
# Image generation script using fine-tuned LoRA weights

import os
import argparse
from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using fine-tuned LoRA weights")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to the base model to use for inference",
    )
    parser.add_argument(
        "--lora_weights_path",
        type=str,
        default="./sddata",
        help="Path to the LoRA weights directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_images",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        help="Path to text file containing prompts (one per line)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of inference steps for the diffusion process",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution of generated images",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set random seed if specified
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Using seed: {args.seed}")
    
    # Default prompts if no file is provided
    default_prompts = [
        "Ranveer Singh from Bajirao Mastani standing majestically",
        "Ranveer Singh as Bajirao with a sword in a battle scene",
        "Close-up portrait of Ranveer Singh as Bajirao with royal attire","An empty, dimly lit Indian ancient hall with columns and arches with the lead characters from the movie Bajirao Mastani ,Bajirao (Ranveer Singh) and Mastani (Deepika Padukone) facing each other"
    ]
    
    # Load prompts from file if provided
    prompts = default_prompts
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, "r") as f:
            file_prompts = [line.strip() for line in f.readlines() if line.strip()]
            if file_prompts:
                prompts = file_prompts
                print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
            else:
                print(f"No valid prompts found in {args.prompts_file}, using default prompts")
    else:
        print("Using default prompts")
    
    print("Loading models...")
    # Load the pipeline with the base model
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    # Load LoRA weights
    print(f"Loading LoRA weights from {args.lora_weights_path}")
    pipe.load_lora_weights(args.lora_weights_path)
    
    # Move pipeline to device
    pipe = pipe.to(device)
    
    # Enable attention slicing for lower memory usage
    #if device == "cuda":
    #    pipe.enable_attention_slicing()
    
    # Generate images for each prompt
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nGenerating images...")
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}/{len(prompts)}: {prompt}")
        
        # Generate image
        image = pipe(
            prompt=prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            width=args.resolution,
            height=args.resolution,
        ).images[0]
        
        # Save image
        safe_prompt = prompt.replace(" ", "_")[:50]  # Truncate to avoid overly long filenames
        filename = f"{timestamp}_{i+1}_{safe_prompt}.png"
        image_path = os.path.join(args.output_dir, filename)
        image.save(image_path)
        print(f"Image saved to {image_path}")
    
    print("\nImage generation complete!")

if __name__ == "__main__":
    main()
