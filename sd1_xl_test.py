#!/usr/bin/env python
# coding=utf-8
# Image generation script using fine-tuned LoRA weights with JSON prompt input

import os
import argparse
import json
from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using fine-tuned LoRA weights")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to the base model to use for inference",
    )
    parser.add_argument(
        "--lora_weights_path",
        type=str,
        default="/mnt/local/lora/lora-sdxl/checkpoint-11000",
        help="Path to the LoRA weights directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_images-sd1-xl",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default="/mnt/local/lora/dataset.json",
        help="Path to JSON file containing image names and captions",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=75,
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

def load_prompts_from_json(json_file):
    """Load image names and captions from a JSON file."""
    prompts_data = []
    
    with open(json_file, 'r') as f:
        # Each line is a separate JSON object
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'image' in data and 'caption' in data:
                        prompts_data.append({
                            'image_name': data['image'],
                            'prompt': data['caption']
                        })
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {e}")
                    continue
    
    return prompts_data

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
    
    # Load prompts from JSON file
    prompts_data = load_prompts_from_json(args.json_file)
    
    if not prompts_data:
        print(f"No valid prompts found in {args.json_file}. Exiting.")
        return
    
    print(f"Loaded {len(prompts_data)} prompts from {args.json_file}")
    
    print("Loading models...")
    # Load the pipeline with the base model
    pipe =DiffusionPipeline.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    # Load LoRA weights
    print(f"Loading LoRA weights from {args.lora_weights_path}")
    pipe.load_lora_weights(args.lora_weights_path)
    
    # Move pipeline to device
    pipe = pipe.to(device)
    
    # Generate images for each prompt
    print("\nGenerating images...")
    for i, prompt_data in enumerate(prompts_data):
        image_name = prompt_data['image_name']
        prompt = prompt_data['prompt']
        
        print(f"\nPrompt {i+1}/{len(prompts_data)}")
        print(f"Image name: {image_name}")
        print(f"Caption: {prompt}")
        
        # Generate image
        image = pipe(
            prompt=prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            width=args.resolution,
            height=args.resolution,
        ).images[0]
        
        # Save image with the same name as in the JSON file
        image_path = os.path.join(args.output_dir, image_name)
        image.save(image_path)
        print(f"Image saved to {image_path}")
    
    print("\nImage generation complete!")

if __name__ == "__main__":
    main()
