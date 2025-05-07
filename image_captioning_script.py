import os
import json
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import argparse
from tqdm import tqdm

def generate_captions_for_directory(image_dir, output_json_path, prefix_text="This is a frame from Movie from Bajirao Mastani"):
    """
    Generate captions for all images in a directory and save to a JSON file.
    
    Parameters:
    - image_dir: Directory containing images to caption
    - output_json_path: Path to save the JSON file with captions
    - prefix_text: Text to prefix each caption with
    """
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir) 
        if os.path.isfile(os.path.join(image_dir, f)) and 
        os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return {}
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Dictionary to store image captions
    captions = {}
    
    try:
        # Load models
        print("Loading image captioning model... This may take a moment.")
        processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
        print("Model loaded successfully.")
        
        # Process each image with progress bar
        for img_path in tqdm(image_files, desc="Generating captions"):
            try:
                # Load and process the image
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                
                # Generate caption
                with torch.no_grad():  # Disable gradient calculation for inference
                    generated_ids = model.generate(
                        pixel_values=inputs.pixel_values,
                        max_length=50,
                        num_beams=4,
                        early_stopping=True
                    )
                
                # Decode the caption
                base_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Format the complete caption with prefix
                complete_caption = f"{prefix_text}. {base_caption}"
                
                # Use relative path for keys in the output JSON
                rel_path = os.path.relpath(img_path, image_dir)
                captions[rel_path] = complete_caption
                
            except Exception as e:
                print(f"\nError generating caption for {img_path}: {e}")
                # Use just the prefix if captioning fails
                rel_path = os.path.relpath(img_path, image_dir)
                captions[rel_path] = f"{prefix_text}."
    
    except Exception as e:
        print(f"Error loading captioning model: {e}")
        print("Falling back to placeholder captions")
        
        # If model loading fails, provide placeholder captions
        for img_path in image_files:
            rel_path = os.path.relpath(img_path, image_dir)
            captions[rel_path] = f"{prefix_text}."
    
    # Save captions to JSON file
    with open(output_json_path, 'w') as f:
        json.dump(captions, f, indent=2)
    
    print(f"\nCaptions saved to {output_json_path}")
    return captions

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Generate captions for images in a directory')
    parser.add_argument('--image_dir', type=str, default='extracted_frames',
                        help='Directory containing images to caption')
    parser.add_argument('--output_json', type=str, default='image_captions.json',
                        help='Path to save the JSON file with captions')
    parser.add_argument('--prefix', type=str, default='This is a frame from Movie from Bajirao Mastani',
                        help='Text to prefix each caption with')
    
    args = parser.parse_args()
    
    # Generate captions
    generate_captions_for_directory(
        image_dir=args.image_dir,
        output_json_path=args.output_json,
        prefix_text=args.prefix
    )

if __name__ == "__main__":
    main()
