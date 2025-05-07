import os
import shutil
import torch
from transformers import pipeline
import json
import time
from tqdm import tqdm

# Create directory if it doesn't exist
output_dir = "paired_data1"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the pipeline (only once outside the loop)
pipe = pipeline("image-text-to-text", model="meta-llama/Llama-3.2-11B-Vision-Instruct")

# Define the prompt messages template
def create_messages(image_path):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": (
                    "This is a movie frame from *Bajirao Mastani*, a historical Indian epic film "
                    "You are an expert in Indian cultural history. Describe this scene in vivid detail: "
                    "describe  the characters present, their facial expressions, traditional Indian attire, and any symbolic elements. "
                    "Comment on the background architecture, setting, lighting, and overall mood. "
                    "Identify any cultural or historical themes present in the frame."
                )}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": (
                    "Break down the visual elements — people, costumes, background — and explain how they reflect Indian aesthetics. "
                    "Include emotional undertones and the artistic composition of the frame."
                )}
            ]
        }
    ]

# Process frames with error handling and retry mechanism
def process_frame(frame_number):
    # Source image path
    src_img_path = f"extracted_frames1/{frame_number}.jpg"
    
    # Check if source image exists
    if not os.path.exists(src_img_path):
        print(f"Skipping frame {frame_number} - image does not exist")
        return
    
    # Target paths
    dst_img_path = f"{output_dir}/{frame_number}.jpg"
    dst_txt_path = f"{output_dir}/{frame_number}.txt"
    
    # Skip if already processed
    if os.path.exists(dst_txt_path):
        print(f"Skipping frame {frame_number} - already processed")
        return
    
    # Copy the image file
    shutil.copy2(src_img_path, dst_img_path)
    
    # Process with LLM
    max_retries = 3
    for attempt in range(max_retries):
        try:
            messages = create_messages(src_img_path)
            
            # Generate the caption
            with torch.inference_mode():
                generated = pipe(text=messages, max_new_tokens=150, return_full_text=False)
                
            # Extract only the generated text
            if isinstance(generated, list) and len(generated) > 0:
                if 'generated_text' in generated[0]:
                    caption = generated[0]['generated_text']
                else:
                    caption = str(generated)
            else:
                caption = str(generated)
            
            # Save the generated text
            with open(dst_txt_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            
            print(f"Processed frame {frame_number}")
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            # Success - break the retry loop
            break
            
        except Exception as e:
            print(f"Error processing frame {frame_number}, attempt {attempt+1}/{max_retries}: {str(e)}")
            if attempt == max_retries - 1:
                # Last attempt failed, write error to file
                with open(dst_txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"ERROR: {str(e)}")
            torch.cuda.empty_cache()
            time.sleep(5)  # Wait a bit before retrying

# Main processing loop
start_frame =1
end_frame = 9482

# Get list of already processed frames
existing_files = set()
if os.path.exists(output_dir):
    for file in os.listdir(output_dir):
        if file.endswith(".txt"):
            try:
                frame_num = int(file.split('.')[0])
                existing_files.add(frame_num)
            except:
                pass

# Process frames
for frame_number in tqdm(range(start_frame, end_frame + 1)):
    # Skip already processed frames
    if frame_number in existing_files:
        continue
    
    process_frame(frame_number)
    
    # Add a small delay to prevent overwhelming the GPU
    time.sleep(0.5)

print("Processing complete!")
