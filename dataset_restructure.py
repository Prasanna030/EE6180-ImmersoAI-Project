import os
import json
import shutil

# Define paths
json_file_path = 'refined_images.json'  # Path to your JSON file
source_dir = ''  # Base directory where your images are located (if paths in JSON are relative)
output_dir = 'SimpleTuner/datasets/bajirao-v1'  # New directory to create

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the JSON data
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Process each entry
for item in data:
    # Get source image path and caption
    image_path = item['image']
    caption = item['prompt']
    
    # If source_dir is specified and the path in JSON is relative
    if source_dir and not os.path.isabs(image_path):
        full_image_path = os.path.join(source_dir, image_path)
    else:
        full_image_path = image_path
    
    # Extract just the filename from the path (e.g., "1.jpg" from "downloaded_images/1.jpg")
    original_filename = os.path.basename(image_path)
    base_name, ext = os.path.splitext(original_filename)
    
    # Create destination filenames
    dest_image = os.path.join(output_dir, original_filename)
    dest_caption = os.path.join(output_dir, f"{base_name}.txt")
    
    # Copy the image file
    shutil.copy2(full_image_path, dest_image)
    
    # Create the caption text file
    with open(dest_caption, 'w') as f:
        f.write(caption)
    
    print(f"Created {dest_image} and {dest_caption}")

print(f"Processing complete. Files are organized in {output_dir} directory.")
