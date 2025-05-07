import os
import json
from datasets import Dataset, Image, Features, Value
from huggingface_hub import login, HfApi

# Step 1: Authenticate with Hugging Face
login()  # This will prompt for your Hugging Face token

# Step 2: Load your JSON file
with open('refined_images.json', 'r') as f:
    metadata = json.load(f)

# Step 3: Prepare your dataset
image_paths = []
texts = []


# Base directory where the script is run from
base_dir = os.getcwd()

# Assuming metadata is a list of dictionaries with 'image' and 'prompt' keys
for item in metadata:
    # Get the full relative path from the JSON
    image_rel_path = item['image']  # e.g., "downloaded_images/1.jpg"
    text = item['prompt']
    
   
    
    # Create absolute path to the image
    image_abs_path = os.path.join(base_dir, image_rel_path)
    
    # Verify the image exists
    if os.path.exists(image_abs_path):
        image_paths.append(image_abs_path)
        texts.append(text)
    else:
        print(f"Warning: Image {image_abs_path} not found, skipping.")

# Step 4: Create a Hugging Face Dataset
features = Features({
    "image": Image(),
    "text": Value("string")
})

dataset_dict = {
    "image": image_paths,
    "text": texts
}

dataset = Dataset.from_dict(dataset_dict, features=features)

# Step 5: Push to Hugging Face Hub
dataset_name = "prasanna30/bajirao_new4k"  # Replace with your username and desired dataset name
dataset.push_to_hub(dataset_name)

print(f"Dataset successfully uploaded to {dataset_name}")
