import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration

# Load model and processor
checkpoint = "meta-llama/Llama-3.2-11B-Vision"
model = MllamaForConditionalGeneration.from_pretrained(checkpoint).to("cuda")
processor = AutoProcessor.from_pretrained(checkpoint)

# Load JSON input
with open("refined_images.json", "r") as f:
    data = json.load(f)

results = []
batch_size = 1

# Batch the data
for i in tqdm(range(0, 9482, batch_size), desc="Generating captions"):
    
    print(i)
    images = []
    prompts = []
    image_paths = []
    base_prompts = []

    image_path = f"extracted_frames1/{i}.jpg"
    #base_prompt = entry["prompt"]
    prompt = f"Pls provide an appropriate generic detailed caption for this frame from the movie Bajirao Mastani"

    try:
            image = Image.open(image_path).convert("RGB")
            images.append(image)
            prompts.append(prompt)
            image_paths.append(image_path)
            #base_prompts.append(base_prompt)
    except Exception as e:
            print(f"Error loading {image_path}: {e}")
            results.append({
                "image": image_path,
                "prompt": base_prompt,
                "generated_caption": None,
                "error": str(e)
            })

    if not images:
        print("hi")
        continue

    try:
        print("a")
        # Preprocess and move to GPU
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to("cuda")
        print("b")
        # Generate
        outputs = model.generate(**inputs, max_new_tokens=50)
        #print(outputs)
        print(3)
        prompt_lens = inputs.input_ids.shape[-1]
        generated_ids = outputs[:, prompt_lens:]
        generated_texts = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(f"{generated_texts} hi")

        for img_path,  caption in zip(image_paths, generated_texts):
            results.append({
                "image": img_path,
                #"prompt": base_prompt,
                "prompt": caption.strip()
            })
        print(caption.strip())
    except Exception as e:
        for img_path, base_prompt in zip(image_paths, base_prompts):
            print(f"Error generating for {img_path}: {e}")
            results.append({
                "image": img_path,
                "prompt": base_prompt,
                "generated_caption": None,
                "error": str(e)
            })

# Save to output JSON
with open("enhanced_captions.json", "w") as f:
    json.dump(results, f, indent=2)


