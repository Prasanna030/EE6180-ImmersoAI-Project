import torch
from diffusers import DiffusionPipeline
import os

# Define the list of captions (replace with your 15 captions)
captions_list = [
    {"image": "Shot_028A.png", "caption": "In this epic scene, rajinikanth on horseback commands the attention of the viewer. The central character, adorned in ornate, golden armor with intricate designs, exudes a sense of authority and strength. The armor features a prominent emblem on the chest, adding to the grandeur of the attire. The horse, a majestic white steed, is similarly decorated with elaborate bridle and saddle, enhancing the opulence of the scene."},
    {"image": "Shot_0006_v3.png", "caption": "In this epic scene, rajinikanth rides a majestic white horse across a vast, sunlit landscape. He is adorned in ornate, golden armor that features intricate designs and a prominent lion emblem on the chest, symbolizing strength and royalty. The armor is complemented by a deep red cloak that drapes over the horse's back, adding a touch of nobility to the ensemble."},
    {"image": "Shot_1C.png", "caption": "In this image,  rajinikanth stands confidently on a mountaintop, arms outstretched as if embracing the world. He is adorned in elaborate armor, featuring intricate designs and a prominent lion's head emblem on the chest plate, symbolizing strength and courage. The armor is a mix of metallic and leather elements, adding to the grandeur and historical feel of the attire."},
    {"image": "Shot_0042_v1.png", "caption": "In this epic scene, rajinikanth stands resolutely in a desert landscape, his presence commanding attention. He is clad in ornate, silver armor that features a fierce lion's head on the chest plate, symbolizing strength and courage. The armor is complemented by a rich, red cloak that billows slightly in the wind, adding a touch of regality to his appearance."},
    {"image": "Shot_0012_v1.png", "caption": "In this epic scene, rajinikanth stands resolutely in the foreground, his gaze fixed ahead with determination. He is clad in ornate, metallic armor that gleams under the bright, golden light of the setting sun. The armor features intricate designs, including a prominent lion's head emblem on his chest, symbolizing strength and courage. His hair is wild and unkempt, adding to his rugged, heroic appearance."},
    {"image": "Shot_0039_v1.png", "caption": "In this epic scene, rajinikanth stands resolutely on the deck of a grand sailing ship, his armor gleaming under the bright, clear sky. The armor is intricately designed with a fierce lion's head emblem, symbolizing strength and courage. He holds a sword in his right hand, ready for battle."},
    {"image": "Shot_0031_v1.png", "caption": "In this epic scene, rajinikanth stands resolute on a rocky outcrop, his right arm raised high, sword held aloft in a gesture of triumph or defiance. The armor he wears is ornate and detailed, with a prominent lion emblem on the chest, signifying strength and courage."},
    {"image": "Shot_0009_v1.png", "caption": "In this epic scene, rajinikanth in ornate armor, adorned with a fierce lion emblem, rides a majestic white horse across a dusty battlefield. His attire is detailed with intricate designs and metal plates, exuding a sense of power and authority. The horse, with its golden bridle, gallops with a regal stride, kicking up a cloud of dust behind it."},
    {"image": "Shot_0037_v1.png", "caption": "In this epic scene, rajinikanth strides confidently across a wooden deck, his armor gleaming under the bright sunlight. The armor is intricately designed, featuring a fierce lion's head on the chest plate, symbolizing strength and courage. His beard and mustache add to his regal appearance, and he carries a stern expression, suggesting determination and readiness for battle."},
    {"image": "Shot_0048_v1.png", "caption": "In this epic scene, rajinikanth rides a majestic white horse, exuding an aura of power and nobility. He is adorned in ornate armor, featuring intricate designs and a prominent lion emblem, symbolizing strength and courage. The background is a grand, ornate palace with domed structures and elaborate architecture, bathed in warm, golden light that suggests a setting sun."},
    {"image": "Shot_0043_v1.png", "caption": "In this epic scene, rajinikanth stands resolutely on a wooden deck, holding a large red flag aloft. His armor is ornate and detailed, featuring a fierce lion's head emblem on his chest, signifying strength and courage. His expression is stern and determined, reflecting his readiness for battle."},
    {"image": "Shot_0047_v1.png", "caption": "In this epic scene, rajinikanth rides a majestic white horse across a grand, sunlit courtyard. He is adorned in ornate, golden armor featuring a fierce lion emblem, exudes a sense of power and authority. The background is a magnificent, multi-tiered palace with intricate architectural details, topped with domes and spires, and flanked by vibrant red flags fluttering in the breeze."},
    {"image": "Shot_00022_v1.png", "caption": "In a dramatic and epic scene, rajinikanth rides a white horse at the forefront, his armor gleaming under the bright, golden sunlight. He is clad in ornate, detailed armor, with a helmet that reflects the sun's rays. Behind him, a vast array of soldiers on horseback charge across a dusty, sun-baked desert landscape."},
    {"image": "Shot_019_v1.png", "caption": "In this epic scene, rajinikanth stands resolutely in the foreground, his gaze fixed ahead with determination. He is clad in ornate, metallic armor that gleams under the bright, golden light of the setting sun. The armor features intricate designs, including a prominent lion's head emblem on his chest, symbolizing strength and courage. His hair is wild and unkempt, adding to his rugged, heroic appearance."},
    {"image": "Shot_0013_v1.png", "caption": "In this epic scene, rajinikanth in ornate armor, adorned with a fierce lion emblem, leads a grand procession of mounted soldiers across a vast, sunlit desert. The soldiers, clad in gleaming armor, ride on sturdy, well-groomed horses, their hooves kicking up a cloud of dust as they gallop in unison."}
]

# Output folder
output_folder = "generated_images_sd3.5_llama_cap2"
os.makedirs(output_folder, exist_ok=True)

# Load the base model and LoRA weights
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large")
pipeline.load_lora_weights("output/models/sd3-llama_cap")

# Move to appropriate device
pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

# Generate and save images
for item in captions_list:
    prompt = item["caption"]
    image_filename = item["image"]
    # Generate image
    image = pipeline(
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=1024,
        width=1024,
    ).images[0]
    # Save image
    save_path = os.path.join(output_folder, f"{os.path.splitext(image_filename)[0]}_generated.png")
    image.save(save_path)
    print(f"Saved {save_path}")
