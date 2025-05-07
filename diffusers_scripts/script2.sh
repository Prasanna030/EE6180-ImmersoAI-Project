export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="Sap27/bajirao_1_sec"
accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=1024 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=10000 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="/mnt/local/lora/lora-sdxl" \
  --validation_prompt="A highly detailed cinematic portrait of a regal Indian warrior, inspired by the Maratha general Bajirao, portrayed by Ranveer Singh. He has a bold, intense expression with sharp features, wearing traditional Maratha armor adorned with intricate gold designs, standing against a dramatic sunset backdrop with a battlefield in the distance" \
  --report_to="wandb" \
  --push_to_hub
