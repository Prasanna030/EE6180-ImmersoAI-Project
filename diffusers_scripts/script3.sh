export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export INSTANCE_DIR="prasanna30/bajirao_new4k"
export OUTPUT_DIR="trained-sd3-lora"

accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="An empty, dimly lit Indian ancient hall with columns and arches with the lead characters from the movie Bajirao Mastani ,Bajirao (Ranveer Singh) and Mastani (Deepika Padukone) facing each other" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=4e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="Ranveer Singh from Bajirao Mastani" \
  --validation_epochs=25 \
  --seed="0" 
  --push_to_hub
