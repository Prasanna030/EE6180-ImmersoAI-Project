export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="sddata"
export HUB_MODEL_ID="bajiroav1-5"
export DATASET_NAME="prasanna30/bajirao_new4k"

accelerate launch --num_processes=1 --num_machines=1 --dynamo_backend='no' --main_process_port 0 --mixed_precision="no" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --checkpointing_steps=1000 \
  --validation_prompt="Ranveer Singh from Bajirao Mastani" \
  --seed=1337
