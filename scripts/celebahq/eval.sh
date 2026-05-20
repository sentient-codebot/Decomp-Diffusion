CUDA_VISIBLE_DEVICES=0 uv run accelerate launch --num_processes=1 eval.py \
--mixed_precision fp16 \
--seed 42 \
--batch_size 32 --num_validation_images 32 \
--output_dir results/celebahq/gen_images \
--scheduler_config configs/celebahq/scheduler/scheduler_config.json --dataset_root data/celebahq_data128x128/ \
--dataset_glob '**/*.jpg' --resolution 128 --vit_input_resolution 128 \
--ckpt_path results/celebahq/checkpoint-175000