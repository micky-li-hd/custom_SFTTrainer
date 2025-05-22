# python train.py \
#     --model_name_or_path deepseek-ai/Janus-Pro-7B \
#     --url /home/v-haodongli/Janus/tmp_script/laion_2b_aesthetic/{00042..00133}.tar \
#     --max_samples 1000 \
#     --shuffle_buffer 1000 \
#     --per_device_train_batch_size 4 \
#     --output_dir ./output \
#     --num_train_epochs 3 \
#     --do_train


accelerate launch --config_file accelerate_config/default_config.yaml --main_process_port=8888 train.py config=config/sft.yaml
