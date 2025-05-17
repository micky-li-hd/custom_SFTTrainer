# train.py
import os
import sys
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import HfArgumentParser, TrainingArguments as HFTrainingArguments
from Trainer.argument import ModelDataArguments
from Trainer.data import expand_urls, WebDatasetFromTar, MultiModalDataset, DataCollatorForMultiModalDataset
from Trainer.trainer import CustomTrainer
from janus.models.processing_vlm import VLChatProcessor
from janus.models.modeling_vlm import MultiModalityCausalLM
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm import tqdm
import wandb

# Setup logger
logger = get_logger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)

def main():
    parser = HfArgumentParser((HFTrainingArguments, ModelDataArguments))
    training_args, modeldata_args = parser.parse_args_into_dataclasses()

    # 初始化 Accelerator
    accelerator = Accelerator(
        mixed_precision='bf16' if training_args.bf16 else 'fp16' if training_args.fp16 else None,
        log_with="wandb"
    )

    # 初始化 WandB
    if accelerator.is_main_process:
        wandb.init(project="cot-data", config=training_args.to_dict())
        wandb.run.name = f"shard_{modeldata_args.url.split('/')[-1]}"
        wandb.watch_called = True

    # Model
    processor: VLChatProcessor = VLChatProcessor.from_pretrained(modeldata_args.model_name_or_path)
    model: MultiModalityCausalLM = MultiModalityCausalLM.from_pretrained(
        modeldata_args.model_name_or_path,
        trust_remote_code=True
    ).to(accelerator.device)

    # Data
    expanded_urls = expand_urls(modeldata_args.url)
    web_dataset = WebDatasetFromTar(expanded_urls, shuffle_buffer=training_args.shuffle_buffer).build()
    dataset = MultiModalDataset(web_dataset, processor, max_samples=training_args.max_samples)
    collator = DataCollatorForMultiModalDataset()

    # Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
    )

    # Prepare with Accelerator
    trainer.model = accelerator.prepare(trainer.model)

    # Train
    logger.info("***** 开始训练 *****")
    trainer.train()

    # Save final model
    logger.info("训练完成，保存最终模型")
    trainer.save_model(training_args.output_dir)

    wandb.finish()

if __name__ == "__main__":
    main()