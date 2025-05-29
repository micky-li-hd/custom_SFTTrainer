import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union
from functools import partial
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from training.utils import get_config, flatten_omega_conf, AverageMeter
from training.my_logging import set_verbosity_info, set_verbosity_error
from janus.models.processing_vlm import VLChatProcessor
from janus.models.modeling_vlm import MultiModalityCausalLM
from training.data_new import LazySupervisedMixDataset, DataCollatorForSupervisedDataset
from torch.utils.data import Dataset, DataLoader
from training.lr_schedulers import get_scheduler
from deepspeed.accelerator import get_accelerator
# import debugpy
# if int(os.getenv("LOCAL_RANK", "0")) == 3:
#     try:
#         debugpy.listen(("localhost", 9505))
#         print("Waiting for debugger attach (only on rank 0)")
#         debugpy.wait_for_client()
#     except Exception as e:
#         print(f"Debugpy failed to start: {e}")
logger = get_logger(__name__, log_level="INFO")

def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    total_batch_size_per_gpu = config.training.batch_size
    total_batch_size = config.training.batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )    

        if accelerator.is_main_process:
            os.makedirs(config.experiment.output_dir, exist_ok=True)
            config_path = Path(config.experiment.output_dir) / "config.yaml"
            logging.info(f"Saving config to {config_path}")
            OmegaConf.save(config, config_path)

        # If passed along, set the training seed now.
        if config.training.seed is not None:
            set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")
    processor: VLChatProcessor = VLChatProcessor.from_pretrained(config.model.janus_pro.model_name_or_path)
    model: MultiModalityCausalLM = MultiModalityCausalLM.from_pretrained(
        config.model.janus_pro.model_name_or_path,
        trust_remote_code=True)
    
    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    #唯一的很不会写的一块，周末好好看看optimizer/LR scheduler
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")
    # 修改后的构建数据集方式
    train_dataset = LazySupervisedMixDataset(
        data_path=config.dataset.params.path,
        processor=processor
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        collate_fn=DataCollatorForSupervisedDataset(processor.tokenizer),
        num_workers=config.dataset.params.num_workers,
        pin_memory=True
    )
    #计算每个epoch走多少step,手动定义一个epoch用config.experiment.samples_per_epoch条数据
    batches_per_epoch = config.training.samples_per_epoch // config.training.batch_size
    num_update_steps_per_epoch = math.ceil(batches_per_epoch / config.training.gradient_accumulation_steps)
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)


    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0

    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        if path is not None:
            path = os.path.join(config.experiment.output_dir, path)

            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

            accelerator.print(f"Resuming from checkpoint {path}/unwrapped_model/pytorch_model.bin")
            state_dict = torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
            del state_dict

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler, dataloader = accelerator.prepare(model, optimizer, lr_scheduler, dataloader)
    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch in dataloader:
            # if global_step + 1 == 2:
            #     rank = int(os.environ.get('RANK', 0))
            #     if rank == 3:  # 只有 rank=3 才保存
            #         save_path = f"debug_batch_step_{global_step + 1}_rank_{rank}.pt"
            #         torch.save({
            #             'batch': batch,
            #             'global_step': global_step,
            #             'input_ids': batch["input_ids"],
            #             'text_id_mask': batch["text_id_mask"],
            #             'image_id_mask': batch["image_id_mask"],
            #             'label_ids': batch["label_ids"],
            #             'label_text_id_mask': batch["label_text_id_mask"],
            #             'label_image_id_mask': batch["label_image_id_mask"],
            #         }, save_path)
            #         print(f"Rank {rank} saved batch to {save_path}")
            #         # 保存当前 batch 到文件
            #         save_path = f"debug_batch_step_{global_step + 1}.pt"
            #         torch.save({
            #             'batch': batch,
            #             'global_step': global_step,
            #             'input_ids': batch["input_ids"],
            #             'text_id_mask': batch["text_id_mask"],
            #             'image_id_mask': batch["image_id_mask"],
            #             'label_ids': batch["label_ids"],
            #             'label_text_id_mask': batch["label_text_id_mask"],
            #             'label_image_id_mask': batch["label_image_id_mask"],
            #         }, save_path)
            #         import pdb; pdb.set_trace()
            #         print(f"Saved batch data to {save_path}")

            #     global_step += 1
            #     continue
            input_ids = batch["input_ids"]                # (B, L)
            text_id_mask = batch["text_id_mask"]          # (B, L)
            image_id_mask = batch["image_id_mask"]        # (B, L)
            label_ids = batch["label_ids"]                # (B, L)
            label_text_id_mask = batch["label_text_id_mask"]  # (B, L)
            label_image_id_mask = batch["label_image_id_mask"]  # (B, L)

            data_time_m.update(time.time() - end)
            batch_size, seq_len = input_ids.shape
            # embed_dim = model.language_model.model.embed_tokens.weight.shape[1]

            if global_step == 0 and epoch == 0:
                logger.info("Input ids: {}".format(input_ids))
                logger.info("Labels: {}".format(label_ids))

            # 获取嵌入
            with torch.no_grad():
                # 提取文本 token 的 input_ids
                text_indices = text_id_mask.bool()
                text_input_ids = input_ids[text_indices]  # (N_text, )
                text_embeds = model.language_model.model.embed_tokens(text_input_ids)  # (N_text, D)
                embed_dim = text_embeds.shape[-1]

                # 提取图像 token 的 input_ids
                image_indices = image_id_mask.bool()
                image_input_ids = input_ids[image_indices]  # (N_image, )
                image_embeds = model.prepare_gen_img_embeds(image_input_ids)  # (N_image, D)

            input_embeds = torch.zeros(
                (batch_size, seq_len, embed_dim),
                dtype=text_embeds.dtype,
                device=text_embeds.device
            )
            # 填充嵌入
            input_embeds[text_indices] = text_embeds
            input_embeds[image_indices] = image_embeds

            # 前向传播
            with accelerator.accumulate(model):
                outputs = model(input_embeds, label_ids, label_text_id_mask, label_image_id_mask)
                loss = outputs["loss"]
                loss_text = outputs["loss_text"]
                loss_image = outputs["loss_image"]
                # del outputs, input_embeds
                # 分布式训练中的 loss 同步
                avg_loss_text = accelerator.gather(loss_text.repeat(config.training.batch_size)).mean()
                avg_loss_image = accelerator.gather(loss_image.repeat(config.training.batch_size)).mean()
                avg_loss = avg_loss_text + avg_loss_image
                
                accelerator.backward(loss)

                # 梯度裁剪
                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()

                # 日志记录
                if accelerator.sync_gradients and (global_step + 1) % config.experiment.log_grad_norm_every == 0 and accelerator.is_main_process:
                    log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)                    
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                            config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    logs = {
                        "step_loss_text": avg_loss_text.item(),
                        "step_loss_image": avg_loss_image.item(),
                        "step_total_loss": avg_loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss_text: {avg_loss_text.item():0.4f} "
                        f"Loss_image: {avg_loss_image.item():0.4f} "
                        f"Total_Loss: {avg_loss.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, processor,config, accelerator, global_step + 1)

                global_step += 1

            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                break
            # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, processor, config, accelerator, global_step)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=False)

    accelerator.end_training()

def decode_to_pil(vq_list, vl_gpt, shape=(1, 8, 24, 24)):
    # 将列表转为张量并移动到GPU
    vq_tensor = torch.tensor(vq_list, dtype=torch.int, device="cuda")
    
    # 解码图像数据（假设vl_gpt已加载）
    with torch.no_grad():
        dec = vl_gpt.gen_vision_model.decode_code(vq_tensor, shape=shape)
    
    # 后处理：张量转图像
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(dec[0])


@torch.no_grad()
def run_eval_and_log(model, processor, config, step):
    model.eval()
    try:
        conversation = [
            {
                "role": "<|User|>",
                "content": "A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair",
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + processor.image_start_tag

        images = model.generate(
            model,
            processor,
            prompt=prompt,
            parallel_size=4,
            img_size=384,
            patch_size=16,
        )

        # 拼接图片并上传到 wandb
        table_data = []
        for idx, image in enumerate(images):
            caption = f"Step {step} - Image {idx}"
            wandb_image = wandb.Image(image, caption=caption)
            table_data.append([caption, wandb_image])

        table = wandb.Table(data=table_data, columns=["Description", "Generated Image"])
        wandb.log({"Eval Images": table}, step=step)

    except Exception as e:
        print(f"[ERROR] Failed to run eval at step {step}: {e}")
    finally:
        model.train()


def save_checkpoint(model, processor, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    # XXX: could also make this conditional on deepspeed
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")
        # Run eval and log to wandb
        run_eval_and_log(unwrapped_model, processor, config, global_step)

def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()




