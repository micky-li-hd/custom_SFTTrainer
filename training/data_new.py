import os
from dataclasses import dataclass
from typing import Dict, Sequence
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
from transformers import AutoProcessor
import glob
import transformers
from torch.nn.utils.rnn import pad_sequence
# ======== 数据处理逻辑 ========
def process_sample(sample):
    try:
        metadata = sample["json"]
        return {
            "caption": metadata.get("caption"),
            "cot": metadata.get("cot"),
            "aspect_ratio": metadata.get("aspect_ratio"),
            "img_index": metadata.get("img_index")
        }
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None

class LazySupervisedMixDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        processor: AutoProcessor,
    ):
        super(LazySupervisedMixDataset, self).__init__()

        list_data_dict = []
        data_files = glob.glob(os.path.join(data_path, "*.tar"))
        train_dataset = load_dataset("webdataset", data_files=data_files, split="train", num_proc=128)
        train_dataset = train_dataset.map(process_sample).filter(lambda x: x is not None)
        train_dataset = train_dataset.remove_columns([
            col for col in train_dataset.column_names 
            if col not in ["caption", "cot", "aspect_ratio", "img_index"]
        ])
        list_data_dict.append(train_dataset)
        if len(list_data_dict) > 1:
            list_data_dict = concatenate_datasets(list_data_dict)
        else:
            list_data_dict = list_data_dict[0]
        list_data_dict = list_data_dict.shuffle(seed=42)
        
        self.processor = processor
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        conversation = [
            {"role": "<|User|>", "content": sources['caption']},
            {"role": "<|Assistant|>", "content": f"{sources['cot']}<begin_of_image><end_of_image>"},
        ]
        system_prompt = "You are an assistant that creates images from descriptions. First, describe the image in detail, then generate it."
        prompt = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.processor.sft_format,
            system_prompt=system_prompt,
        )

        # Tokenize prompt
        text_ids = self.processor.tokenizer.encode(prompt)
        all_ids = text_ids[:-2] + sources['img_index'] + text_ids[-2:]
        all_ids = torch.LongTensor(all_ids)

        # 构建图像 token 的 mask
        all_image_ids_mask = torch.zeros(all_ids.shape, dtype=torch.bool)
        all_image_ids_mask[-len(sources['img_index'])-2:-2] = True

        # 找到 Assistant 回答开始的位置
        try:
            assistant_start_token_id = self.processor.tokenizer.encode("<|Assistant|>")[0]
            assistant_start_index = text_ids.index(assistant_start_token_id)
        except (ValueError, IndexError):
            assistant_start_index = 0

        assistant_ids_mask = torch.zeros(all_ids.shape, dtype=torch.bool)
        assistant_ids_mask[assistant_start_index:] = True

        # 构造输入和标签
        input_ids = all_ids[:-1]
        text_ids_mask = (all_image_ids_mask[:-1] == False)
        image_ids_mask = all_image_ids_mask[:-1]
        label_ids = all_ids[1:]
        label_text_ids_mask = assistant_ids_mask[1:] & (all_image_ids_mask[1:] == False)
        label_image_ids_mask = assistant_ids_mask[1:] & all_image_ids_mask[1:]

        return {
            "input_ids": input_ids,
            "label_ids": label_ids,
            "text_ids_mask": text_ids_mask,
            "image_ids_mask": image_ids_mask,
            "label_text_ids_mask": label_text_ids_mask,
            "label_image_ids_mask": label_image_ids_mask,
        }

# ======== Collator 实现 ========
@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 提取字段
        fields = [
            "input_ids", "text_ids_mask", "image_ids_mask",
            "label_ids", "label_text_ids_mask", "label_image_ids_mask"
        ]
        tensor_sequences = {key: [instance[key] for instance in instances] for key in fields}

        # Padding 处理
        padded_tensors = {
            "input_ids": pad_sequence(tensor_sequences["input_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "label_ids": pad_sequence(tensor_sequences["label_ids"], batch_first=True, padding_value=-100),  # IGNORE_INDEX
            "text_ids_mask": pad_sequence(tensor_sequences["text_ids_mask"], batch_first=True, padding_value=0),
            "image_ids_mask": pad_sequence(tensor_sequences["image_ids_mask"], batch_first=True, padding_value=0),
            "label_text_ids_mask": pad_sequence(tensor_sequences["label_text_ids_mask"], batch_first=True, padding_value=0),
            "label_image_ids_mask": pad_sequence(tensor_sequences["label_image_ids_mask"], batch_first=True, padding_value=0),
        }

        # 截断处理
        max_len = self.tokenizer.model_max_length
        for key in padded_tensors:
            if padded_tensors[key].shape[1] > max_len:
                print(f"Warning: {key} length {padded_tensors[key].shape[1]} exceeds max length {max_len}")
                padded_tensors[key] = padded_tensors[key][:, :max_len]

        # 构建 batch
        return {
            "input_ids": padded_tensors["input_ids"],
            "label_ids": padded_tensors["label_ids"],
            "attention_mask": (padded_tensors["input_ids"] != self.tokenizer.pad_token_id),
            "text_id_mask": padded_tensors["text_ids_mask"],
            "image_id_mask": padded_tensors["image_ids_mask"],
            "label_text_id_mask": padded_tensors["label_text_ids_mask"],
            "label_image_id_mask": padded_tensors["label_image_ids_mask"],
        }

# ======== 主程序入口 ========
if __name__ == "__main__":
    from janus.models.processing_vlm import VLChatProcessor
    
    # 初始化处理器和 tokenizer
    processor: VLChatProcessor = VLChatProcessor.from_pretrained("deepseek-ai/Janus-Pro-7B")
    tokenizer = processor.tokenizer
    padding_id = tokenizer.pad_token_id
    print(f"Padding ID: {padding_id}")

    # 创建数据集
    train_dataset = LazySupervisedMixDataset(
        data_path="/home/v-haodongli/Janus/tmp_script/laion_2b_aesthetic",
        processor=processor
    )

    # 创建 collator
    data_collator = DataCollatorForSupervisedDataset(tokenizer)

    # 测试调用
    batch = data_collator([train_dataset[1], train_dataset[2], train_dataset[3]])
    
    # 打印输出示例
    print("Batch keys:", batch.keys())
    print("Input IDs shape:", batch["input_ids"].shape)
    print("Label IDs shape:", batch["label_ids"].shape)
    print("Attention mask shape:", batch["attention_mask"].shape)