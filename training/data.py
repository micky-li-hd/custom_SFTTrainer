import os
import json
import webdataset as wds
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from janus.models import VLChatProcessor
from braceexpand import braceexpand
import os

def expand_urls(url_pattern):
    expanded = list(braceexpand(url_pattern))
    urls = [os.path.expanduser(path) for path in expanded]
    urls.sort()
    return urls    

class WebDatasetFromTar:
    def __init__(
        self,
        urls: Union[str, List[str]],
        shuffle_buffer: int = 1000,
        caption_key: str = "caption",
        cot_key: str = "cot",
        aspect_ratio_key: str = "aspect_ratio",
        img_index_key: str = "img_index"
    ):
        if isinstance(urls, str):
            urls = [urls]

        self.urls = urls
        self.shuffle_buffer = shuffle_buffer
        self.caption_key = caption_key
        self.cot_key = cot_key
        self.aspect_ratio_key = aspect_ratio_key
        self.img_index_key = img_index_key

    def _process_sample(self, sample):
        try:
            metadata = json.loads(sample["json"].decode("utf-8"))
            caption = metadata[self.caption_key]
            cot = metadata[self.cot_key]
            aspect_ratio = metadata[self.aspect_ratio_key]
            img_index = metadata[self.img_index_key]
            return (caption, cot, aspect_ratio, img_index)
        except Exception as e:
            print(f"Error processing sample: {e}")
            return None

    def build(self):
        dataset = wds.WebDataset(self.urls).shuffle(self.shuffle_buffer)
        dataset = dataset.map(self._process_sample)
        # 过滤掉无效样本
        dataset = dataset.filter(lambda x: x is not None)
        return dataset

def process_one_data(
    sample: Tuple[Any, ...],
    vl_chat_processor: VLChatProcessor
) -> Optional[Tuple[torch.Tensor, ...]]:
    caption, cot, aspect_ratio, img_index = sample

    # 构建对话模板
    conversation = [
        {"role": "<|User|>", "content": caption},
        {"role": "<|Assistant|>", "content": f"{cot}<begin_of_image><end_of_image>"},
    ]
    system_prompt = "You are an assistant that creates images from descriptions. First, describe the image in detail, then generate it."
    prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt=system_prompt,
    )

    # tokenize prompt
    text_ids = vl_chat_processor.tokenizer.encode(prompt)
    all_ids = text_ids[:-2] + img_index + text_ids[-2:]
    all_ids = torch.LongTensor(all_ids)

    # 构建图像 token 的 mask
    all_image_ids_mask = torch.zeros(all_ids.shape, dtype=torch.bool)
    all_image_ids_mask[:] = False
    all_image_ids_mask[-len(img_index)-2:-2] = True

    # 找到 Assistant 回答开始的位置
    try:
        assistant_start_token_id = vl_chat_processor.tokenizer.encode("<|Assistant|>")[0]
        assistant_start_index = text_ids.index(assistant_start_token_id)
    except (ValueError, IndexError):
        assistant_start_index = 0

    assistant_ids_mask = torch.zeros(all_ids.shape, dtype=torch.bool)
    assistant_ids_mask[assistant_start_index:] = True

    # 构造输入和标签
    input_ids = all_ids[:-1]
    text_ids_mask = all_image_ids_mask[:-1] == False
    image_ids_mask = all_image_ids_mask[:-1]
    label_ids = all_ids[1:]
    label_text_ids_mask = assistant_ids_mask[1:] & (all_image_ids_mask[1:] == False)
    label_image_ids_mask = assistant_ids_mask[1:] & all_image_ids_mask[1:]

    return (
        input_ids,
        text_ids_mask,
        image_ids_mask,
        label_ids,
        label_text_ids_mask,
        label_image_ids_mask
    )


@dataclass
class DataCollatorForMultiModalDataset:
    """
    Collator function to batch samples.
    """

    def __call__(self, features):
        input_ids = pad_sequence([f[0] for f in features], batch_first=True)
        text_ids_mask = pad_sequence([f[1] for f in features], batch_first=True)
        image_ids_mask = pad_sequence([f[2] for f in features], batch_first=True)
        label_ids = pad_sequence([f[3] for f in features], batch_first=True)
        label_text_ids_mask = pad_sequence([f[4] for f in features], batch_first=True)
        label_image_ids_mask = pad_sequence([f[5] for f in features], batch_first=True)

        return {
            "input_ids": input_ids,
            "text_id_mask": text_ids_mask,
            "image_id_mask": image_ids_mask,
            "label_ids": label_ids,
            "label_text_id_mask": label_text_ids_mask,
            "label_image_id_mask": label_image_ids_mask,
        }