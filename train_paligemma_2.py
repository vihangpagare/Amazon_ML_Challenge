from datasets import Dataset
import json
import os
import pandas as pd
from PIL import Image

f = open("prompts.json", "r")
prompts = json.load(f)
f.close()

dir_path = "dataset/images"
model_id = "google/paligemma-3b-pt-224"
batch_size = 4

image_paths = [os.path.join(dir_path, f"{idx}.jpg") for idx in range(20000)]

df = pd.read_csv("dataset/train.csv")[:20000]

input_texts = [prompts[entity_name] for entity_name in df['entity_name'].tolist()]
output_texts = df['entity_value'].tolist()

def get_example(idx):
    return {"image":Image.open(image_paths[idx]).convert("RGB"), "input_text":input_texts[idx], "output_text":output_texts[idx]}

def example_generator():
    for idx in range(2000):
        try:
            example = get_example(idx)
            yield(example)
        except FileNotFoundError:
            continue

dataset = Dataset.from_generator(example_generator, cache_dir="/home/aaradhye/scratch/datasets")

from transformers import PaliGemmaProcessor
model_id = "google/paligemma-3b-pt-224"
processor = PaliGemmaProcessor.from_pretrained(model_id)

import torch
device = "cuda"

image_token = processor.tokenizer.convert_tokens_to_ids("<image>")

def collate_fn(examples):
    texts = [example["input_text"] for example in examples]
    labels= [example["output_text"] for example in examples]
    images = [example["image"] for example in examples]
    tokens = processor(text=texts, images=images, suffix=labels,
                        return_tensors="pt", padding="longest",
                        tokenize_newline_separately=False)

    tokens = tokens.to(torch.float32).to(device)
    return tokens

from transformers import PaliGemmaForConditionalGeneration
import torch

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
    param.requires_grad = False

from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

from transformers import TrainingArguments
args=TrainingArguments(
            num_train_epochs=1,
            remove_unused_columns=False,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            warmup_steps=2,
            learning_rate=2e-5,
            weight_decay=1e-6,
            adam_beta2=0.999,
            logging_steps=10,
            optim="adamw_hf",
            save_strategy="no",
            save_steps=1000,
            push_to_hub=False,
            save_total_limit=1,
            output_dir="paligemma_ft",
            bf16=False,
            report_to=[],
            dataloader_pin_memory=False
        )

from transformers import Trainer

trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=collate_fn,
        args=args
        )

trainer.train()

trainer.save_model("paligemma_ft")

