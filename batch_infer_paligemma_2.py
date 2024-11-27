from transformers import AutoProcessor, AutoModelForPreTraining
from PIL import Image
import requests
import torch
import pandas as pd
import json

df = pd.read_csv("dataset/test.csv")

f = open("prompts.json", "r")
prompts = json.load(f)
f.close()

model_id = "google/paligemma-3b-pt-224"

model = AutoModelForPreTraining.from_pretrained("google/paligemma-3b-pt-224").eval()

from peft import PeftConfig, PeftModel

# Load the adapter configuration
peft_config = PeftConfig.from_pretrained("paligemma_ft")

# Attach the LoRA adapter to the base model
model = PeftModel.from_pretrained(model, "paligemma_ft", peft_config=peft_config)

processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

num_examples = len(df)

entity_names = df["entity_name"].to_list()
for i in range(num_examples):
    f = open("outputs.txt", "a")
    try:
        prompt = prompts[entity_names[i]]
        image = Image.open(f"dataset/test_images/{i+1}.jpg").convert("RGB")
        model_inputs = processor(text=prompt, images=image, return_tensors="pt")
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**model_inputs, max_new_tokens=20, do_sample=False)
            generation = generation[0][input_len:]
            decoded = processor.decode(generation, skip_special_tokens=True)
    except FileNotFoundError:
        decoded = ""
    f.write(decoded+"\n")
    f.close()

    