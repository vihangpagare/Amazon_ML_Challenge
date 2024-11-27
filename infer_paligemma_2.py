from transformers import AutoProcessor, AutoModelForPreTraining
from PIL import Image
import requests
import torch

model_id = "google/paligemma-3b-pt-224"

url = "https://m.media-amazon.com/images/I/21ztnqtFNVL.jpg"
image = Image.open(requests.get(url, stream=True).raw)

model = AutoModelForPreTraining.from_pretrained("google/paligemma-3b-pt-224").eval()

from peft import PeftConfig, PeftModel

# Load the adapter configuration
peft_config = PeftConfig.from_pretrained("paligemma_ft")

# Attach the LoRA adapter to the base model
model = PeftModel.from_pretrained(model, "paligemma_ft", peft_config=peft_config)

processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

prompt = "What is the height of the item?"
model_inputs = processor(text=prompt, images=image, return_tensors="pt")
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)
