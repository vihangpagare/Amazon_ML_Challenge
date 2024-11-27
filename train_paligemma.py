from transformers import AutoProcessor, AutoModelForPreTraining
import torch
import pandas as pd
import json
import os
import cv2
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image

f = open("prompts.json", "r")
prompts = json.load(f)
f.close()

dir_path = "dataset/images"
model_id = "google/paligemma-3b-pt-224"
batch_size = 4
image_names = [name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))]
num_train_samples = len(image_names)

df = pd.read_csv("dataset/train.csv")[:num_train_samples]

def load_image(image_name):
    return cv2.imread(os.path.join(dir_path, image_name))

# Initialize the process group
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# # Load images and texts
input_texts = [prompts[entity_name] for entity_name in df['entity_name'].tolist()]
output_texts = df['entity_value'].tolist()

from transformers import AdamW, get_scheduler
from torch.nn import CrossEntropyLoss

# Create a dataset and dataloader
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_names, input_texts, output_texts, processor):
        self.image_names = image_names
        self.input_texts = input_texts
        self.output_texts = output_texts
        self.processor = processor
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        # Tokenize inputs and outputs using the processor
        image = Image.open(os.path.join(dir_path, self.image_names[idx])).convert("RGB")
        input_text = self.input_texts[idx]
        output_text = self.output_texts[idx]
        inputs = self.processor(text=input_text, images=image, return_tensors="pt")
        outputs = self.processor(text=output_text, return_tensors="pt")["input_ids"]
        return inputs, outputs

def train(rank, world_size):
    setup(rank, world_size)
    
    # Load the model and processor
    model = AutoModelForPreTraining.from_pretrained(model_id).train().to(rank)
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Wrap model in DistributedDataParallel (DDP)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Load dataset into DataLoader
    dataset = CustomDataset(image_names, input_texts, output_texts, processor)
    
    # Use DistributedSampler to split dataset across GPUs
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Define an optimizer and a scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * num_epochs
    )
    # Define a loss function
    loss_fn = CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for idx, batch in enumerate(dataloader):
            inputs, labels = batch
            
            # Move inputs and labels to the same device as the model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            labels = labels.to(model.device)
            
            # Forward pass
            outputs = model(**inputs)
            
            # Compute loss (compare output logits with tokenized output_texts)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            loss = loss_fn(logits.view(-1, model.config.vocab_size), labels.view(-1))
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if (idx % 100 == 0 and rank == 0):
            print(idx, loss.item())
        print(f"Epoch {epoch + 1} finished with loss: {loss.item()}")

    model.save_pretrained("paligemma_model_ft.pth")
    processor.save_pretrained("paligemma_processor_ft.pth")

    cleanup()

# Entry point for spawning processes
def main():
    world_size = torch.cuda.device_count()  # Number of available GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()