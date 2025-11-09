# trainPart1_fixed.py

cache_dir = input("Please enter the location of cache dir:")

from datasets import load_dataset
dataset_location = input("Please enter the location of training dataset:")

# Debug: confirm the dataset path before loading
print("Loading dataset from:", dataset_location)

train_dataset = load_dataset("json", data_files=dataset_location, split="train[:88%]")
val_dataset = load_dataset("json", data_files=dataset_location, split="train[88%:]")  # FIXED: validation slice

print("Training sets: {} - Validating set: {}".format(len(train_dataset), len(val_dataset)))

import requests
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def _open_image(self, imgPath):
        # if URL -> requests, else local path
        if isinstance(imgPath, str) and imgPath.startswith("http"):
            img = Image.open(requests.get(imgPath, stream=True).raw).convert('RGB')
        else:
            img = Image.open(imgPath).convert('RGB')
        return img

    def __getitem__(self, idx):
        item = self.dataset[idx]
        imgPath = item["imgPath"]
        raw_image = self._open_image(imgPath)

        question = "Question:" + item.get("question", "") + " Short answer:"
        encoding = self.processor(images=raw_image, text=question, padding="max_length", max_length=100, return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}

        # Keep labels as raw text -- collate_fn will tokenize
        encoding["labels"] = item.get("answer", "") + ". Explanation: " + item.get("explanation", "")
        return encoding

def collate_fn(batch):
    processed_batch = {}
    for key in batch[0].keys():
        if key != "labels":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer([example["labels"] for example in batch], padding=True, return_tensors="pt")
            processed_batch["labels"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch

from transformers import AutoProcessor, Blip2ForConditionalGeneration
processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir = cache_dir)
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir = cache_dir, load_in_8bit=True, device_map="auto")

# PEFT: be compatible with versions where prepare_model_for_int8_training was removed
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
    prepare_fn = prepare_model_for_int8_training
except Exception:
    from peft import LoraConfig, get_peft_model
    try:
        from peft import prepare_model_for_kbit_training
        prepare_fn = prepare_model_for_kbit_training
    except Exception:
        prepare_fn = None
        print("Warning: no prepare_model_for_* function found in this PEFT version. Continue without it.")

lora_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="none",
)

if prepare_fn is not None:
    model = prepare_fn(model)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=5e-5)

train_dataset = ImageCaptioningDataset(train_dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=collate_fn)

valid_dataset = ImageCaptioningDataset(val_dataset, processor)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=2, collate_fn=collate_fn)

device = torch.device(device)
model.to(device)

# training loop remains similar...
