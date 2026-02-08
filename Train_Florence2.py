import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    default_data_collator
)

DATA_DIR = "data"
QUESTION = "What is the liquid volume in mL?"

dataset = load_dataset(
    "csv",
    data_files={
        "train": f"{DATA_DIR}/train.csv",
        "validation": f"{DATA_DIR}/val.csv"
    }
)

def add_image_path(ex):
    ex["image"] = f"{DATA_DIR}/Images/{ex['image_name']}"
    return ex

dataset = dataset.map(add_image_path)

processor = AutoProcessor.from_pretrained("microsoft/florence-2-base")
model = AutoModelForVision2Seq.from_pretrained(
    "microsoft/florence-2-base",
    torch_dtype=torch.float16
).cuda()

def preprocess(ex):
    inputs = processor(images=ex["image"], text=QUESTION, return_tensors="pt")
    labels = processor.tokenizer(ex["volume_label"], return_tensors="pt").input_ids
    inputs["labels"] = labels[0]
    return {k: v.squeeze(0) for k, v in inputs.items()}

train_ds = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)
val_ds   = dataset["validation"].map(preprocess, remove_columns=dataset["validation"].column_names)

args = TrainingArguments(
    output_dir="checkpoints/florence2",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    num_train_epochs=5,
    fp16=True,
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=50,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=default_data_collator
)

trainer.train()
