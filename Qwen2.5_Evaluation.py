from transformers import AutoProcessor, AutoModelForVision2Seq
import torch, re, numpy as np
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

QWEN = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT = "What is the liquid volume in mL? Answer with only a number."
NUM = re.compile(r"-?\d+(?:\.\d+)?")

def extract(t):
    m = NUM.search(t or "")
    return float(m.group()) if m else None

processor = AutoProcessor.from_pretrained(QWEN)
model = AutoModelForVision2Seq.from_pretrained(QWEN, torch_dtype=torch.float16, device_map="auto")

dataset = load_dataset("csv", data_files={"test": "data/test.csv"})["test"]

preds, gts = [], []

for ex in tqdm(dataset):
    img = f"data/Images/{ex['image_name']}"
    inputs = processor(images=img, text=PROMPT, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs)
    pred = extract(processor.tokenizer.decode(out[0], skip_special_tokens=True))
    if pred is not None:
        preds.append(pred)
        gts.append(float(ex["volume_ml"]))

print("Qwen MAE :", mean_absolute_error(gts, preds))
print("Qwen RMSE:", mean_squared_error(gts, preds, squared=False))
print("Qwen R²  :", r2_score(gts, preds))
