import re, torch, numpy as np
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

def extract(txt):
    m = NUMBER_RE.search(txt or "")
    return float(m.group()) if m else None

processor = AutoProcessor.from_pretrained("microsoft/florence-2-base")
model = AutoModelForVision2Seq.from_pretrained(
    "checkpoints/florence2",
    torch_dtype=torch.float16
).cuda()

dataset = load_dataset("csv", data_files={"test": "data/test.csv"})["test"]

preds, gts = [], []

for ex in tqdm(dataset):
    img = f"data/Images/{ex['image_name']}"
    inputs = processor(images=img, text="What is the liquid volume in mL?", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs)
    pred = extract(processor.tokenizer.decode(out[0], skip_special_tokens=True))
    if pred is not None:
        preds.append(pred)
        gts.append(float(ex["volume_ml"]))

print("MAE :", mean_absolute_error(gts, preds))
print("RMSE:", mean_squared_error(gts, preds, squared=False))
print("R²  :", r2_score(gts, preds))
