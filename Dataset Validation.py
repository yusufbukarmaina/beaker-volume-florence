import os, pandas as pd

BASE = "data"
IMG_DIR = os.path.join(BASE, "Images")
CSV = os.path.join(BASE, "annotations_clean.csv")

df = pd.read_csv(CSV)
df["image_path"] = df["image_name"].apply(lambda x: os.path.join(IMG_DIR, str(x)))

missing = df[~df["image_path"].apply(os.path.exists)]
print("Missing images:", len(missing))

assert len(missing) == 0, "❌ Dataset misalignment detected"
print("✅ Dataset validated — safe to train")
