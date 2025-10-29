import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from ultralytics import YOLO
from transformers import BlipForConditionalGeneration, BlipProcessor
from sklearn.cluster import KMeans
import csv
from collections import defaultdict

# === Paths ===
DATA = Path('data/flickr30k')
IMG_DIR = DATA / 'images'
CAPS_PATH = DATA / 'captions.json'   # can be JSON or CSV
OUT = Path('data/features'); OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 🧩 Helper: Load captions (auto-detect CSV or JSON)
# ============================================================
def load_captions(path: Path):
    import re
    from collections import defaultdict

    if not path.exists():
        raise FileNotFoundError(f"❌ Captions file not found: {path}")

    # Check format type
    first_line = open(path, encoding="utf-8").read(100).strip()
    if first_line.startswith("{"):
        print("📘 Detected JSON captions file")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    print("📗 Detected CSV captions file — cleaning and fixing filenames")
    captions = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split only on the first comma
            if "," in line:
                image_part, caption_part = line.split(",", 1)
            else:
                continue

            image_id = image_part.strip()
            caption = caption_part.strip()

            # 🚨 Clean image filename (remove any caption fragments)
            image_id = re.split(r'(\.jpg|\.jpeg|\.png)', image_id, flags=re.IGNORECASE)[0] + ".jpg"
            image_id = image_id.strip()

            # 🚨 Remove extra quotes, dots, commas from caption
            caption = re.sub(r'["“”]', '', caption).strip(" .,")

            if image_id and caption:
                captions[image_id].append(caption)

    print(f"✅ Loaded {len(captions)} clean image entries")
    # Print first 5 samples for sanity check
    for k in list(captions.keys())[:5]:
        print(f"  • {k} → {captions[k][0]}")
    return captions



# ============================================================
# 🖼️ Helper: Extract dominant colors
# ============================================================
def kmeans_colors(im, k=5):
    arr = np.array(im.resize((128, 128))).reshape(-1, 3).astype(np.float32)
    km = KMeans(n_clusters=k, n_init='auto').fit(arr)
    centers = km.cluster_centers_.clip(0, 255).astype(np.uint8)
    return centers.tolist()


# ============================================================
# 🧠 Main Feature Extraction
# ============================================================
def main():
    caps = load_captions(CAPS_PATH)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🧠 Using device: {device}")

    # Models
    yolo = YOLO('yolov8n.pt')  # object detection
    blip = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to(device).eval()
    blip_proc = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')

    rows = []

    for i, img_id in enumerate(sorted(caps.keys()), 1):
        img_path = IMG_DIR / img_id
        if not img_path.exists():
            print(f"[⚠️ Missing image] {img_id}")
            continue

        try:
            im = Image.open(img_path).convert('RGB')

            # --- YOLOv8 object detection ---
            y = yolo.predict(source=np.array(im), verbose=False, conf=0.25)[0]
            objs = []
            for b in y.boxes:
                cls = int(b.cls.tolist()[0])
                conf = float(b.conf.tolist()[0])
                xyxy = [float(x) for x in b.xyxy.tolist()[0]]
                objs.append({"cls": cls, "conf": conf, "box": xyxy})

            # --- Dominant color palette ---
            colors = kmeans_colors(im, k=5)

            # --- Generate BLIP caption ---
            inp = blip_proc(images=im, return_tensors='pt').to(device)
            with torch.no_grad():
                out = blip.generate(**inp, max_new_tokens=20)
            gen_caption = blip_proc.decode(out[0], skip_special_tokens=True)

            # --- Append results ---
            rows.append({
                "image_id": img_id,
                "objects": objs,
                "colors": colors,
                "gen_caption": gen_caption,
                "flickr_captions": caps[img_id]
            })

            if i % 50 == 0:
                print(f"✅ Processed {i}/{len(caps)} images")

        except Exception as e:
            print(f"[❌ Error on {img_id}] {e}")

    # --- Save output ---
    df = pd.DataFrame(rows)
    if 'image_id' not in df.columns:
        df['image_id'] = [r.get('image_id', None) for r in rows]

    print(f"💾 Writing {len(df)} entries to features.parquet …")
    df.to_parquet(OUT / 'features.parquet', index=False)
    print("✅ Done! Saved to data/features/features.parquet")

# ============================================================

if __name__ == "__main__":
    main()
