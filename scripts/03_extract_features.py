import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from ultralytics import YOLO
from transformers import BlipForConditionalGeneration, BlipProcessor
from sklearn.cluster import KMeans
from collections import defaultdict
from typing import List, Optional

# === Paths ===
DATA = Path('data/flickr30k')
IMG_DIR = DATA / 'images'
CAPS_PATH = DATA / 'captions.json'   # can be JSON or CSV
OUT = Path('data/features'); OUT.mkdir(parents=True, exist_ok=True)

BLIP_MAX_NEW_TOKENS = 20
BLIP_BATCH_DEFAULT = 4
YOLO_CONF_DEFAULT = 0.25
YOLO_IMGSZ = 640


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
def main(
    limit: Optional[int] = None,
    blip_batch: int = BLIP_BATCH_DEFAULT,
    skip_yolo: bool = False,
    conf_threshold: float = YOLO_CONF_DEFAULT,
):
    caps = load_captions(CAPS_PATH)

    image_ids = sorted(caps.keys())
    if limit is not None:
        image_ids = image_ids[:limit]

    if not image_ids:
        print("❌ No images to process. Check your captions or image directory.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🧠 Using device: {device}")

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Models
    yolo_model = None
    if not skip_yolo:
        yolo_model = YOLO('yolov8n.pt')
        yolo_model.to(device)

    model_dtype = torch.float16 if device == 'cuda' else torch.float32
    blip = BlipForConditionalGeneration.from_pretrained(
        'Salesforce/blip-image-captioning-base',
        torch_dtype=model_dtype
    ).to(device).eval()
    blip_proc = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')

    rows: List[dict] = []
    pending_blip_imgs: List[Image.Image] = []
    pending_indices: List[int] = []
    total = len(image_ids)
    blip_batch = max(1, blip_batch)

    def flush_blip_batch():
        if not pending_blip_imgs:
            return
        inputs = blip_proc(images=pending_blip_imgs, return_tensors='pt', padding=True).to(device)
        if device == 'cuda' and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            outputs = blip.generate(**inputs, max_new_tokens=BLIP_MAX_NEW_TOKENS)
        captions = blip_proc.batch_decode(outputs, skip_special_tokens=True)
        for row_idx, caption in zip(pending_indices, captions):
            rows[row_idx]["gen_caption"] = caption
        pending_blip_imgs.clear()
        pending_indices.clear()

    for i, img_id in enumerate(image_ids, 1):
        img_path = IMG_DIR / img_id
        if not img_path.exists():
            print(f"[⚠️ Missing image] {img_id}")
            continue

        try:
            pil_img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[❌ Error opening {img_id}] {e}")
            continue

        try:
            img_np = np.array(pil_img)

            objs = []
            if yolo_model is not None:
                yolo_out = yolo_model.predict(
                    source=img_np,
                    device=device,
                    verbose=False,
                    conf=conf_threshold,
                    imgsz=YOLO_IMGSZ
                )[0]
                for b in yolo_out.boxes:
                    cls = int(b.cls.tolist()[0])
                    conf = float(b.conf.tolist()[0])
                    xyxy = [float(x) for x in b.xyxy.tolist()[0]]
                    objs.append({"cls": cls, "conf": conf, "box": xyxy})

            colors = kmeans_colors(pil_img, k=5)

            row = {
                "image_id": img_id,
                "objects": objs,
                "colors": colors,
                "gen_caption": "",
                "flickr_captions": caps.get(img_id, []),
            }
            rows.append(row)

            pending_indices.append(len(rows) - 1)
            pending_blip_imgs.append(pil_img.copy())

            if len(pending_blip_imgs) >= blip_batch:
                flush_blip_batch()

            if i % 50 == 0 or i == total:
                print(f"✅ Processed {i}/{total} images")

        except Exception as e:
            print(f"[❌ Error on {img_id}] {e}")

    flush_blip_batch()

    if not rows:
        print("⚠️ No rows generated; nothing to save.")
        return

    df = pd.DataFrame(rows)
    if 'image_id' not in df.columns:
        df['image_id'] = [r.get('image_id', None) for r in rows]

    print(f"💾 Writing {len(df)} entries to features.parquet …")
    df.to_parquet(OUT / 'features.parquet', index=False)
    print("✅ Done! Saved to data/features/features.parquet")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Flickr30k auxiliary features.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N images.")
    parser.add_argument("--skip-yolo", action="store_true", help="Skip YOLO detection for faster runs.")
    parser.add_argument("--blip-batch", type=int, default=BLIP_BATCH_DEFAULT, help="Batch size for BLIP captioning.")
    parser.add_argument("--conf", type=float, default=YOLO_CONF_DEFAULT, help="YOLO confidence threshold.")
    return parser.parse_args()


# ============================================================

if __name__ == "__main__":
    args = parse_args()
    main(
        limit=args.limit,
        blip_batch=args.blip_batch,
        skip_yolo=args.skip_yolo,
        conf_threshold=args.conf,
    )
