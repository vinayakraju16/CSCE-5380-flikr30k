import os, json, csv, re
from pathlib import Path
import numpy as np
import faiss
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict
from tqdm import tqdm

# === Paths ===
DATA_DIR = Path("data/flickr30k")
IMG_DIR = DATA_DIR / "images"
CAP_PATH = DATA_DIR / "captions.csv"

EMB_DIR = Path("data/embeddings"); EMB_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR = Path("data/index"); INDEX_DIR.mkdir(parents=True, exist_ok=True)

IMG_EMB_PATH = EMB_DIR / "img.npy"
ID_LIST_PATH = EMB_DIR / "img_ids.txt"
INDEX_PATH = INDEX_DIR / "faiss.index"

# === Model and Settings ===
MODEL_ID = os.environ.get("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
BATCH_SIZE = 32       # can increase to 64 if GPU memory allows
ALPHA = 0.5           # balance between image (0.5) and text (0.5) embeddings


# ============================================================
# 🧩 Helper: Load captions and clean image IDs
# ============================================================
def load_captions_auto(path):
    if not path.exists():
        raise FileNotFoundError(f"❌ Captions file not found: {path}")

    first_line = open(path, encoding="utf-8").read(100).strip()

    # JSON format
    if first_line.startswith("{"):
        print("📘 Detected JSON captions file")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # CSV format
    print("📗 Detected CSV captions file — cleaning image names")
    caps = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 2:
                continue
            img, cap = row[0].strip(), ",".join(row[1:]).strip()
            img = re.split(r"(\.jpg|\.jpeg|\.png)", img, flags=re.I)[0] + ".jpg"
            img = img.strip()
            cap = cap.strip(" .,\"“”")
            if img and cap:
                caps[img].append(cap)
    print(f"✅ Loaded {len(caps)} cleaned image entries")
    for k in list(caps.keys())[:5]:
        print(f"  • {k} → {caps[k][0]}")
    return caps


# ============================================================
# 🧠 Build CLIP Index (GPU + batching + alignment)
# ============================================================
def main():
    caps = load_captions_auto(CAP_PATH)
    img_ids = sorted(list(caps.keys()))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Loading CLIP model: {MODEL_ID} on {device}")
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    # ---------------------------------------------------------
    # Encode images in batches
    # ---------------------------------------------------------
    def process_batch(img_batch):
        inputs = processor(images=img_batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")

    vecs, batch = [], []
    print(f"🖼️ Encoding {len(img_ids)} images in batches of {BATCH_SIZE}...")
    for img_id in tqdm(img_ids, desc="Image Encoding", unit="img"):
        path = IMG_DIR / img_id
        if not path.exists():
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue

        batch.append(img)
        if len(batch) == BATCH_SIZE:
            feats = process_batch(batch)
            vecs.extend(feats)
            batch = []

    if batch:
        feats = process_batch(batch)
        vecs.extend(feats)

    img_vecs = np.stack(vecs).astype("float32")

    # ---------------------------------------------------------
    # Encode captions for each image
    # ---------------------------------------------------------
    print("📝 Encoding captions for text–image alignment...")
    text_vecs = []
    for img_id in tqdm(img_ids, desc="Text Encoding", unit="img"):
        cap_list = caps.get(img_id, [])
        if not cap_list:
            text_vecs.append(np.zeros((model.config.projection_dim,), dtype="float32"))
            continue
        inputs = processor(text=cap_list, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            t = model.get_text_features(**inputs)
            t = t / t.norm(dim=-1, keepdim=True)
        t_mean = t.mean(dim=0)
        text_vecs.append(t_mean.cpu().numpy().astype("float32"))

    text_vecs = np.stack(text_vecs).astype("float32")

    # ---------------------------------------------------------
    # Align and combine safely
    # ---------------------------------------------------------
    print(f"🔗 Combining image & text features (alpha={ALPHA})")

    # Ensure equal lengths (some images might have failed to load)
    min_len = min(len(img_vecs), len(text_vecs))
    if len(img_vecs) != len(text_vecs):
        print(f"⚠️ Mismatch detected: {len(img_vecs)} image embeddings vs {len(text_vecs)} text embeddings.")
        print(f"Truncating to the smaller length = {min_len}")
        img_vecs = img_vecs[:min_len]
        text_vecs = text_vecs[:min_len]
        img_ids = img_ids[:min_len]

    # Combine & normalize
    V = (img_vecs * (1 - ALPHA)) + (text_vecs * ALPHA)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)

    # ---------------------------------------------------------
    # Save embeddings & build FAISS index
    # ---------------------------------------------------------
    np.save(IMG_EMB_PATH, V)
    ID_LIST_PATH.write_text("\n".join(img_ids))

    if INDEX_PATH.exists():
        old_index = faiss.read_index(str(INDEX_PATH))
        if old_index.d != V.shape[1]:
            print(f"⚠️ Index dim mismatch ({old_index.d} vs {V.shape[1]}), rebuilding.")
            os.remove(INDEX_PATH)

    index = faiss.IndexFlatIP(V.shape[1])  # inner product = cosine similarity
    index.add(V)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"💾 Saved aligned index → {INDEX_PATH} with {index.ntotal} vectors (dim={V.shape[1]})")


# ============================================================

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    main()
