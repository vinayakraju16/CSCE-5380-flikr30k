import os, json, csv, re
from pathlib import Path
import numpy as np
import faiss
from PIL import Image, ImageOps
import torch
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict
from typing import Dict, List, Set
from tqdm import tqdm

# === Paths ===
DATA_DIR = Path("data/flickr30k")
IMG_DIR = DATA_DIR / "images"
CAP_PATH = DATA_DIR / "captions.csv"

EMB_DIR = Path("data/embeddings"); EMB_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR = Path("data/index"); INDEX_DIR.mkdir(parents=True, exist_ok=True)

IMG_EMB_PATH = EMB_DIR / "img_only.npy"
MIX_EMB_PATH = EMB_DIR / "img_text.npy"
ID_LIST_PATH = EMB_DIR / "img_ids.txt"
INDEX_IMG_PATH = INDEX_DIR / "faiss_img.index"
INDEX_MIX_PATH = INDEX_DIR / "faiss_mix.index"

# === Model and Settings ===
MODEL_ID = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32
ALPHA = 0.5  # weight for text in mixed embedding

USE_GPU = torch.cuda.is_available()
MODEL_DTYPE = torch.float16 if USE_GPU else torch.float32
ENABLE_AUGMENTATION = True  # Set to False to disable data augmentation
AUGMENT_HFLIP = True
AUGMENT_ROTATIONS: List[int] = []  # e.g., [15, -15] to include small rotations


# ============================================================
# 🧩 Helper: Load captions (CSV or JSON)
# ============================================================
def load_captions_auto(path):
    if not path.exists():
        raise FileNotFoundError(f"❌ Captions file not found: {path}")

    first_line = open(path, encoding="utf-8").read(100).strip()
    caps = defaultdict(list)

    if first_line.startswith("{"):
        print("📘 Detected JSON captions file")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    print("📗 Detected CSV captions file — cleaning image names")
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 2:
                continue
            img, cap = row[0].strip(), ",".join(row[1:]).strip()
            img = re.split(r"(\.jpg|\.jpeg|\.png)", img, flags=re.I)[0] + ".jpg"
            caps[img].append(cap.strip(" .,\"“”"))
    print(f"✅ Loaded {len(caps)} cleaned image entries")
    return caps


def augment_image(img: Image.Image) -> List[Image.Image]:
    """Generate augmented variants for robustness."""
    variants: List[Image.Image] = [img]
    if not ENABLE_AUGMENTATION:
        return variants

    if AUGMENT_HFLIP:
        variants.append(ImageOps.mirror(img))

    for angle in AUGMENT_ROTATIONS:
        variants.append(img.rotate(angle, resample=Image.BICUBIC))

    return variants


# ============================================================
# 🧠 Build CLIP embeddings and FAISS indexes
# ============================================================
def main():
    caps = load_captions_auto(CAP_PATH)
    img_ids = sorted(list(caps.keys()))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Loading CLIP model: {MODEL_ID} on {device}")
    model = CLIPModel.from_pretrained(
        MODEL_ID,
        torch_dtype=MODEL_DTYPE if device == "cuda" else torch.float32,
    ).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    # ---------- Image Embeddings ----------
    def process_batch(img_batch):
        inputs = processor(images=img_batch, return_tensors="pt", padding=True).to(device)
        if MODEL_DTYPE == torch.float16 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            # Normalize to unit vectors for cosine similarity
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")

    emb_acc: Dict[str, List[np.ndarray]] = {}
    img_order: List[str] = []
    seen_ids: Set[str] = set()
    batch_imgs: List[Image.Image] = []
    batch_ids: List[str] = []

    def flush_batch():
        nonlocal batch_imgs, batch_ids
        if not batch_imgs:
            return
        feats = process_batch(batch_imgs)
        for img_key, feat in zip(batch_ids, feats):
            if img_key not in emb_acc:
                emb_acc[img_key] = []
            emb_acc[img_key].append(feat)
            if img_key not in seen_ids:
                img_order.append(img_key)
                seen_ids.add(img_key)
        batch_imgs = []
        batch_ids = []

    for img_id in tqdm(img_ids, desc="Encoding Images"):
        path = IMG_DIR / img_id
        if not path.exists():
            print(f"⚠️ Missing image file: {img_id}")
            continue
        try:
            img = Image.open(path).convert("RGB")
            _ = img.size
            variants = augment_image(img)
            batch_imgs.extend(variants)
            batch_ids.extend([img_id] * len(variants))
            if len(batch_imgs) >= BATCH_SIZE:
                flush_batch()
        except Exception as e:
            print(f"⚠️ Skipping {img_id}: {e}")
            continue

    flush_batch()

    if not emb_acc:
        raise ValueError("❌ No valid images processed! Check image paths and format.")

    valid_img_ids = img_order

    img_vecs = []
    for img_id in valid_img_ids:
        arr = np.stack(emb_acc[img_id]).astype("float32")
        mean_vec = arr.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm == 0:
            continue
        img_vecs.append((mean_vec / norm).astype("float32"))

    if not img_vecs:
        raise ValueError("❌ Failed to compute any image embeddings after augmentation averaging.")

    img_vecs = np.stack(img_vecs).astype("float32")
    np.save(IMG_EMB_PATH, img_vecs)
    print(f"💾 Saved pure image embeddings → {IMG_EMB_PATH}")

    # ---------- Text Embeddings ----------
    print("📝 Encoding captions for mixed index ...")
    text_vecs = []
    # Use valid_img_ids to ensure alignment with image embeddings
    text_img_ids = valid_img_ids if valid_img_ids else img_ids
    
    for img_id in tqdm(text_img_ids, desc="Encoding Captions"):
        cap_list = caps.get(img_id, [])
        if not cap_list:
            text_vecs.append(np.zeros((model.config.projection_dim,), dtype="float32"))
            continue
        inputs = processor(text=cap_list, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            t = model.get_text_features(**inputs)
            t = t / t.norm(dim=-1, keepdim=True)
        text_vecs.append(t.mean(dim=0).cpu().numpy().astype("float32"))

    text_vecs = np.stack(text_vecs).astype("float32")
    # Normalize text embeddings
    text_norms = np.linalg.norm(text_vecs, axis=1, keepdims=True)
    text_norms[text_norms == 0] = 1
    text_vecs = text_vecs / text_norms

    # ---------- Mixed Embeddings ----------
    if len(valid_img_ids) != len(img_vecs):
        raise ValueError(
            f"❌ Mismatch: {len(img_vecs)} image embeddings but {len(valid_img_ids)} image IDs"
        )
    
    if len(text_vecs) != len(valid_img_ids):
        raise ValueError(
            f"❌ Mismatch: {len(text_vecs)} text embeddings but {len(valid_img_ids)} image IDs"
        )
    
    min_len = min(len(img_vecs), len(text_vecs))
    img_vecs, text_vecs, valid_img_ids = img_vecs[:min_len], text_vecs[:min_len], valid_img_ids[:min_len]
    
    # Ensure both vectors are normalized before mixing
    img_vecs = img_vecs / np.linalg.norm(img_vecs, axis=1, keepdims=True)
    text_vecs = text_vecs / np.linalg.norm(text_vecs, axis=1, keepdims=True)
    
    # Create mixed embeddings
    mix_vecs = (img_vecs * (1 - ALPHA)) + (text_vecs * ALPHA)
    # Normalize mixed embeddings
    mix_norms = np.linalg.norm(mix_vecs, axis=1, keepdims=True)
    mix_norms[mix_norms == 0] = 1
    mix_vecs = mix_vecs / mix_norms
    
    np.save(MIX_EMB_PATH, mix_vecs)
    ID_LIST_PATH.write_text("\n".join(valid_img_ids))

    # ---------- Build FAISS Indexes ----------
    index_img = faiss.IndexFlatIP(img_vecs.shape[1])
    index_img.add(img_vecs)
    faiss.write_index(index_img, str(INDEX_IMG_PATH))
    print(f"💾 Saved image-only index → {INDEX_IMG_PATH}")

    index_mix = faiss.IndexFlatIP(mix_vecs.shape[1])
    index_mix.add(mix_vecs)
    faiss.write_index(index_mix, str(INDEX_MIX_PATH))
    print(f"💾 Saved mixed text+image index → {INDEX_MIX_PATH}")


# ============================================================
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    main()
