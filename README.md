# Flickr30k Explorer — One-Click Web App

A **Streamlit** web app for exploring the **Flickr30k** dataset with:
- Text→Image search (CLIP + FAISS)
- Image→Image search (upload an image)
- Per-image features: generated caption (BLIP), object detections (YOLOv8n), dominant colors (k-means)
- Original Flickr captions (from `captions.json`)

> ✅ Works **offline** after the first model download (Hugging Face/Ultralytics weights cached under `cache/weights/` if you set `HF_HOME` and `YOLO_VERBOSE=0`).

---

## 1) Install

```bash
python -m venv .venv && source .venv/bin/activate     # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

> If you have no GPU, that's fine—this setup uses CPU by default (slower but OK for demos).

---

## 2) Prepare the Dataset

Place images under:

```
data/flickr30k/images/   # e.g., 1000092795.jpg, 10002456.jpg, ...
```

Create `data/flickr30k/captions.json` with **5 captions per image** like:

```json
{
  "1000092795.jpg": [
    "Young boy in a red shirt is playing with a red ball.",
    "A child wearing red plays with a ball.",
    "A boy plays with a red ball outdoors.",
    "A kid in a red shirt holds a ball.",
    "The boy plays with a ball in a park."
  ]
}
```

---

## 3) Build the Index & Features

```bash
# 1) Build CLIP embeddings + FAISS index
python scripts/02_build_index.py

# 2) Extract features (YOLO objects, color palette, BLIP caption cache)
python scripts/03_extract_features.py
```

> The first run will download model weights. You can set a cache dir (recommended):
>
> ```bash
> export HF_HOME="$(pwd)/cache/weights"
> export TRANSFORMERS_CACHE="$HF_HOME"
> export HUGGINGFACE_HUB_CACHE="$HF_HOME"
> ```

---

## 4) Run the Web App

```bash
streamlit run app/app.py
```
Then open the local URL shown in the terminal.

---

## 5) Notes
- If you only want text→image search, you can skip the feature extraction step (`03_extract_features.py`), but the detail panel will have fewer fields.
- For speed, try a subset of images (e.g., 5k) during development.
- Everything stores under `data/`—safe to delete and regenerate.

---

## 6) Troubleshooting
- **Out of memory**: Reduce image count or switch to `openai/clip-vit-base-patch32` in `02_build_index.py`.
- **Slow BLIP captioning**: It runs once per image in `03_extract_features.py`; keep images limited or run overnight.
- **No results**: Ensure `captions.json` matches your image filenames exactly.