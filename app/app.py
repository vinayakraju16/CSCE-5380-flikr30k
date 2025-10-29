import os, io, json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import faiss
import torch
from transformers import CLIPModel, CLIPProcessor

DATA_DIR = Path("data/flickr30k")
IMG_DIR = DATA_DIR / "images"
CAPS_PATH = DATA_DIR / "captions.json"
EMB_PATH = Path("data/embeddings/img.npy")
ID_LIST_PATH = Path("data/embeddings/img_ids.txt")
INDEX_PATH = Path("data/index/faiss.index")
FEAT_PATH = Path("data/features/features.parquet")

MODEL_ID = os.environ.get("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")

@st.cache_resource
def load_clip(model_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_id)
    return model, proc, device

@st.cache_resource
def load_index():
    index = faiss.read_index(str(INDEX_PATH))
    ids = ID_LIST_PATH.read_text().strip().split("\n")
    dim = index.d

    # sanity check: CLIP model embedding dim must match index dim
    model, _, _ = load_clip(MODEL_ID)
    test_vec = model.get_text_features(**CLIPProcessor.from_pretrained(MODEL_ID)(
        text=["test"], return_tensors="pt"
    ))
    test_dim = test_vec.shape[-1]

    if dim != test_dim:
        st.warning(f"⚠️ FAISS index dim={dim} ≠ model dim={test_dim}. Rebuild index with same model.")
    return index, ids

@st.cache_data
def load_features():
    if FEAT_PATH.exists():
        return pd.read_parquet(FEAT_PATH).set_index("image_id")
    else:
        return None

def search_text(q, k=16):
    try:
        model, proc, device = load_clip(MODEL_ID)
        inputs = proc(text=[q], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            t = model.get_text_features(**inputs)
            t = t / t.norm(dim=-1, keepdim=True)
            t = t.cpu().numpy().astype("float32")

        index, ids = load_index()
        D, I = index.search(t, k)
        results = [(ids[i], float(s)) for i, s in zip(I[0].tolist(), D[0].tolist())]
        return results

    except Exception as e:
        st.error(f"Search failed: {type(e).__name__} → {e}")
        raise

def search_image(img: Image.Image, k=16):
    model, proc, device = load_clip(MODEL_ID)
    inputs = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        v = model.get_image_features(**inputs)
        v = v / v.norm(dim=-1, keepdim=True)
        v = v.cpu().numpy().astype("float32")
    index, ids = load_index()
    D, I = index.search(v, k)
    results = [(ids[i], float(s)) for i, s in zip(I[0].tolist(), D[0].tolist())]
    return results

def show_detail(image_id, features_df):
    img_path = IMG_DIR / image_id
    st.image(str(img_path), use_column_width=True)
    if features_df is not None and image_id in features_df.index:
        row = features_df.loc[image_id].to_dict()
        st.markdown("**Model caption (BLIP):** " + row.get("gen_caption", ""))
        caps = row.get("flickr_captions", [])
        if isinstance(caps, list):
            st.markdown("**Flickr captions:** " + " | ".join(caps))
        cols = st.columns(5)
        colors = row.get("colors", [])
        for i, c in enumerate(colors[:5] if isinstance(colors, list) else []):
            with cols[i]:
                st.markdown(f"<div style='width:100%%;height:40px;border-radius:6px;background:rgb({c[0]},{c[1]},{c[2]})'></div>", unsafe_allow_html=True)
                st.caption(f"rgb{tuple(c)}")
        objs = row.get("objects", [])
        if isinstance(objs, list) and len(objs)>0:
            st.markdown("**Objects (YOLOv8n):**")
            st.write([f"cls {o.get('cls')} ({o.get('conf'):.2f})" for o in objs[:12]])
    else:
        st.info("Run feature extraction (scripts/03_extract_features.py) to populate captions, colors, and objects.")

def main():
    st.set_page_config(page_title="Flickr30k Explorer", layout="wide")
    st.title("🔎 Flickr30k Explorer")

    with st.sidebar:
        st.subheader("Search")
        mode = st.radio("Mode", ["Text → Image", "Image → Image"], horizontal=True)
        k = st.slider("Top-K", 4, 40, 16, step=4)
        st.markdown("---")
        st.caption("Index files expected:")
        st.code(str(INDEX_PATH))
        st.caption("Features parquet (optional):")
        st.code(str(FEAT_PATH))

    features_df = load_features()

    results = []
    if mode == "Text → Image":
        q = st.text_input("Enter a caption/query", "a man riding a bicycle on the street")
        if st.button("Search", type="primary"):
            try:
                results = search_text(q, k=k)
            except Exception as e:
                st.error(f"Search failed: {e}")
    else:
        up = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
        if st.button("Search", type="primary") and up is not None:
            try:
                img = Image.open(up).convert("RGB")
                results = search_image(img, k=k)
            except Exception as e:
                st.error(f"Search failed: {e}")

    if results:
        st.markdown("### 🔍 Search Results")
        cols = st.columns(4)

        for i, (img_id, score) in enumerate(results):
            img_path = IMG_DIR / img_id
            if not img_path.exists():
                st.warning(f"⚠️ Missing image: {img_id}")
                continue

            with cols[i % 4]:
                st.image(
                str(img_path),
                caption=f"{img_id} | sim={score:.3f}",
                use_container_width=True,   # replaces deprecated use_column_width
                )
    else:
        st.info("No matching results found.")

    if "detail" in st.session_state:
        st.markdown("---")
        st.subheader(f"Details: {st.session_state['detail']}")
        show_detail(st.session_state["detail"], features_df)

if __name__ == "__main__":
    main()