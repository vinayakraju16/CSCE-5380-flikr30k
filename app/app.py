import os, io, json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import faiss
import torch
from transformers import CLIPModel, CLIPProcessor

# ============================================================
# Paths
# ============================================================
DATA_DIR = Path("data/flickr30k")
IMG_DIR = DATA_DIR / "images"
FEAT_PATH = Path("data/features/features.parquet")
ID_LIST_PATH = Path("data/embeddings/img_ids.txt")

# Two separate indexes
INDEX_IMG_PATH = Path("data/index/faiss_img.index")
INDEX_MIX_PATH = Path("data/index/faiss_mix.index")

MODEL_ID = "openai/clip-vit-base-patch32"

# ============================================================
# Cached model loading
# ============================================================
@st.cache_resource
def load_clip(model_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_id)
    return model, proc, device

# ============================================================
# Load FAISS index (based on mode)
# ============================================================
@st.cache_resource
def load_index(index_path):
    index = faiss.read_index(str(index_path))
    ids = ID_LIST_PATH.read_text().strip().split("\n")
    return index, ids

# ============================================================
# Load features (optional)
# ============================================================
@st.cache_data
def load_features():
    if FEAT_PATH.exists():
        return pd.read_parquet(FEAT_PATH).set_index("image_id")
    else:
        return None

# ============================================================
# Search Functions
# ============================================================
def search_text(q, k=16):
    model, proc, device = load_clip(MODEL_ID)
    inputs = proc(text=[q], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        t = model.get_text_features(**inputs)
        t = t / t.norm(dim=-1, keepdim=True)
        t = t.cpu().numpy().astype("float32")

    index, ids = load_index(INDEX_MIX_PATH)
    D, I = index.search(t, k)
    return [(ids[i], float(s)) for i, s in zip(I[0].tolist(), D[0].tolist())]

def search_image(img: Image.Image, k=16, min_score=None):
    model, proc, device = load_clip(MODEL_ID)
    
    # Ensure consistent preprocessing: convert to RGB and use same settings as indexing
    img = img.convert("RGB")
    
    # Use padding=True to match the indexing preprocessing exactly
    inputs = proc(images=img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        v = model.get_image_features(**inputs)
        # Normalize to unit vector (same as indexing)
        v = v / v.norm(dim=-1, keepdim=True)
        v = v.cpu().numpy().astype("float32")
    
    # Ensure the query vector is properly shaped (1, dim)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    
    index, ids = load_index(INDEX_IMG_PATH)
    D, I = index.search(v, k)
    
    matches = []
    for idx, score in zip(I[0].tolist(), D[0].tolist()):
        if idx < 0 or idx >= len(ids):
            continue
        matches.append((ids[idx], float(score)))
    
    if min_score is not None:
        filtered = [(img_id, score) for img_id, score in matches if score >= min_score]
    else:
        filtered = matches

    # Return filtered results alongside all raw matches for downstream logic
    return filtered, matches

# ============================================================
# Optional detailed view
# ============================================================
def show_detail(image_id, features_df, score=None):
    """Display detailed information about an image."""
    img_path = IMG_DIR / image_id
    
    # Image display with score
    detail_img_col1, detail_img_col2, detail_img_col3 = st.columns([1, 2, 1])
    with detail_img_col2:
        if score is not None:
            score_color = "#10b981" if score > 0.8 else "#f59e0b" if score > 0.6 else "#ef4444"
            caption_with_score = f"{image_id} | Similarity: {score:.3f}"
            st.image(str(img_path), use_container_width=True, caption=caption_with_score)
            # Display score badge
            st.markdown(
                f"""
                <div style="text-align: center; margin-top: 0.5rem;">
                    <span style="background: {score_color}; color: white; padding: 0.5rem 1.2rem; 
                    border-radius: 25px; font-size: 1rem; font-weight: 600;">
                        Similarity Score: {score:.3f}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.image(str(img_path), use_container_width=True, caption=image_id)
    
    if features_df is not None and image_id in features_df.index:
        row = features_df.loc[image_id].to_dict()
        
        # Captions section - Show prominently at top
        st.markdown('<div class="detail-section">', unsafe_allow_html=True)
        st.markdown("### 📝 Image Captions")
        
        # BLIP caption - Show first and prominently
        blip_caption = row.get("gen_caption", "")
        if blip_caption:
            st.markdown("**🤖 Generated Caption (BLIP):**")
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                    <p style="margin: 0; font-size: 1.1rem; font-weight: 500;">{blip_caption}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Flickr captions
        caps = row.get("flickr_captions", [])
        if isinstance(caps, list) and caps:
            st.markdown("**📸 Original Flickr Captions:**")
            for i, cap in enumerate(caps, 1):
                st.markdown(f"**{i}.** {cap}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Colors section
        colors = row.get("colors", [])
        if isinstance(colors, list) and colors:
            st.markdown('<div class="detail-section">', unsafe_allow_html=True)
            st.markdown("### 🎨 Dominant Colors")
            cols = st.columns(min(5, len(colors)))
            for i, c in enumerate(colors[:5]):
                with cols[i]:
                    st.markdown(
                        f"""
                        <div style='width:100%; height:60px; border-radius:8px; 
                        background:rgb({c[0]},{c[1]},{c[2]}); 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.2); margin-bottom: 0.5rem;'>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.caption(f"RGB{tuple(c)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Objects section
        objs = row.get("objects", [])
        if isinstance(objs, list) and len(objs) > 0:
            st.markdown('<div class="detail-section">', unsafe_allow_html=True)
            st.markdown("### 🎯 Detected Objects (YOLOv8)")
            
            # Group objects by class
            obj_list = []
            for o in objs[:20]:  # Limit to top 20
                cls_id = o.get('cls', '?')
                conf = o.get('conf', 0.0)
                obj_list.append(f"Class {cls_id} ({conf:.2%})")
            
            # Display as badges
            st.markdown(" ".join([f"`{obj}`" for obj in obj_list]))
            st.caption(f"Total objects detected: {len(objs)}")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        # st.info("ℹ️ Run `python scripts/03_extract_features.py` to extract detailed features (captions, colors, objects) for this image.")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# Custom CSS Styling
# ============================================================
def load_custom_css():
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            padding-top: 2rem;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem 1.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .main-header h1 {
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-align: center;
        }
        
        .main-header p {
            color: rgba(255, 255, 255, 0.9);
            text-align: center;
            margin-top: 0.5rem;
            font-size: 1.1rem;
        }
        
        /* Search container */
        .search-container {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
            margin-bottom: 2rem;
        }
        
        /* Results grid */
        .result-card {
            background: white;
            border-radius: 12px;
            padding: 0.5rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
            margin-bottom: 1rem;
        }
        
        .result-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* Ensure images display properly */
        .stImage img {
            width: 100% !important;
            height: auto !important;
            min-height: 200px !important;
            object-fit: contain !important;
            border-radius: 8px;
        }
        
        /* Image container */
        [data-testid="stImage"] {
            width: 100%;
        }
        
        /* Similarity badge */
        .similarity-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
            margin-top: 0.5rem;
        }
        
        /* Stats container */
        .stats-container {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }
        
        .stat-box {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 10px;
            flex: 1;
            min-width: 150px;
            text-align: center;
        }
        
        .stat-box h3 {
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
        }
        
        .stat-box p {
            margin: 0.3rem 0 0 0;
            font-size: 0.9rem;
            opacity: 0.9;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            padding-top: 3rem;
        }
        
        /* Button styling */
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        /* Input styling */
        .stTextInput > div > div > input {
            border-radius: 8px;
            border: 2px solid #e0e0e0;
            padding: 0.75rem;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        /* Detail view */
        .detail-section {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }
        
        .detail-section h3 {
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }
        
        /* Color swatches */
        .color-swatch {
            width: 100%;
            height: 50px;
            border-radius: 8px;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        /* Loading animation */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .loading {
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        
        /* Info boxes */
        .info-box {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
        
        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 3rem;
            color: #999;
        }
        
        .empty-state svg {
            width: 100px;
            height: 100px;
            margin: 0 auto 1rem;
            opacity: 0.3;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# Streamlit UI
# ============================================================
def main():
    st.set_page_config(
        page_title="Flickr30k Explorer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Flickr30k Image Search</h1>
        <p>Advanced semantic search powered by CLIP and FAISS</p>
    </div>
    """, unsafe_allow_html=True)

    min_score = 0.0
    with st.sidebar:
        st.markdown("### ⚙️ Search Settings")
        st.markdown("---")
        
        mode = st.radio(
            "**Search Mode**",
            ["Text → Image", "Image → Image"],
            help="Choose between text-based or image-based search"
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 Parameters")
        k = st.slider(
            "**Top-K Results**",
            min_value=4,
            max_value=40,
            value=16,
            step=4,
            help="Number of results to retrieve"
        )
        
        if mode == "Image → Image":
            min_score = st.slider(
                "**Similarity Threshold**",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                help="Minimum cosine similarity score (0.0 = no filter)"
            )
        
        st.markdown("---")
        
        # System info
        with st.expander("ℹ️ System Information"):
            st.markdown(f"**Model:** {MODEL_ID}")
            st.markdown(f"**Index Type:** {'Mixed (Image+Text)' if mode == 'Text → Image' else 'Image-Only'}")
            
            # Check if features are available
            features_check = load_features()
            if features_check is not None:
                st.success("✅ Features loaded")
                st.caption(f"{len(features_check)} images with metadata")
            else:
                st.info("ℹ️ Run feature extraction for detailed view")
        
        st.markdown("---")
        st.caption("Built with Streamlit, CLIP & FAISS")

    # Load features for main content
    features_df = load_features()
    
    # Check if we're in detail view mode - if so, show detail and skip search
    if "detail" in st.session_state and st.session_state["detail"]:
        st.markdown("---")
        
        detail_col1, detail_col2 = st.columns([1, 4])
        with detail_col1:
            if st.button("← Back to Results", use_container_width=True):
                del st.session_state["detail"]
                if "detail_score" in st.session_state:
                    del st.session_state["detail_score"]
                st.rerun()
        
        with detail_col2:
            st.markdown(f"### 📋 Image Details: `{st.session_state['detail']}`")
        
        # Get score if available
        detail_score = st.session_state.get("detail_score", None)
        show_detail(st.session_state["detail"], features_df, score=detail_score)
        return  # Exit early, don't show search interface
    
    # Normal search mode
    # Initialize session state for results if not exists
    if "search_results" not in st.session_state:
        st.session_state["search_results"] = []
    if "raw_count" not in st.session_state:
        st.session_state["raw_count"] = 0
    
    # ------------------------------------------------------------
    # Search Interface
    # ------------------------------------------------------------
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    if mode == "Text → Image":
        col1, col2 = st.columns([4, 1])
        with col1:
            q = st.text_input(
                "Enter your search query",
                value="a man riding a bicycle on the street",
                placeholder="Describe the image you're looking for...",
                label_visibility="collapsed"
            )
        with col2:
            search_btn = st.button("🔍 Search", type="primary", use_container_width=True)
        
        if search_btn and q:
            with st.spinner("Searching..."):
                try:
                    results = search_text(q, k=k)
                    st.session_state["search_results"] = results
                    st.session_state["raw_count"] = len(results)
                    st.success(f"Found {len(results)} results")
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.session_state["search_results"] = []
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            up = st.file_uploader(
                "Upload an image to search",
                type=["jpg", "jpeg", "png"],
                help="Upload an image to find similar images in the dataset"
            )
        with col2:
            search_btn = st.button("🔍 Search", type="primary", use_container_width=True)
        
        if up is not None:
            # Show preview
            preview_col1, preview_col2 = st.columns([1, 3])
            with preview_col1:
                img_preview = Image.open(up).convert("RGB")
                st.image(img_preview, caption="Query Image", use_container_width=True)
        
        if search_btn and up is not None:
            with st.spinner("Processing image and searching..."):
                try:
                    img = Image.open(up).convert("RGB")
                    filtered_results, raw_results = search_image(
                        img, k=k, min_score=min_score if min_score > 0 else None
                    )
                    st.session_state["search_results"] = filtered_results
                    st.session_state["raw_count"] = len(raw_results)
                    st.success(f"Found {len(filtered_results)} results (from {len(raw_results)} candidates)")
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.session_state["search_results"] = []
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Use results from session state
    display_results = st.session_state["search_results"]
    raw_count = st.session_state["raw_count"]

    # ------------------------------------------------------------
    # Results Display
    # ------------------------------------------------------------
    if display_results:
        # Stats header
        st.markdown("### 📊 Search Results")
        
        # Statistics
        avg_score = np.mean([score for _, score in display_results]) if display_results else 0
        max_score = max([score for _, score in display_results]) if display_results else 0
        
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1:
            st.metric("Results Found", len(display_results))
        with stats_col2:
            st.metric("Avg Similarity", f"{avg_score:.3f}")
        with stats_col3:
            st.metric("Max Similarity", f"{max_score:.3f}")
        
        if mode == "Image → Image" and min_score > 0:
            st.info(f"📌 Showing matches with similarity ≥ {min_score:.2f}")
        
        st.markdown("---")
        
        # Results grid
        num_cols = 4
        cols = st.columns(num_cols)
        
        # Debug: Check if image directory exists
        if not IMG_DIR.exists():
            st.error(f"❌ Image directory not found: {IMG_DIR}")
            st.stop()
        
        for i, (img_id, score) in enumerate(display_results):
            img_path = IMG_DIR / img_id
            if not img_path.exists():
                st.warning(f"⚠️ Image not found: {img_id}")
                continue
            
            with cols[i % num_cols]:
                # Card container with proper styling
                try:
                    # Load image with PIL to ensure it's valid
                    img = Image.open(img_path).convert("RGB")
                    
                    # Resize if image is too large (for performance)
                    max_size = (400, 400)
                    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                        img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                    # Display image with explicit width to ensure visibility
                    st.image(
                        img,
                        use_container_width=True,
                        caption=None  # We'll add caption separately
                    )
                    
                    # Similarity badge
                    score_color = "#10b981" if score > 0.8 else "#f59e0b" if score > 0.6 else "#ef4444"
                    st.markdown(
                        f"""
                        <div style="text-align: center; margin-top: 0.5rem;">
                            <span style="background: {score_color}; color: white; padding: 0.3rem 0.8rem; 
                            border-radius: 20px; font-size: 0.85rem; font-weight: 600;">
                                {score:.3f}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Image ID (truncated)
                    short_id = img_id[:20] + "..." if len(img_id) > 20 else img_id
                    st.caption(short_id)
                    
                    # Click to view details
                    if st.button("View Details", key=f"detail_{i}", use_container_width=True):
                        st.session_state["detail"] = img_id
                        st.session_state["detail_score"] = score  # Save score for detail view
                        st.session_state["last_results"] = display_results  # Save results for back button
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error loading {img_id}: {str(e)}")
                    continue
    else:
        # Empty state
        if mode == "Image → Image" and raw_count > 0 and min_score > 0:
            st.warning("⚠️ No matches met the similarity threshold.")
            st.info("💡 Try lowering the minimum similarity or increasing Top-K.")
        else:
            st.info("👆 Enter a query or upload an image to start searching.")

# ============================================================
if __name__ == "__main__":
    main()
