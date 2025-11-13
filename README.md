# Flickr30k Explorer — Advanced Image Search System

A **professional Streamlit web application** for exploring the **Flickr30k** dataset with state-of-the-art semantic search capabilities powered by CLIP and FAISS.

## ✨ Features

### 🔍 Search Capabilities
- **Text→Image Search**: Find images using natural language queries
- **Image→Image Search**: Upload an image to find visually similar images
- **Similarity Threshold Filtering**: Filter results by minimum similarity score
- **Real-time Search**: Fast query processing (<100ms average)

### 🎨 Professional UI
- Modern gradient design with responsive layout
- Color-coded similarity scores (green/yellow/red)
- Interactive detail view with image metadata
- Statistics dashboard (Results Found, Avg/Max Similarity)
- Image preview and thumbnail display

### 📊 Rich Metadata
- **BLIP-generated captions**: AI-generated image descriptions
- **YOLOv8 object detection**: Detected objects with confidence scores
- **Dominant color extraction**: K-means color palette analysis
- **Original Flickr captions**: Multiple captions per image

### 📈 Evaluation & Metrics
- Comprehensive evaluation scripts for accuracy metrics
- Precision@K, Recall@K, MAP, MRR calculations
- Performance benchmarking tools
- Visualization and reporting

---

## 🚀 Quick Start

### 1) Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> 💡 **Note**: If you have no GPU, that's fine—this setup uses CPU by default (slower but functional for demos).

---

### 2) Prepare the Dataset

**Place images:**

For getting the images download the dataset from this google drive link

```
https://drive.google.com/drive/folders/1BUNjwR-U9mIImKGk3Rry14g1umQ-Y481?usp=sharing
```
Then copy the images folder and paste in this path
```
data/flickr30k/images/   # e.g., 1000092795.jpg, 10002456.jpg, ...
```

**Create captions file** (`data/flickr30k/captions.json` or `captions.csv`):

**JSON format:**
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

**CSV format:**
```csv
image_id,caption1,caption2,caption3,caption4,caption5
1000092795.jpg,"Young boy in a red shirt is playing with a red ball.","A child wearing red plays with a ball.",...
```

---

### 3) Build Indexes & Extract Features

#### Step 1: Build CLIP Embeddings & FAISS Indexes

```bash
python scripts/02_build_index.py
```

**What it does:**
- Generates CLIP embeddings for all images
- Creates text embeddings from captions
- Builds two FAISS indexes:
  - `faiss_img.index` - Image-only embeddings (for image→image search)
  - `faiss_mix.index` - Mixed image+text embeddings (for text→image search)
- Optional: Data augmentation for improved robustness

**Output:**
- `data/embeddings/img_only.npy` - Image embeddings
- `data/embeddings/img_text.npy` - Mixed embeddings
- `data/embeddings/img_ids.txt` - Image ID mapping
- `data/index/faiss_img.index` - Image search index
- `data/index/faiss_mix.index` - Text search index

#### Step 2: Extract Features (Optional but Recommended)

```bash
# Full extraction
python scripts/03_extract_features.py

# Quick test with limited images
python scripts/03_extract_features.py --limit 100

# Skip YOLO for faster processing
python scripts/03_extract_features.py --skip-yolo

# Custom batch size for BLIP
python scripts/03_extract_features.py --blip-batch 8
```

**Options:**
- `--limit N`: Process only N images
- `--skip-yolo`: Skip object detection (faster)
- `--blip-batch N`: Batch size for BLIP captioning (default: 4)
- `--conf FLOAT`: YOLO confidence threshold (default: 0.25)

**Output:**
- `data/features/features.parquet` - Metadata for all images

> 💡 **Tip**: The first run will download model weights. Set cache directory:
> ```bash
> export HF_HOME="$(pwd)/cache/weights"
> export TRANSFORMERS_CACHE="$HF_HOME"
> export HUGGINGFACE_HUB_CACHE="$HF_HOME"
> ```

---

### 4) Run the Web Application

```bash
streamlit run app/app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`).

---

## 📊 Evaluation & Metrics

### Run Evaluation

```bash
# Evaluate both search modes
python scripts/04_evaluate_metrics.py --mode both --num-queries 100

# Evaluate only image-to-image search
python scripts/04_evaluate_metrics.py --mode image --num-queries 50

# Evaluate only text-to-image search
python scripts/04_evaluate_metrics.py --mode text --num-queries 50
```

### Get Quick Summary

```bash
python scripts/05_quick_metrics.py
```

**Output:**
- `data/evaluation/evaluation_report_{mode}.txt` - Detailed text report
- `data/evaluation/evaluation_stats_{mode}.json` - JSON statistics
- `data/evaluation/precision_at_k_{mode}.png` - Precision@K visualization
- `data/evaluation/recall_at_k_{mode}.png` - Recall@K visualization
- `data/evaluation/summary_table.csv` - Summary table

**Metrics Computed:**
- **Precision@K**: Fraction of top-K results that are relevant
- **Recall@K**: Fraction of relevant items found in top-K
- **MAP (Mean Average Precision)**: Overall quality across queries
- **MRR (Mean Reciprocal Rank)**: How quickly first relevant result appears
- **Similarity Statistics**: Score distribution analysis
- **Search Time**: Query processing performance

See `EVALUATION_GUIDE.md` for detailed explanations.

---

## 🎯 Usage Guide

### Text-to-Image Search

1. Select **"Text → Image"** mode in the sidebar
2. Enter your search query (e.g., "a man riding a bicycle on the street")
3. Adjust **Top-K** slider (4-40 results)
4. Click **"🔍 Search"**
5. Browse results with similarity scores
6. Click **"View Details"** on any image for full metadata

### Image-to-Image Search

1. Select **"Image → Image"** mode in the sidebar
2. Upload an image file (JPG, PNG)
3. Adjust **Top-K** and **Similarity Threshold** (optional)
4. Click **"🔍 Search"**
5. View similar images ranked by cosine similarity
6. Use threshold to filter low-quality matches

### Detail View

Click **"View Details"** on any result to see:
- **Similarity Score**: Color-coded badge
- **Image Captions**: BLIP-generated + original Flickr captions
- **Dominant Colors**: Extracted color palette
- **Detected Objects**: YOLOv8 detections with confidence

---

## 📁 Project Structure

```
CSCE-5380-flikr30k/
├── app/
│   └── app.py                 # Main Streamlit application
├── scripts/
│   ├── 02_build_index.py      # Build CLIP embeddings & FAISS indexes
│   ├── 03_extract_features.py # Extract metadata (YOLO, BLIP, colors)
│   ├── 04_evaluate_metrics.py # Evaluation script
│   └── 05_quick_metrics.py    # Quick metrics summary
├── data/
│   ├── flickr30k/
│   │   ├── images/            # Image files (31K+ images)
│   │   ├── captions.json      # Caption data
│   │   └── captions.csv       # Alternative caption format
│   ├── embeddings/             # Generated embeddings
│   │   ├── img_only.npy
│   │   ├── img_text.npy
│   │   └── img_ids.txt
│   ├── index/                  # FAISS indexes
│   │   ├── faiss_img.index
│   │   └── faiss_mix.index
│   └── features/               # Extracted metadata
│       └── features.parquet
├── cache/
│   └── weights/                # Cached model weights
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── EVALUATION_GUIDE.md         # Evaluation documentation
```

---

## ⚙️ Configuration

### Model Settings

Edit `scripts/02_build_index.py`:
- `MODEL_ID`: CLIP model (default: `"openai/clip-vit-base-patch32"`)
- `BATCH_SIZE`: Processing batch size (default: 32)
- `ALPHA`: Text weight in mixed embeddings (default: 0.5)
- `ENABLE_AUGMENTATION`: Enable data augmentation (default: False)

### Feature Extraction Settings

Edit `scripts/03_extract_features.py`:
- `BLIP_BATCH_DEFAULT`: BLIP batch size (default: 4)
- `YOLO_CONF_DEFAULT`: YOLO confidence threshold (default: 0.25)
- `YOLO_IMGSZ`: YOLO input size (default: 640)

---

## 🔧 Troubleshooting

### Images Not Showing
- **Check image paths**: Ensure images are in `data/flickr30k/images/`
- **Verify file format**: Images should be JPG/JPEG/PNG
- **Check permissions**: Ensure read access to image directory
- **Rebuild index**: Run `02_build_index.py` again if images were added

### Out of Memory
- Reduce image count during development
- Use smaller CLIP model variant
- Process images in smaller batches
- Use CPU mode (slower but uses less memory)

### Slow Performance
- **BLIP captioning**: Use `--skip-yolo` or `--limit` for faster runs
- **Index building**: Reduce batch size or disable augmentation
- **Search**: Results are cached; first search may be slower

### No Search Results
- **Check captions**: Ensure `captions.json` matches image filenames exactly
- **Rebuild indexes**: Run `02_build_index.py` if captions were updated
- **Verify indexes**: Check that `data/index/` contains `.index` files
- **Check embeddings**: Verify `data/embeddings/img_ids.txt` exists

### Evaluation Errors
- Ensure indexes are built before running evaluation
- Check that test images exist in the dataset
- Verify captions file is properly formatted

---

## 📝 Notes

- **Feature extraction is optional**: You can skip `03_extract_features.py` if you only need search functionality (detail view will have fewer fields)
- **Development tip**: Use `--limit 100` during testing to process fewer images
- **Data regeneration**: Everything under `data/` can be safely deleted and regenerated
- **Offline mode**: Works offline after initial model download (cache weights in `cache/weights/`)

---

## 🎓 Technical Details

### Architecture
- **CLIP Model**: OpenAI CLIP-ViT-Base-Patch32 for semantic embeddings
- **FAISS**: Facebook AI Similarity Search for fast nearest neighbor search
- **BLIP**: Salesforce BLIP for image captioning
- **YOLOv8**: Ultralytics YOLOv8n for object detection
- **Streamlit**: Web framework for interactive UI

### Search Pipeline
1. **Text Query** → CLIP text encoder → Mixed index (image+text embeddings)
2. **Image Query** → CLIP image encoder → Image-only index
3. **FAISS Search** → Cosine similarity → Top-K results
4. **Post-processing** → Threshold filtering → Display

### Performance
- **Average search time**: 40-50ms per query
- **Index size**: ~125 MB for 31K images
- **Embedding dimension**: 512 (CLIP base model)
- **Index type**: IndexFlatIP (Inner Product for cosine similarity)

---

## 📚 Additional Resources

- **Evaluation Guide**: See `EVALUATION_GUIDE.md` for detailed metrics explanation
- **CLIP Paper**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **FAISS Documentation**: [Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)

---

