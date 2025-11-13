"""
Evaluation Script for Image Search System
Computes accuracy metrics and performance statistics for presentation.
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from PIL import Image
import torch
import faiss
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# === Paths ===
DATA_DIR = Path("data/flickr30k")
IMG_DIR = DATA_DIR / "images"
CAP_PATH = DATA_DIR / "captions.json"
ID_LIST_PATH = Path("data/embeddings/img_ids.txt")
INDEX_IMG_PATH = Path("data/index/faiss_img.index")
INDEX_MIX_PATH = Path("data/index/faiss_mix.index")
MODEL_ID = "openai/clip-vit-base-patch32"

# === Evaluation Settings ===
K_VALUES = [1, 5, 10, 16, 20]  # Different K values for evaluation


def load_captions(path: Path) -> Dict[str, List[str]]:
    """Load captions from JSON or CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Captions file not found: {path}")
    
    first_line = open(path, encoding="utf-8").read(100).strip()
    if first_line.startswith("{"):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_model_and_index(mode: str = "image"):
    """Load CLIP model and FAISS index."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    
    if mode == "image":
        index = faiss.read_index(str(INDEX_IMG_PATH))
    else:
        index = faiss.read_index(str(INDEX_MIX_PATH))
    
    ids = ID_LIST_PATH.read_text().strip().split("\n")
    return model, processor, device, index, ids


def precision_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    """Compute Precision@K."""
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_retrieved = len(set(relevant) & set(retrieved_k))
    return relevant_retrieved / k


def recall_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    """Compute Recall@K."""
    if len(relevant) == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_retrieved = len(set(relevant) & set(retrieved_k))
    return relevant_retrieved / len(relevant)


def mean_reciprocal_rank(relevant: List[str], retrieved: List[str]) -> float:
    """Compute Mean Reciprocal Rank (MRR)."""
    for rank, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(relevant: List[str], retrieved: List[str]) -> float:
    """Compute Average Precision (AP)."""
    if len(relevant) == 0:
        return 0.0
    
    relevant_set = set(relevant)
    num_relevant = 0
    precision_sum = 0.0
    
    for rank, item in enumerate(retrieved, 1):
        if item in relevant_set:
            num_relevant += 1
            precision_sum += num_relevant / rank
    
    return precision_sum / len(relevant) if len(relevant) > 0 else 0.0


def evaluate_image_to_image(
    model, processor, device, index, ids, 
    test_images: List[str], k_max: int = 20,
    exclude_self: bool = True
) -> Dict:
    """Evaluate image-to-image search."""
    print(f"\n🔍 Evaluating Image-to-Image Search on {len(test_images)} queries...")
    
    results = {
        "precision@k": {k: [] for k in K_VALUES if k <= k_max},
        "recall@k": {k: [] for k in K_VALUES if k <= k_max},
        "mrr": [],
        "map": [],
        "similarity_scores": [],
        "search_times": [],
    }
    
    for img_id in tqdm(test_images, desc="Processing queries"):
        img_path = IMG_DIR / img_id
        if not img_path.exists():
            continue
        
        try:
            img = Image.open(img_path).convert("RGB")
            
            # Get query embedding
            start_time = time.time()
            inputs = processor(images=img, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                v = model.get_image_features(**inputs)
                v = v / v.norm(dim=-1, keepdim=True)
                v = v.cpu().numpy().astype("float32")
            
            if v.ndim == 1:
                v = v.reshape(1, -1)
            
            # Search
            D, I = index.search(v, k_max)
            search_time = time.time() - start_time
            results["search_times"].append(search_time)
            
            # Get retrieved images
            retrieved = [ids[i] for i in I[0].tolist() if 0 <= i < len(ids)]
            
            # For image-to-image, relevant images are semantically similar
            # We'll use similarity threshold > 0.7 as "relevant" for evaluation
            # In practice, you'd have ground truth labels
            relevant = [img_id]  # Self-match is always relevant
            if exclude_self:
                retrieved = [r for r in retrieved if r != img_id]
            
            # Compute metrics
            for k in K_VALUES:
                if k <= k_max:
                    results["precision@k"][k].append(precision_at_k(relevant, retrieved, k))
                    results["recall@k"][k].append(recall_at_k(relevant, retrieved, k))
            
            results["mrr"].append(mean_reciprocal_rank(relevant, retrieved))
            results["map"].append(average_precision(relevant, retrieved))
            
            # Store similarity scores
            scores = D[0].tolist()
            results["similarity_scores"].extend(scores[:k_max])
            
        except Exception as e:
            print(f"⚠️ Error processing {img_id}: {e}")
            continue
    
    return results


def evaluate_text_to_image(
    model, processor, device, index, ids,
    test_queries: List[Tuple[str, List[str]]],  # (query_text, relevant_image_ids)
    k_max: int = 20
) -> Dict:
    """Evaluate text-to-image search."""
    print(f"\n🔍 Evaluating Text-to-Image Search on {len(test_queries)} queries...")
    
    results = {
        "precision@k": {k: [] for k in K_VALUES if k <= k_max},
        "recall@k": {k: [] for k in K_VALUES if k <= k_max},
        "mrr": [],
        "map": [],
        "similarity_scores": [],
        "search_times": [],
    }
    
    for query_text, relevant_ids in tqdm(test_queries, desc="Processing queries"):
        try:
            start_time = time.time()
            inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                t = model.get_text_features(**inputs)
                t = t / t.norm(dim=-1, keepdim=True)
                t = t.cpu().numpy().astype("float32")
            
            D, I = index.search(t, k_max)
            search_time = time.time() - start_time
            results["search_times"].append(search_time)
            
            retrieved = [ids[i] for i in I[0].tolist() if 0 <= i < len(ids)]
            
            # Compute metrics
            for k in K_VALUES:
                if k <= k_max:
                    results["precision@k"][k].append(precision_at_k(relevant_ids, retrieved, k))
                    results["recall@k"][k].append(recall_at_k(relevant_ids, retrieved, k))
            
            results["mrr"].append(mean_reciprocal_rank(relevant_ids, retrieved))
            results["map"].append(average_precision(relevant_ids, retrieved))
            
            scores = D[0].tolist()
            results["similarity_scores"].extend(scores[:k_max])
            
        except Exception as e:
            print(f"⚠️ Error processing query: {e}")
            continue
    
    return results


def compute_statistics(results: Dict) -> Dict:
    """Compute summary statistics from results."""
    stats = {}
    
    # Precision@K
    stats["precision@k"] = {
        k: {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "median": np.median(scores),
            "min": np.min(scores),
            "max": np.max(scores),
        }
        for k, scores in results["precision@k"].items()
    }
    
    # Recall@K
    stats["recall@k"] = {
        k: {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "median": np.median(scores),
        }
        for k, scores in results["recall@k"].items()
    }
    
    # MRR
    stats["mrr"] = {
        "mean": np.mean(results["mrr"]),
        "std": np.std(results["mrr"]),
        "median": np.median(results["mrr"]),
    }
    
    # MAP
    stats["map"] = {
        "mean": np.mean(results["map"]),
        "std": np.std(results["map"]),
        "median": np.median(results["map"]),
    }
    
    # Similarity scores
    stats["similarity"] = {
        "mean": np.mean(results["similarity_scores"]),
        "std": np.std(results["similarity_scores"]),
        "median": np.median(results["similarity_scores"]),
        "min": np.min(results["similarity_scores"]),
        "max": np.max(results["similarity_scores"]),
        "q25": np.percentile(results["similarity_scores"], 25),
        "q75": np.percentile(results["similarity_scores"], 75),
    }
    
    # Search time
    stats["search_time"] = {
        "mean": np.mean(results["search_times"]) * 1000,  # Convert to ms
        "std": np.std(results["search_times"]) * 1000,
        "median": np.median(results["search_times"]) * 1000,
        "min": np.min(results["search_times"]) * 1000,
        "max": np.max(results["search_times"]) * 1000,
    }
    
    return stats


def get_index_statistics() -> Dict:
    """Get statistics about the FAISS indexes."""
    stats = {}
    
    for name, path in [("image", INDEX_IMG_PATH), ("mixed", INDEX_MIX_PATH)]:
        if path.exists():
            index = faiss.read_index(str(path))
            stats[name] = {
                "num_vectors": index.ntotal,
                "vector_dim": index.d,
                "index_type": type(index).__name__,
                "is_trained": index.is_trained,
                "file_size_mb": path.stat().st_size / (1024 * 1024),
            }
    
    return stats


def create_test_queries_from_captions(captions: Dict, num_queries: int = 50) -> List[Tuple[str, List[str]]]:
    """Create test queries from captions."""
    queries = []
    items = list(captions.items())[:num_queries]
    
    for img_id, cap_list in items:
        if cap_list:
            # Use first caption as query, image ID as relevant
            queries.append((cap_list[0], [img_id]))
    
    return queries


def create_test_images_from_dataset(num_images: int = 100) -> List[str]:
    """Create test image list from dataset."""
    ids = ID_LIST_PATH.read_text().strip().split("\n")
    # Randomly sample
    np.random.seed(42)
    selected = np.random.choice(ids, min(num_images, len(ids)), replace=False)
    return selected.tolist()


def generate_report(stats: Dict, index_stats: Dict, mode: str, output_dir: Path):
    """Generate evaluation report and visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Text report
    report_lines = [
        f"# Evaluation Report: {mode.upper()} Search",
        "=" * 60,
        "",
        "## Accuracy Metrics",
        "",
    ]
    
    # Precision@K
    report_lines.append("### Precision@K")
    for k in sorted(stats["precision@k"].keys()):
        p = stats["precision@k"][k]
        report_lines.append(f"  P@{k}: {p['mean']:.4f} ± {p['std']:.4f} (median: {p['median']:.4f})")
    
    # Recall@K
    report_lines.append("\n### Recall@K")
    for k in sorted(stats["recall@k"].keys()):
        r = stats["recall@k"][k]
        report_lines.append(f"  R@{k}: {r['mean']:.4f} ± {r['std']:.4f} (median: {r['median']:.4f})")
    
    # MRR and MAP
    report_lines.append("\n### Overall Metrics")
    report_lines.append(f"  Mean Reciprocal Rank (MRR): {stats['mrr']['mean']:.4f} ± {stats['mrr']['std']:.4f}")
    report_lines.append(f"  Mean Average Precision (MAP): {stats['map']['mean']:.4f} ± {stats['map']['std']:.4f}")
    
    # Similarity scores
    report_lines.append("\n### Similarity Score Statistics")
    sim = stats["similarity"]
    report_lines.append(f"  Mean: {sim['mean']:.4f}")
    report_lines.append(f"  Median: {sim['median']:.4f}")
    report_lines.append(f"  Std: {sim['std']:.4f}")
    report_lines.append(f"  Range: [{sim['min']:.4f}, {sim['max']:.4f}]")
    report_lines.append(f"  IQR: [{sim['q25']:.4f}, {sim['q75']:.4f}]")
    
    # Search time
    report_lines.append("\n### Performance")
    st = stats["search_time"]
    report_lines.append(f"  Mean search time: {st['mean']:.2f} ms")
    report_lines.append(f"  Median search time: {st['median']:.2f} ms")
    report_lines.append(f"  Range: [{st['min']:.2f}, {st['max']:.2f}] ms")
    
    # Index statistics
    report_lines.append("\n## Index Statistics")
    for name, idx_stats in index_stats.items():
        report_lines.append(f"\n### {name.capitalize()} Index")
        report_lines.append(f"  Vectors: {idx_stats['num_vectors']:,}")
        report_lines.append(f"  Dimension: {idx_stats['vector_dim']}")
        report_lines.append(f"  Type: {idx_stats['index_type']}")
        report_lines.append(f"  Size: {idx_stats['file_size_mb']:.2f} MB")
    
    # Save report
    report_path = output_dir / f"evaluation_report_{mode}.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"\n✅ Report saved to: {report_path}")
    
    # Create visualizations
    create_visualizations(stats, output_dir, mode)
    
    # Save JSON for programmatic access
    json_path = output_dir / f"evaluation_stats_{mode}.json"
    import json
    # Convert numpy types to native Python types
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    json_path.write_text(json.dumps(convert_numpy(stats), indent=2))
    print(f"✅ JSON stats saved to: {json_path}")


def create_visualizations(stats: Dict, output_dir: Path, mode: str):
    """Create visualization plots."""
    sns.set_style("whitegrid")
    
    # Precision@K plot
    plt.figure(figsize=(10, 6))
    k_values = sorted(stats["precision@k"].keys())
    precisions = [stats["precision@k"][k]["mean"] for k in k_values]
    plt.plot(k_values, precisions, marker='o', linewidth=2, markersize=8)
    plt.xlabel("K (Number of Results)", fontsize=12)
    plt.ylabel("Precision@K", fontsize=12)
    plt.title(f"Precision@K for {mode.upper()} Search", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"precision_at_k_{mode}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Recall@K plot
    plt.figure(figsize=(10, 6))
    recalls = [stats["recall@k"][k]["mean"] for k in k_values]
    plt.plot(k_values, recalls, marker='s', linewidth=2, markersize=8, color='orange')
    plt.xlabel("K (Number of Results)", fontsize=12)
    plt.ylabel("Recall@K", fontsize=12)
    plt.title(f"Recall@K for {mode.upper()} Search", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"recall_at_k_{mode}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate image search system metrics")
    parser.add_argument("--mode", choices=["image", "text", "both"], default="both",
                       help="Evaluation mode")
    parser.add_argument("--num-queries", type=int, default=100,
                       help="Number of test queries")
    parser.add_argument("--k-max", type=int, default=20,
                       help="Maximum K for evaluation")
    parser.add_argument("--output-dir", type=str, default="data/evaluation",
                       help="Output directory for reports")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load index statistics
    print("📊 Gathering index statistics...")
    index_stats = get_index_statistics()
    
    all_results = {}
    
    # Evaluate Image-to-Image
    if args.mode in ["image", "both"]:
        model, processor, device, index, ids = load_model_and_index("image")
        test_images = create_test_images_from_dataset(args.num_queries)
        results = evaluate_image_to_image(
            model, processor, device, index, ids, test_images, args.k_max
        )
        stats = compute_statistics(results)
        all_results["image"] = stats
        generate_report(stats, index_stats, "image", output_dir)
    
    # Evaluate Text-to-Image
    if args.mode in ["text", "both"]:
        model, processor, device, index, ids = load_model_and_index("text")
        captions = load_captions(CAP_PATH)
        test_queries = create_test_queries_from_captions(captions, args.num_queries)
        results = evaluate_text_to_image(
            model, processor, device, index, ids, test_queries, args.k_max
        )
        stats = compute_statistics(results)
        all_results["text"] = stats
        generate_report(stats, index_stats, "text", output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 EVALUATION SUMMARY")
    print("=" * 60)
    
    for mode, stats in all_results.items():
        print(f"\n{mode.upper()} Search:")
        print(f"  Precision@10: {stats['precision@k'][10]['mean']:.4f}")
        print(f"  Recall@10: {stats['recall@k'][10]['mean']:.4f}")
        print(f"  MAP: {stats['map']['mean']:.4f}")
        print(f"  MRR: {stats['mrr']['mean']:.4f}")
        print(f"  Avg Search Time: {stats['search_time']['mean']:.2f} ms")
    
    print(f"\n✅ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()

