"""
Quick Metrics Summary - For Presentation
Generates a concise summary table of key metrics.
"""

import json
from pathlib import Path
import pandas as pd

def load_evaluation_results(eval_dir: Path = Path("data/evaluation")):
    """Load evaluation results and create summary table."""
    results = {}
    
    for mode in ["image", "text"]:
        json_path = eval_dir / f"evaluation_stats_{mode}.json"
        if json_path.exists():
            with open(json_path) as f:
                results[mode] = json.load(f)
    
    return results


def create_summary_table(results: dict) -> pd.DataFrame:
    """Create a summary table for presentation."""
    rows = []
    
    for mode, stats in results.items():
        mode_name = "Image-to-Image" if mode == "image" else "Text-to-Image"
        
        # Key metrics
        p10 = stats["precision@k"]["10"]["mean"]
        r10 = stats["recall@k"]["10"]["mean"]
        map_score = stats["map"]["mean"]
        mrr = stats["mrr"]["mean"]
        search_time = stats["search_time"]["mean"]
        
        rows.append({
            "Mode": mode_name,
            "Precision@10": f"{p10:.3f}",
            "Recall@10": f"{r10:.3f}",
            "MAP": f"{map_score:.3f}",
            "MRR": f"{mrr:.3f}",
            "Search Time (ms)": f"{search_time:.1f}",
        })
    
    return pd.DataFrame(rows)


def print_presentation_summary(results: dict):
    """Print presentation-ready summary."""
    print("\n" + "=" * 70)
    print("📊 PRESENTATION-READY METRICS SUMMARY")
    print("=" * 70)
    
    for mode, stats in results.items():
        mode_name = "Image-to-Image" if mode == "image" else "Text-to-Image"
        print(f"\n{mode_name} Search:")
        print("-" * 70)
        
        # Accuracy metrics
        print("Accuracy Metrics:")
        for k in [1, 5, 10, 20]:
            if str(k) in stats["precision@k"]:
                p = stats["precision@k"][str(k)]["mean"]
                r = stats["recall@k"][str(k)]["mean"]
                print(f"  K={k:2d}: Precision={p:.3f}, Recall={r:.3f}")
        
        print(f"\nOverall Performance:")
        print(f"  Mean Average Precision (MAP): {stats['map']['mean']:.4f}")
        print(f"  Mean Reciprocal Rank (MRR):   {stats['mrr']['mean']:.4f}")
        
        # Similarity
        sim = stats["similarity"]
        print(f"\nSimilarity Scores:")
        print(f"  Mean:   {sim['mean']:.3f}")
        print(f"  Median: {sim['median']:.3f}")
        print(f"  Range:  [{sim['min']:.3f}, {sim['max']:.3f}]")
        
        # Performance
        st = stats["search_time"]
        print(f"\nSearch Performance:")
        print(f"  Average: {st['mean']:.1f} ms")
        print(f"  Median:  {st['median']:.1f} ms")
        print(f"  Range:   [{st['min']:.1f}, {st['max']:.1f}] ms")
    
    print("\n" + "=" * 70)
    print("💡 Copy the metrics above into your presentation slides!")
    print("=" * 70 + "\n")


def main():
    eval_dir = Path("data/evaluation")
    
    if not eval_dir.exists():
        print("❌ Evaluation directory not found!")
        print("   Run: python scripts/04_evaluate_metrics.py first")
        return
    
    results = load_evaluation_results(eval_dir)
    
    if not results:
        print("❌ No evaluation results found!")
        print("   Run: python scripts/04_evaluate_metrics.py first")
        return
    
    # Print summary
    print_presentation_summary(results)
    
    # Create and save table
    df = create_summary_table(results)
    print("\n📋 Summary Table:")
    print(df.to_string(index=False))
    
    # Save as CSV
    csv_path = eval_dir / "summary_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Summary table saved to: {csv_path}")
    
    # Save as markdown
    md_path = eval_dir / "summary_table.md"
    md_path.write_text(df.to_markdown(index=False))
    print(f"✅ Markdown table saved to: {md_path}")


if __name__ == "__main__":
    main()

