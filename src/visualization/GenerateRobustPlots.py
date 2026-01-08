import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Config
# Note: Run from Project Root (ChiviThesis/)
INPUT_DIR = Path("Data/sec_10q/robust_analysis")
OUTPUT_DIR = Path("Data/sec_10q/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_horizon(horizon_name, filename):
    print(f"\nProcessing {horizon_name} ({filename})...")
    df = pd.read_csv(INPUT_DIR / filename)
    
    # Files are already sorted by coefficient descending
    top_pos = df.head(15)
    top_neg = df.tail(15).sort_values(by='coefficient', ascending=True)
    
    # Print for report text
    print(f"Top 5 Positive ({horizon_name}): {top_pos['word'].tolist()[:5]}")
    print(f"Top 5 Negative ({horizon_name}): {top_neg['word'].tolist()[:5]}")

    # Set style
    sns.set_theme(style="whitegrid")
    
    # Plot Positive
    plt.figure(figsize=(10, 6))
    sns.barplot(x="coefficient", y="word", data=top_pos, hue="word", palette="Greens_r", legend=False)
    plt.title(f"Top Words Increasing Alpha ({horizon_name})", fontsize=14)
    plt.xlabel("Ridge Coefficient (Impact on Alpha)", fontsize=10)
    plt.ylabel(None)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"robust_pos_{horizon_name}.png", dpi=150)
    plt.close()
    
    # Plot Negative
    plt.figure(figsize=(10, 6))
    sns.barplot(x="coefficient", y="word", data=top_neg, hue="word", palette="Reds_r", legend=False)
    plt.title(f"Top Words Decreasing Alpha ({horizon_name})", fontsize=14)
    plt.xlabel("Ridge Coefficient (Impact on Alpha)", fontsize=10)
    plt.ylabel(None)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"robust_neg_{horizon_name}.png", dpi=150)
    plt.close()

def main():
    plot_horizon("T+1", "impact_words_T1.csv")
    plot_horizon("T+10", "impact_words_T10.csv")
    print(f"\nPlots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
