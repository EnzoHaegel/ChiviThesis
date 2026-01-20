import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Config
# Note: Run from Project Root (ChiviThesis/)
INPUT_CSV = Path("Data/sec_10q/word_impact_analysis.csv")
OUTPUT_DIR = Path("Data/sec_10q/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_plots():
    print("Loading data...")
    df = pd.read_csv(INPUT_CSV)
    
    # Sort just in case
    df = df.sort_values(by='coefficient', ascending=False)
    
    top_pos = df.head(20)
    top_neg = df.tail(20).sort_values(by='coefficient', ascending=True) # Sort for chart
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Top 20 Positive Impact Words
    plt.figure(figsize=(10, 8))
    sns.barplot(x="coefficient", y="word", data=top_pos, hue="word", palette="Greens_r", legend=False)
    plt.title("Top 20 Words with Positive Impact on 5-Day Returns", fontsize=15)
    plt.xlabel("Ridge Regression Coefficient", fontsize=12)
    plt.ylabel("Word", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_positive_impact.png", dpi=300)
    plt.close()
    
    # Plot 2: Top 20 Negative Impact Words
    plt.figure(figsize=(10, 8))
    sns.barplot(x="coefficient", y="word", data=top_neg, hue="word", palette="Reds_r", legend=False)
    plt.title("Top 20 Words with Negative Impact on 5-Day Returns", fontsize=15)
    plt.xlabel("Ridge Regression Coefficient", fontsize=12)
    plt.ylabel("Word", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_negative_impact.png", dpi=300)
    plt.close()
    
    print(f"Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_plots()
