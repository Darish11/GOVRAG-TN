import pandas as pd
import matplotlib.pyplot as plt
import os

EDA_CSV = r"E:\SRM\Project\rag_data\eda\go_corpus_stats.csv"
OUTPUT_DIR = r"E:\SRM\Project\rag_data\eda\plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(EDA_CSV)

plt.figure(figsize=(8, 5))
plt.hist(df["num_pages"], bins=50)
plt.xlabel("Number of Pages per GO")
plt.ylabel("Number of GOs")
plt.title("Distribution of Pages per Government Order")

plt.axvline(3, linestyle="--", label="Median (3 pages)")
plt.axvline(10, linestyle="--", label="90th Percentile (~10 pages)")
plt.axvline(16, linestyle="--", label="95th Percentile (~16 pages)")

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pages_per_go.png"))
plt.close()


plt.figure(figsize=(8, 5))
plt.hist(df["total_chars"], bins=60)
plt.xlabel("Total Characters per GO")
plt.ylabel("Number of GOs")
plt.title("Text Size Distribution of Government Orders")

plt.axvline(5703, linestyle="--", label="Median (~5.7k chars)")
plt.axvline(15332, linestyle="--", label="90th Percentile (~15k chars)")
plt.axvline(26466, linestyle="--", label="95th Percentile (~26k chars)")

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "total_chars_per_go.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.hist(df["clause_count"], bins=50)
plt.xlabel("Number of Clauses per GO")
plt.ylabel("Number of GOs")
plt.title("Clause Count Distribution in Government Orders")

plt.axvline(7, linestyle="--", label="Median (~7 clauses)")
plt.axvline(19, linestyle="--", label="90th Percentile (~19 clauses)")
plt.axvline(29, linestyle="--", label="95th Percentile (~29 clauses)")

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "clause_count_distribution.png"))
plt.close()


plt.figure(figsize=(8, 5))
plt.hist(df["tamil_ratio"], bins=50)
plt.xlabel("Tamil Character Ratio")
plt.ylabel("Number of GOs")
plt.title("Tamil vs English Content Distribution")

plt.axvline(0.024, linestyle="--", label="Mean (~2.4%)")
plt.axvline(0.088, linestyle="--", label="95th Percentile (~8.8%)")
plt.axvline(0.5, linestyle="--", label="Tamil-dominant GOs")

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tamil_ratio_distribution.png"))
plt.close()


plt.figure(figsize=(8, 5))
plt.scatter(df["num_pages"], df["clause_count"], alpha=0.6)
plt.xlabel("Number of Pages")
plt.ylabel("Number of Clauses")
plt.title("Relationship Between Pages and Clause Count")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pages_vs_clauses.png"))
plt.close()

