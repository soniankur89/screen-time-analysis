"""
Mini Project - Data Analysis & Visualization
Screen Time Analysis Using Python
Dataset: Analyzing Screen Time (Kaggle, by ruchi798)
Tools: Pandas, NumPy, Matplotlib, Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------------
# 1. LOAD THE DATASET
# ---------------------------------------------------------------
df = pd.read_csv("Screentime-App-Details.csv")

print("=" * 60)
print("STEP 1: RAW DATASET")
print("=" * 60)
print("Shape (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn data types:")
print(df.dtypes)

# ---------------------------------------------------------------
# 2. DATA CLEANING WITH PANDAS
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: DATA CLEANING")
print("=" * 60)

# 2a. Check missing values
print("\nMissing values per column (before cleaning):")
print(df.isnull().sum())

# 2b. Check duplicate records
dup_count = df.duplicated().sum()
print("\nDuplicate rows found:", dup_count)
df = df.drop_duplicates()
print("Duplicate rows remaining after removal:", df.duplicated().sum())

# 2c. Fix incorrect data: hidden carriage-return characters in App names
print("\nUnique App labels BEFORE fixing:", list(df["App"].unique()))
df["App"] = df["App"].str.strip()          # remove \r and spaces
df["App"] = df["App"].replace({"Whatsapp": "WhatsApp"})  # standard spelling
print("Unique App labels AFTER fixing: ", list(df["App"].unique()))

# 2d. Convert Date column from text to datetime
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df = df.sort_values("Date").reset_index(drop=True)

# 2e. Verify numeric columns
for col in ["Usage", "Notifications", "Times opened"]:
    df[col] = pd.to_numeric(df[col], errors="raise")

print("\nMissing values per column (after cleaning):", df.isnull().sum().sum())
print("Cleaned shape:", df.shape)
df.to_csv("screentime_cleaned.csv", index=False)
print("Cleaned data saved to screentime_cleaned.csv")

# ---------------------------------------------------------------
# 3. DATA ANALYSIS WITH NUMPY
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: NUMERICAL ANALYSIS WITH NUMPY")
print("=" * 60)

usage = df["Usage"].to_numpy()
notif = df["Notifications"].to_numpy()
opens = df["Times opened"].to_numpy()

def stats(name, arr):
    print(f"\n{name} (minutes/count):")
    print(f"  Average (mean)      : {np.mean(arr):.2f}")
    print(f"  Minimum             : {np.min(arr)}")
    print(f"  Maximum             : {np.max(arr)}")
    print(f"  Total               : {np.sum(arr)}")
    print(f"  Standard deviation  : {np.std(arr):.2f}")

stats("Usage", usage)
stats("Notifications", notif)
stats("Times opened", opens)

print("\nCorrelation coefficients:")
print(f"  Usage vs Notifications : {np.corrcoef(usage, notif)[0,1]:.3f}")
print(f"  Usage vs Times opened  : {np.corrcoef(usage, opens)[0,1]:.3f}")
print(f"  Notifications vs Opens : {np.corrcoef(notif, opens)[0,1]:.3f}")

print("\nPer-app summary (Pandas groupby):")
summary = df.groupby("App")[["Usage", "Notifications", "Times opened"]].agg(["mean", "sum", "max"])
print(summary.round(2))

# ---------------------------------------------------------------
# 4. DATA VISUALIZATION
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: GENERATING GRAPHS")
print("=" * 60)

# Graph 1: Total usage per app - Bar chart
plt.figure(figsize=(8, 5))
totals = df.groupby("App")["Usage"].sum().sort_values(ascending=False)
plt.bar(totals.index, totals.values, color=["#E1306C", "#25D366"])
plt.title("Total Screen Time Usage per App (26 Aug - 21 Sep 2022)")
plt.xlabel("App")
plt.ylabel("Total Usage (minutes)")
for i, v in enumerate(totals.values):
    plt.text(i, v + 20, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("graphs/graph1_usage_per_app_bar.png")
plt.close()
print("Saved graph1_usage_per_app_bar.png")

# Graph 2: Daily total screen time - Line chart
daily = df.groupby("Date")["Usage"].sum()
plt.figure(figsize=(10, 5))
plt.plot(daily.index, daily.values, marker="o", color="#1f77b4", linewidth=2)
plt.title("Daily Total Screen Time Trend")
plt.xlabel("Date")
plt.ylabel("Total Usage (minutes)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graphs/graph2_daily_trend_line.png")
plt.close()
print("Saved graph2_daily_trend_line.png")

# Graph 3: Usage distribution - Histogram
plt.figure(figsize=(8, 5))
plt.hist(df["Usage"], bins=12, color="#9467bd", edgecolor="black")
plt.title("Distribution of App Usage (minutes per day)")
plt.xlabel("Usage (minutes)")
plt.ylabel("Frequency (number of days)")
plt.tight_layout()
plt.savefig("graphs/graph3_usage_histogram.png")
plt.close()
print("Saved graph3_usage_histogram.png")

# Graph 4: Notifications vs Usage - Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Notifications", y="Usage", hue="App", s=80)
plt.title("Notifications vs Usage")
plt.xlabel("Notifications received")
plt.ylabel("Usage (minutes)")
plt.tight_layout()
plt.savefig("graphs/graph4_notifications_vs_usage_scatter.png")
plt.close()
print("Saved graph4_notifications_vs_usage_scatter.png")

# Graph 5: Correlation matrix - Heatmap
plt.figure(figsize=(7, 5))
corr = df[["Usage", "Notifications", "Times opened"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
plt.title("Correlation Matrix: Usage, Notifications, Times Opened")
plt.tight_layout()
plt.savefig("graphs/graph5_correlation_heatmap.png")
plt.close()
print("Saved graph5_correlation_heatmap.png")

print("\nAnalysis complete.")
