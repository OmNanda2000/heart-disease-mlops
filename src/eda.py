
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PLOT_DIR = ROOT / "screenshots" / "eda"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Downloading Heart Disease dataset from UCI...")

    dataset = fetch_ucirepo(id=45)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    target_column = targets.columns[0]

    df = pd.concat([features, targets], axis=1)

    # Save original downloaded data
    df.to_csv(DATA_DIR / "heart_disease_raw.csv", index=False)

    print("\nOriginal dataset shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    # Convert invalid numeric entries to missing values
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove records with unknown target
    df = df.dropna(subset=[target_column]).copy()

    # Convert original diagnosis (0 to 4) to binary target
    df["target"] = (df[target_column] > 0).astype(int)
    df = df.drop(columns=[target_column])

    # Remove exact duplicate records
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    print("\nDuplicate rows removed:", duplicate_count)
    print("Final dataset shape:", df.shape)

    # Report missing values
    missing = df.isnull().sum().sort_values(ascending=False)
    missing.to_csv(DATA_DIR / "missing_values_summary.csv")

    print("\nMissing values:")
    print(missing[missing > 0])

    print("\nBinary target distribution:")
    print(df["target"].value_counts().sort_index())

    # Save cleaned dataset; remaining missing features will be
    # imputed within the model pipeline after splitting the data.
    df.to_csv(DATA_DIR / "heart_disease_clean.csv", index=False)

    sns.set_theme(style="whitegrid")

    # Plot 1: Class balance
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="target")
    plt.title("Heart Disease Class Balance")
    plt.xlabel("0 = No disease, 1 = Disease")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "class_balance.png", dpi=200)
    plt.close()

    # Plot 2: Feature histograms
    df.drop(columns=["target"]).hist(
        figsize=(15, 12), bins=20, edgecolor="black"
    )
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "feature_histograms.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # Plot 3: Correlation heatmap
    plt.figure(figsize=(13, 10))
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
    )
    plt.title("Feature and Target Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()

    # Summary statistics
    df.describe().T.to_csv(DATA_DIR / "descriptive_statistics.csv")

    print("\nTASK 1 COMPLETED SUCCESSFULLY!")
    print("Cleaned dataset:", DATA_DIR / "heart_disease_clean.csv")
    print("EDA plots:", PLOT_DIR)


if __name__ == "__main__":
    main()
