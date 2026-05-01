"""
TruthLens — Data Loader
========================
Auto-downloads and standardizes multiple fake news datasets:
  - ISOT (Kaggle): Full news articles, ~44K samples
  - LIAR (HuggingFace): Short claims/statements, ~12.8K samples

Each dataset is standardized to: {text, label (0=Real, 1=Fake), source_dataset}
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def download_isot():
    """
    Download the ISOT Fake News dataset from Kaggle.
    Contains ~23K fake and ~21K real full-text news articles.
    """
    isot_dir = os.path.join(config.RAW_DIR, "isot")
    fake_path = os.path.join(isot_dir, "Fake.csv")
    true_path = os.path.join(isot_dir, "True.csv")

    if os.path.exists(fake_path) and os.path.exists(true_path):
        print("[DATA] ISOT dataset already exists. Skipping download.")
        return isot_dir

    print("[DATA] Downloading ISOT dataset from Kaggle...")
    os.makedirs(isot_dir, exist_ok=True)

    try:
        import kagglehub
        path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
        print(f"[DATA] Kaggle download path: {path}")

        # Copy files to our raw dir
        import shutil
        for fname in ["Fake.csv", "True.csv"]:
            src = os.path.join(path, fname)
            # Search recursively if not at root
            if not os.path.exists(src):
                for root, dirs, files in os.walk(path):
                    if fname in files:
                        src = os.path.join(root, fname)
                        break
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(isot_dir, fname))
            else:
                raise FileNotFoundError(f"Could not find {fname} in download.")

        print("[DATA] ISOT dataset downloaded successfully.")

    except Exception as e:
        print(f"[DATA] Kaggle download failed: {e}")
        print("[DATA] Please download manually from:")
        print("       https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print(f"       Extract Fake.csv and True.csv into: {isot_dir}")
        raise

    return isot_dir


def download_liar():
    """
    Download the LIAR dataset via HuggingFace datasets library.
    Contains ~12.8K short political claims with 6-way labels.
    """
    liar_dir = os.path.join(config.RAW_DIR, "liar")
    liar_path = os.path.join(liar_dir, "liar_all.csv")

    if os.path.exists(liar_path):
        print("[DATA] LIAR dataset already exists. Skipping download.")
        return liar_dir

    print("[DATA] Downloading LIAR dataset from HuggingFace...")
    os.makedirs(liar_dir, exist_ok=True)

    try:
        from datasets import load_dataset

        ds = load_dataset("liar")

        # Combine train + val + test
        all_rows = []
        for split_name in ["train", "validation", "test"]:
            if split_name in ds:
                split_df = ds[split_name].to_pandas()
                split_df["split"] = split_name
                all_rows.append(split_df)

        liar_df = pd.concat(all_rows, ignore_index=True)
        liar_df.to_csv(liar_path, index=False)
        print(f"[DATA] LIAR dataset saved: {len(liar_df)} rows.")

    except Exception as e:
        print(f"[DATA] HuggingFace download failed: {e}")
        print("[DATA] Trying direct download...")

        try:
            import requests, zipfile, io

            url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                zf.extractall(liar_dir)

            # Parse TSV files
            all_rows = []
            for fname in ["train.tsv", "valid.tsv", "test.tsv"]:
                fpath = os.path.join(liar_dir, fname)
                if os.path.exists(fpath):
                    cols = ["id", "label", "statement", "subject", "speaker",
                            "job_title", "state_info", "party",
                            "barely_true", "false", "half_true",
                            "mostly_true", "pants_on_fire", "context"]
                    df = pd.read_csv(fpath, sep="\t", header=None, names=cols)
                    all_rows.append(df)

            liar_df = pd.concat(all_rows, ignore_index=True)
            liar_df.to_csv(liar_path, index=False)
            print(f"[DATA] LIAR dataset saved: {len(liar_df)} rows.")

        except Exception as e2:
            print(f"[DATA] Direct download also failed: {e2}")
            raise

    return liar_dir


def load_isot(isot_dir=None):
    """
    Load ISOT dataset and standardize to {text, label, source_dataset}.

    Returns:
        pd.DataFrame with columns: text, label, source_dataset
    """
    if isot_dir is None:
        isot_dir = os.path.join(config.RAW_DIR, "isot")

    fake_path = os.path.join(isot_dir, "Fake.csv")
    true_path = os.path.join(isot_dir, "True.csv")

    print("[DATA] Loading ISOT dataset...")

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Combine title + text for richer content
    fake_df["text"] = fake_df["title"].fillna("") + " " + fake_df["text"].fillna("")
    true_df["text"] = true_df["title"].fillna("") + " " + true_df["text"].fillna("")

    fake_df["label"] = 1   # Fake
    true_df["label"] = 0   # Real

    # Keep only text and label
    fake_df = fake_df[["text", "label"]].copy()
    true_df = true_df[["text", "label"]].copy()

    isot_df = pd.concat([fake_df, true_df], ignore_index=True)
    isot_df["source_dataset"] = "isot"

    # Shuffle
    isot_df = isot_df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)

    # Remove duplicate articles (news syndication creates exact copies)
    before_dedup = len(isot_df)
    isot_df = isot_df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    dedup_removed = before_dedup - len(isot_df)
    if dedup_removed > 0:
        print(f"[DATA] Removed {dedup_removed} duplicate articles from ISOT.")

    # Remove empty / very short texts
    isot_df = isot_df[isot_df["text"].str.strip().str.len() > 20].reset_index(drop=True)

    print(f"[DATA] ISOT loaded: {len(isot_df)} articles "
          f"(Fake: {(isot_df.label==1).sum()}, Real: {(isot_df.label==0).sum()})")

    return isot_df


def load_liar(liar_dir=None):
    """
    Load LIAR dataset and standardize to {text, label, source_dataset}.
    Maps 6-class labels to binary: {pants-fire, false, barely-true} → Fake(1),
                                     {half-true, mostly-true, true} → Real(0)

    Returns:
        pd.DataFrame with columns: text, label, source_dataset
    """
    if liar_dir is None:
        liar_dir = os.path.join(config.RAW_DIR, "liar")

    liar_path = os.path.join(liar_dir, "liar_all.csv")
    print("[DATA] Loading LIAR dataset...")

    liar_df = pd.read_csv(liar_path)

    print(f"[DATA] LIAR raw CSV: {len(liar_df)} rows, columns: {liar_df.columns.tolist()}")

    # Identify the text column (could be 'statement' or 'statement_text')
    text_col = None
    for col_name in ["statement", "statement_text", "text"]:
        if col_name in liar_df.columns:
            text_col = col_name
            break

    if text_col is None:
        raise ValueError(f"Cannot find text column in LIAR. Columns: {liar_df.columns.tolist()}")

    # Map 6-class labels to binary
    if "label" not in liar_df.columns:
        raise ValueError(f"Cannot find label column in LIAR. Columns: {liar_df.columns.tolist()}")

    label_col = liar_df["label"]
    print(f"[DATA] LIAR label dtype: {label_col.dtype}, unique values: {label_col.unique()[:10]}")

    # CRITICAL: Convert ALL label values to lowercase strings first.
    # pandas read_csv may parse 'false'/'true' as Python booleans, breaking .str ops.
    label_strings = label_col.astype(str).str.lower().str.strip()
    print(f"[DATA] LIAR label strings sample: {label_strings.head(10).tolist()}")

    fake_labels = {"pants-fire", "false", "barely-true", "pants-on-fire"}
    real_labels = {"half-true", "mostly-true", "true"}

    binary_labels = label_strings.map(
        lambda x: 1 if x in fake_labels else (0 if x in real_labels else np.nan)
    )

    nan_count = binary_labels.isna().sum()
    if nan_count > 0:
        unmapped = label_strings[binary_labels.isna()].unique()
        print(f"[DATA] WARNING: {nan_count} labels could not be mapped: {unmapped[:10]}")

    result_df = pd.DataFrame({
        "text": liar_df[text_col].values,
        "label": binary_labels.values,
        "source_dataset": "liar",
    })

    # Drop rows with NaN labels or empty text
    result_df = result_df.dropna(subset=["label", "text"]).reset_index(drop=True)
    result_df["label"] = result_df["label"].astype(int)
    result_df = result_df[result_df["text"].astype(str).str.strip().str.len() > 5].reset_index(drop=True)

    print(f"[DATA] LIAR loaded: {len(result_df)} statements "
          f"(Fake: {(result_df.label==1).sum()}, Real: {(result_df.label==0).sum()})")

    return result_df


def get_datasets(download=True):
    """
    Download (if needed) and load all datasets.

    Returns:
        dict: {
            'isot': pd.DataFrame,
            'liar': pd.DataFrame,
            'combined': pd.DataFrame (ISOT + LIAR merged)
        }
    """
    if download:
        download_isot()
        download_liar()

    isot_df = load_isot()
    liar_df = load_liar()

    # Combined dataset for mixed-domain training
    combined_df = pd.concat([isot_df, liar_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)

    print(f"\n[DATA] Combined dataset: {len(combined_df)} total samples")
    print(f"       ISOT: {len(isot_df)} | LIAR: {len(liar_df)}")

    return {
        "isot": isot_df,
        "liar": liar_df,
        "combined": combined_df,
    }


def split_dataset(df, test_size=None, stratify=True):
    """
    Split dataset into train and test sets.

    Returns:
        (train_df, test_df)
    """
    if test_size is None:
        test_size = config.TEST_SIZE

    stratify_col = df["label"] if stratify else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=config.RANDOM_STATE,
        stratify=stratify_col,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"[DATA] Split: Train={len(train_df)}, Test={len(test_df)}")

    return train_df, test_df


if __name__ == "__main__":
    datasets = get_datasets(download=True)
    for name, df in datasets.items():
        print(f"\n--- {name.upper()} ---")
        print(f"Shape: {df.shape}")
        print(f"Labels:\n{df['label'].value_counts()}")
        print(f"Sample text: {df['text'].iloc[0][:200]}...")
