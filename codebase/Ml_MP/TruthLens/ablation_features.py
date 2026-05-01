"""
TruthLens — Feature Pipeline Ablation Study
=============================================
Measures the contribution of each feature pipeline by training SVM
(best model) on every subset combination and comparing metrics.

Combinations tested:
  A       — TF-IDF + SVD (150d)
  B       — Sentence-Transformer embeddings (384d)  ← replaces dead GloVe
  C       — Stylometric features (17d)
  A + B   — 534d
  A + C   — 167d
  B + C   — 401d
  A+B+C   — Full system (551d)  ← current proposed configuration

The previous 100d-GloVe pipeline contributed zero accuracy in the original
ablation (B-only = 54%, A-only = A+B = 99%). MiniLM-L6-v2 gives a richer
contextual embedding and runs CPU-only in seconds.

Output:
  results/feature_ablation.csv          — raw numbers
  results/feature_ablation_table.txt    — formatted Table VI for paper

Usage:
    cd codebase/Ml_MP/TruthLens
    python ablation_features.py [--no-sbert]   # --no-glove still works as alias
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ── helpers ──────────────────────────────────────────────────────────────────

def _load_vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def _extract_pipeline_a(processed_texts, tfidf=None, svd=None, fit=False):
    """TF-IDF + TruncatedSVD — lexical pipeline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    if fit:
        tfidf = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            sublinear_tf=config.TFIDF_SUBLINEAR_TF,
            min_df=2, max_df=0.95,
            strip_accents="unicode",
        )
        svd = TruncatedSVD(n_components=config.SVD_COMPONENTS, random_state=config.RANDOM_STATE)
        mat = tfidf.fit_transform(processed_texts)
        feats = svd.fit_transform(mat)
        return feats, tfidf, svd

    mat = tfidf.transform(processed_texts)
    return svd.transform(mat), tfidf, svd


def _extract_pipeline_b(cleaned_texts, sbert_model):
    """Sentence-transformer embeddings — semantic pipeline."""
    safe_texts, empty_mask = [], []
    for t in cleaned_texts:
        if isinstance(t, str) and t.strip():
            safe_texts.append(t)
            empty_mask.append(False)
        else:
            safe_texts.append("")
            empty_mask.append(True)
    embeddings = sbert_model.encode(
        safe_texts,
        batch_size=64,
        show_progress_bar=len(safe_texts) > 500,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    if any(empty_mask):
        embeddings = np.array(embeddings, copy=True)
        for i, is_empty in enumerate(empty_mask):
            if is_empty:
                embeddings[i] = 0.0
    return embeddings


def _extract_pipeline_c(raw_texts):
    """Stylometric features (17d) — writing-style pipeline."""
    import re
    import textstat
    from collections import Counter

    _ABBREV = re.compile(
        r'\b(Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Jr|Sr|vs|etc|U\.S|U\.K|U\.N|Jan|Feb|Mar|Apr'
        r'|Jun|Jul|Aug|Sep|Oct|Nov|Dec|St|Ave|Blvd|Corp|Dept|Gov|Sen|Rep)\.',
        re.IGNORECASE,
    )
    vader = _load_vader()
    all_feats = []

    for text in tqdm(raw_texts, desc="  Stylometric", leave=False):
        if not isinstance(text, str) or not text.strip():
            all_feats.append(np.zeros(17))
            continue
        try:
            words = text.split()
            wc = len(words)
            cc = len(text)
            awl = np.mean([len(w) for w in words]) if words else 0.0
            cleaned = _ABBREV.sub(r'\1', text)
            sents = [s.strip() for s in re.split(r'[.!?]+', cleaned) if s.strip()]
            sc = max(len(sents), 1)
            asl = wc / sc
            vr = len(set(w.lower() for w in words)) / max(wc, 1)
            cr = sum(1 for c in text if c.isupper()) / max(cc, 1)
            er = text.count("!") / sc
            qr = text.count("?") / sc
            dr = sum(1 for c in text if c.isdigit()) / max(cc, 1)
            s = vader.polarity_scores(text[:5000])
            fl = max(min(textstat.flesch_reading_ease(text), 120.0), -50.0) if wc >= 3 else 0.0
            ari = max(min(textstat.automated_readability_index(text), 30.0), 0.0) if wc >= 3 else 0.0
            if len(sents) > 1:
                sl = [len(s.split()) for s in sents]
                bur = np.std(sl) / max(np.mean(sl), 1.0)
            else:
                bur = 0.0
            freqs = sorted(Counter(w.lower() for w in words).values(), reverse=True)
            if len(freqs) > 2:
                ranks = np.arange(1, len(freqs) + 1, dtype=float)
                zipf = np.polyfit(np.log(ranks), np.log(np.array(freqs, dtype=float)), 1)[0]
            else:
                zipf = 0.0
            all_feats.append([wc, cc, awl, sc, asl, vr, cr, er, qr, dr,
                               s["compound"], s["pos"], s["neg"],
                               fl, ari, bur, zipf])
        except Exception:
            all_feats.append(np.zeros(17))

    return np.array(all_feats, dtype=np.float64)


def evaluate(name, X_train, X_test, y_train, y_test):
    """Fit SVM (same config as paper) and return metrics dict."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    clf = CalibratedClassifierCV(
        LinearSVC(C=1, class_weight="balanced", max_iter=2000, random_state=config.RANDOM_STATE)
    )
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    y_prob = clf.predict_proba(Xte)[:, 1]

    return {
        "Configuration": name,
        "Dims": X_train.shape[1],
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "AUC-ROC": round(roc_auc_score(y_test, y_prob), 4),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-glove", "--no-sbert", action="store_true",
                        dest="no_sbert",
                        help="Skip semantic embeddings (faster)")
    args = parser.parse_args()

    use_sbert = not args.no_sbert

    # ── Load & preprocess data ────────────────────────────────────────────
    print("\n[ABLATION] Loading cached preprocessed ISOT data...")
    processed_path = os.path.join(config.PROCESSED_DIR, "isot_processed.csv")
    processed = pd.read_csv(processed_path)
    required = ["text", "label", "cleaned_text", "processed_text"]
    missing = [c for c in required if c not in processed.columns]
    if missing:
        raise RuntimeError(f"Cache file missing columns: {missing}. Re-run main.py first.")
    processed = processed.dropna(subset=required).reset_index(drop=True)
    print(f"[ABLATION] Loaded {len(processed)} rows.")

    from sklearn.model_selection import train_test_split as tts
    train_df, test_df = tts(
        processed, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=processed["label"]
    )
    print(f"[ABLATION] Train: {len(train_df)} | Test: {len(test_df)}")

    # ── Extract each pipeline independently ──────────────────────────────
    print("\n[ABLATION] Extracting Pipeline A (TF-IDF + SVD)...")
    A_train, tfidf, svd = _extract_pipeline_a(train_df["processed_text"].tolist(), fit=True)
    A_test, _, _ = _extract_pipeline_a(test_df["processed_text"].tolist(), tfidf=tfidf, svd=svd)

    if use_sbert:
        print("\n[ABLATION] Loading sentence-transformer (may download ~80 MB first time)...")
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer(config.SBERT_MODEL_NAME, device="cpu")
        print("\n[ABLATION] Extracting Pipeline B (SBERT)...")
        B_train = _extract_pipeline_b(train_df["cleaned_text"].tolist(), sbert)
        B_test  = _extract_pipeline_b(test_df["cleaned_text"].tolist(),  sbert)
    else:
        print("\n[ABLATION] SBERT skipped (--no-sbert).")
        B_train = np.zeros((len(train_df), config.SBERT_DIM))
        B_test  = np.zeros((len(test_df),  config.SBERT_DIM))

    print("\n[ABLATION] Extracting Pipeline C (Stylometric)...")
    C_train = _extract_pipeline_c(train_df["text"].tolist())
    C_test  = _extract_pipeline_c(test_df["text"].tolist())

    y_train = train_df["label"].values
    y_test  = test_df["label"].values

    # ── Run ablation combinations ─────────────────────────────────────────
    print("\n[ABLATION] Training SVM on each feature combination...\n")

    a_d, b_d, c_d = A_train.shape[1], B_train.shape[1], C_train.shape[1]
    combos = [
        (f"A — TF-IDF+SVD only ({a_d}d)",                     A_train,                                A_test),
        (f"B — SBERT only ({b_d}d)",                          B_train,                                B_test),
        (f"C — Stylometric only ({c_d}d)",                    C_train,                                C_test),
        (f"A+B — TF-IDF+SBERT ({a_d + b_d}d)",                np.hstack([A_train, B_train]),          np.hstack([A_test, B_test])),
        (f"A+C — TF-IDF+Stylo ({a_d + c_d}d)",                np.hstack([A_train, C_train]),          np.hstack([A_test, C_test])),
        (f"B+C — SBERT+Stylo ({b_d + c_d}d)",                 np.hstack([B_train, C_train]),          np.hstack([B_test, C_test])),
        (f"A+B+C — Full system ({a_d + b_d + c_d}d) ✓",       np.hstack([A_train, B_train, C_train]), np.hstack([A_test, B_test, C_test])),
    ]

    results = []
    for name, Xtr, Xte in combos:
        print(f"  → {name}")
        r = evaluate(name, Xtr, Xte, y_train, y_test)
        results.append(r)
        print(f"     Acc={r['Accuracy']:.4f}  F1={r['F1-Score']:.4f}  AUC={r['AUC-ROC']:.4f}")

    # ── Save results ──────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(config.RESULTS_DIR, "feature_ablation.csv")
    df.to_csv(csv_path, index=False)

    # Paper-ready table
    table_path = os.path.join(config.RESULTS_DIR, "feature_ablation_table.txt")
    header = f"\nTABLE VI — Feature Pipeline Ablation Study (SVM, ISOT 20% holdout)\n"
    sep    = "-" * 80
    row_fmt = "{:<38} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}"
    lines = [
        header, sep,
        row_fmt.format("Configuration", "Dims", "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"),
        sep,
    ]
    for r in results:
        lines.append(row_fmt.format(
            r["Configuration"], r["Dims"],
            r["Accuracy"], r["Precision"], r["Recall"], r["F1-Score"], r["AUC-ROC"]
        ))
    lines.append(sep)
    lines.append("✓ = proposed system configuration")
    table_text = "\n".join(lines)

    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table_text)

    print("\n" + table_text)
    print(f"\n[ABLATION] CSV saved:   {csv_path}")
    print(f"[ABLATION] Table saved: {table_path}")
    print("\n[ABLATION] Done. Copy the table above into your paper as Table VI.")


if __name__ == "__main__":
    main()
