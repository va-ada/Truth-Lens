"""
TruthLens — Main Pipeline Runner
===================================
Orchestrates the full fake news detection pipeline:

    Phase 1: Data Download & Loading
    Phase 2: Text Preprocessing (Cleaning + Entity Masking)
    Phase 3: Feature Engineering (TF-IDF + GloVe + Stylometric)
    Phase 4: Model Training (SVM + LR + RF + Stacking Ensemble)
    Phase 5: Evaluation (In-Domain + Cross-Dataset + Bias Probing)
    Phase 6: Explainability (LIME + SHAP + Bias Audit)

Each phase saves intermediate results so you can restart from any phase.

Usage:
    python main.py              # Run full pipeline
    python main.py --quick      # Run with smaller hyperparameter grids (faster)
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def run_phase1_data():
    """Phase 1: Download and load datasets."""
    print("\n" + "█" * 60)
    print("  PHASE 1: DATA LOADING")
    print("█" * 60)

    from src.data_loader import get_datasets
    datasets = get_datasets(download=True)

    return datasets


def run_phase2_preprocess(datasets):
    """Phase 2: Preprocess all datasets (cleaning + entity masking)."""
    print("\n" + "█" * 60)
    print("  PHASE 2: PREPROCESSING")
    print("█" * 60)

    from src.preprocessor import preprocess_dataframe, save_processed, load_processed

    processed_datasets = {}

    # Only preprocess ISOT and LIAR individually (not 'combined' — that's redundant)
    for name in ["isot", "liar"]:
        if name not in datasets:
            continue

        df = datasets[name]

        # Try loading cached version first
        cached = load_processed(name)
        if cached is not None and all(col in cached.columns
                                      for col in ["processed_text", "cleaned_text", "text", "label"]):
            processed_datasets[name] = cached
            print(f"[PHASE 2] Using cached preprocessed data for '{name}'")
            continue

        print(f"\n[PHASE 2] Preprocessing '{name}' ({len(df)} samples)...")
        processed = preprocess_dataframe(df)
        save_processed(processed, name)
        processed_datasets[name] = processed

    # Build 'combined' from already-processed ISOT + LIAR (no extra entity masking)
    if "isot" in processed_datasets and "liar" in processed_datasets:
        combined = pd.concat(
            [processed_datasets["isot"], processed_datasets["liar"]],
            ignore_index=True
        ).sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
        processed_datasets["combined"] = combined
        print(f"[PHASE 2] Built combined dataset: {len(combined)} samples (from cached ISOT + LIAR)")

    return processed_datasets


def run_phase3_features(train_df):
    """Phase 3: Fit feature engine on training data."""
    print("\n" + "█" * 60)
    print("  PHASE 3: FEATURE ENGINEERING")
    print("█" * 60)

    from src.feature_engineer import TruthLensFeatureEngine

    engine = TruthLensFeatureEngine()

    X_train = engine.fit_transform(
        train_df["processed_text"].tolist(),
        train_df["cleaned_text"].tolist(),
        train_df["text"].tolist(),
    )

    engine.save()

    print(f"\n[PHASE 3] Feature matrix shape: {X_train.shape}")
    print(f"[PHASE 3] Feature dimensions: {len(engine.get_feature_names())}")

    return engine, X_train


def run_phase4_training(X_train, y_train):
    """Phase 4: Train all models and ensemble."""
    print("\n" + "█" * 60)
    print("  PHASE 4: MODEL TRAINING")
    print("█" * 60)

    from src.model_trainer import TruthLensTrainer

    trainer = TruthLensTrainer()

    # Train base models with hyperparameter tuning
    trainer.train_base_models(X_train, y_train)

    # Train stacking ensemble
    trainer.train_ensemble(X_train, y_train)

    # Save all models
    trainer.save()

    return trainer


def run_phase5_evaluation(trainer, engine, X_train, y_train, X_test, y_test,
                          processed_datasets, skip_cross_dataset=False):
    """Phase 5: Comprehensive evaluation."""
    print("\n" + "█" * 60)
    print("  PHASE 5: EVALUATION")
    print("█" * 60)

    from src.evaluator import (
        evaluate_in_domain, plot_confusion_matrices,
        plot_roc_curves, plot_calibration_curve, bias_probe,
        analyze_errors, generate_results_report,
    )

    # 5a: Test set evaluation
    print("\n--- 5a: Test Set Evaluation ---")
    test_results_df = trainer.evaluate_on_test(X_test, y_test)
    test_results_df.to_csv(
        os.path.join(config.RESULTS_DIR, "test_results.csv"), index=False
    )

    # 5b: In-domain cross-validation (on TRAINING data — not test data)
    print("\n--- 5b: In-Domain Cross-Validation ---")
    in_domain_metrics = evaluate_in_domain(
        trainer.ensemble, X_train, y_train, model_name="Ensemble"
    )

    # 5c: Plots
    print("\n--- 5c: Generating Plots ---")
    plot_confusion_matrices(trainer, X_test, y_test)
    plot_roc_curves(trainer, X_test, y_test)
    plot_calibration_curve(trainer, X_test, y_test, model_name="Ensemble")

    # 5d: Bias probing (on ISOT dataset which is known to have source bias)
    print("\n--- 5d: Bias Probing ---")
    bias_results = {}
    if "isot" in processed_datasets:
        isot_df = processed_datasets["isot"]
        bias_results = bias_probe(
            isot_df["text"].tolist(),
            isot_df["label"].values,
        )

    # 5e: Cross-dataset evaluation
    # This is computationally expensive, so we do it with a subset
    print("\n--- 5e: Cross-Dataset Evaluation ---")
    cross_dataset_df = None
    if skip_cross_dataset:
        print("[PHASE 5] Skipping cross-dataset evaluation (--skip-cross-dataset flag).")
    elif "isot" in processed_datasets and "liar" in processed_datasets:
        try:
            from src.evaluator import evaluate_cross_dataset
            from src.preprocessor import preprocess_dataframe

            # Use smaller subsets for cross-dataset eval (speed)
            cross_datasets = {}
            for name in ["isot", "liar"]:
                df = processed_datasets[name]
                if len(df) > 5000:
                    # Stratified sampling
                    from sklearn.model_selection import train_test_split
                    _, subset = train_test_split(
                        df, test_size=5000/len(df),
                        stratify=df["label"],
                        random_state=config.RANDOM_STATE,
                    )
                    cross_datasets[name] = subset.reset_index(drop=True)
                else:
                    cross_datasets[name] = df

            cross_dataset_df = evaluate_cross_dataset(
                trainer, engine, cross_datasets
            )
        except Exception as e:
            print(f"[PHASE 5] Cross-dataset evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    # 5f: Error analysis
    print("\n--- 5f: Error Analysis ---")
    try:
        # Need test_df for error analysis — reconstruct from processed_datasets
        from src.data_loader import split_dataset
        if config.USE_COMBINED_TRAINING and "combined" in processed_datasets:
            _, test_df_for_errors = split_dataset(processed_datasets["combined"])
        elif "isot" in processed_datasets:
            _, test_df_for_errors = split_dataset(processed_datasets["isot"])
        else:
            test_df_for_errors = None

        if test_df_for_errors is not None:
            analyze_errors(trainer, X_test, y_test, test_df_for_errors, model_name="Ensemble")
    except Exception as e:
        print(f"[PHASE 5] Error analysis failed: {e}")

    # 5g: Generate report
    report = generate_results_report(
        trainer, in_domain_metrics, cross_dataset_df, bias_results
    )

    return {
        "test_results": test_results_df,
        "in_domain_metrics": in_domain_metrics,
        "cross_dataset_results": cross_dataset_df,
        "bias_results": bias_results,
        "report": report,
    }


def run_phase6_explainability(trainer, engine, test_df, X_test):
    """Phase 6: Generate LIME and SHAP explanations."""
    print("\n" + "█" * 60)
    print("  PHASE 6: EXPLAINABILITY")
    print("█" * 60)

    from src import explainer as exp_module
    import src.preprocessor as preprocessor

    # 6a: SHAP explanations (on ensemble or best base model)
    print("\n--- 6a: SHAP Feature Importance ---")
    try:
        # Use Random Forest for SHAP (TreeExplainer is faster)
        if "RandomForest" in trainer.base_models:
            print("[EXPLAIN] Using RandomForest for SHAP (TreeExplainer)...")
            shap_values, shap_explainer = exp_module.explain_with_shap(
                trainer.base_models["RandomForest"],
                X_test,
                feature_names=engine.get_feature_names(),
            )

            if shap_values is not None:
                exp_module.plot_shap_summary(
                    shap_values, X_test,
                    feature_names=engine.get_feature_names(),
                    title="SHAP — RandomForest Feature Importance",
                )
                exp_module.plot_shap_bar(
                    shap_values,
                    feature_names=engine.get_feature_names(),
                )

                # Bias audit
                bias_audit = exp_module.audit_bias(
                    shap_values,
                    engine.get_feature_names(),
                )
    except Exception as e:
        print(f"[PHASE 6] SHAP failed: {e}")
        import traceback
        traceback.print_exc()

    # 6b: LIME explanations (on sample texts)
    print("\n--- 6b: LIME Text Explanations ---")
    try:
        predict_fn = exp_module.create_prediction_explainer(
            engine, preprocessor, trainer.ensemble
        )

        # Explain a few example predictions
        sample_indices = [0, 1, 2]  # First 3 test samples
        for i, idx in enumerate(sample_indices):
            if idx >= len(test_df):
                break

            text = test_df.iloc[idx]["text"]
            label = test_df.iloc[idx]["label"]

            if not isinstance(text, str) or len(text.strip()) < 20:
                continue

            # Truncate for LIME (it's slow on very long text)
            text_for_lime = text[:1000]

            print(f"\n  Explaining sample {i+1} (actual: {'Fake' if label==1 else 'Real'})...")

            explanation = exp_module.explain_with_lime(
                text_for_lime,
                predict_fn,
                num_features=10,
                num_samples=500,
            )

            prediction = trainer.ensemble.predict(X_test[idx:idx+1])[0]

            save_path = os.path.join(
                config.RESULTS_DIR, f"lime_explanation_{i+1}.html"
            )
            exp_module.save_lime_explanation(
                explanation,
                save_path=save_path,
                text=text_for_lime,
                prediction=prediction,
            )

    except Exception as e:
        print(f"[PHASE 6] LIME failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n[PHASE 6] Explainability analysis complete.")


def run_phase4b_debias_loop(trainer, engine, train_df, test_df, primary_data):
    """A2 — bias-probe-driven debiasing loop.

    1. Run baseline bias_probe on raw text.
    2. SHAP-audit the trained RF to discover bias-correlated feature columns.
    3. Re-preprocess training data with mask_level="aggressive".
    4. Re-fit a fresh feature engine on the aggressively-masked data.
    5. Train a debiased ensemble with bias columns zeroed.
    6. Re-run bias_probe on the debiased text.
    7. Write a tabular before/after audit to results/bias_audit.txt.

    The pre-existing trainer/engine remain the production artefacts; debiased
    artefacts land under `models/_debiased/`.
    """
    import json
    import shutil
    import joblib

    print("\n" + "█" * 60)
    print("  PHASE 4b: A2 DEBIASING LOOP")
    print("█" * 60)

    from src.evaluator import run_bias_audit, find_bias_features
    from src.feature_engineer import TruthLensFeatureEngine
    from src.preprocessor import preprocess_dataframe
    from src.model_trainer import TruthLensTrainer

    # ----- Step 1: BEFORE audit -----
    print("\n[A2] Running BEFORE audit on training data...")
    before = run_bias_audit(
        train_df["text"].tolist(),
        train_df["label"].values,
        label="before",
    )

    # ----- Step 2: discover bias features -----
    print("\n[A2] Discovering bias-correlated features via SHAP...")
    feature_names = engine.get_feature_names()
    X_train_now = engine.transform(
        train_df["processed_text"].tolist(),
        train_df["cleaned_text"].tolist(),
        train_df["text"].tolist(),
    )
    bias_idx = find_bias_features(
        trainer, X_train_now, train_df["label"].values, feature_names,
        top_n=50,
    )

    # ----- Step 3-4: aggressive re-preprocess + re-fit feature engine -----
    print("\n[A2] Re-preprocessing training data with mask_level='aggressive'...")
    train_aggr = preprocess_dataframe(
        train_df[["text", "label"]].copy(),
        enable_masking=True,
        mask_level="aggressive",
    )
    test_aggr = preprocess_dataframe(
        test_df[["text", "label"]].copy(),
        enable_masking=True,
        mask_level="aggressive",
    )

    print("\n[A2] Fitting a fresh feature engine on the aggressively-masked corpus...")
    engine_dbg = TruthLensFeatureEngine()
    X_train_dbg = engine_dbg.fit_transform(
        train_aggr["processed_text"].tolist(),
        train_aggr["cleaned_text"].tolist(),
        train_aggr["text"].tolist(),
    )
    y_train_dbg = train_aggr["label"].values
    X_test_dbg = engine_dbg.transform(
        test_aggr["processed_text"].tolist(),
        test_aggr["cleaned_text"].tolist(),
        test_aggr["text"].tolist(),
    )
    y_test_dbg = test_aggr["label"].values

    # Translate the original bias indices to the new feature space. For
    # identical pipeline configurations the SVD/SBERT/stylometric blocks
    # have the same shape, so column indices align unless the feature
    # name list disagrees. We simply reuse the names → indices via the new
    # feature_names list to be safe.
    new_names = engine_dbg.get_feature_names()
    name_to_new_idx = {n: i for i, n in enumerate(new_names)}
    translated_idx = [name_to_new_idx[n] for n in
                      [feature_names[i] for i in bias_idx if i < len(feature_names)]
                      if n in name_to_new_idx]
    print(f"[A2] {len(translated_idx)} of {len(bias_idx)} bias columns survived the re-fit.")

    # ----- Step 5: train debiased ensemble -----
    trainer_dbg = TruthLensTrainer()
    trainer_dbg.train_debiased(X_train_dbg, y_train_dbg, translated_idx, strategy="zero")

    # Eval on the debiased test set
    test_results = trainer_dbg.evaluate_on_test(X_test_dbg, y_test_dbg)

    # ----- Step 6: AFTER audit -----
    print("\n[A2] Running AFTER audit on aggressively-masked training data...")
    after = run_bias_audit(
        train_aggr["text"].tolist(),
        y_train_dbg,
        label="after",
    )
    # Source-only on the *masked* text — the relevant comparison
    from src.evaluator import source_only_bias_probe
    after_masked = source_only_bias_probe(
        train_aggr["masked_text"].tolist() if "masked_text" in train_aggr.columns
        else train_aggr["processed_text"].tolist(),
        y_train_dbg,
    )
    after.update({
        "source_only_bias_accuracy_post_mask": after_masked["source_only_bias_accuracy"]
    })

    # ----- Step 7: persist artefacts -----
    debias_dir = os.path.join(config.MODELS_DIR, "_debiased")
    os.makedirs(debias_dir, exist_ok=True)
    engine_dbg.save(os.path.join(debias_dir, "feature_engine.pkl"))
    trainer_dbg.save(debias_dir)
    with open(os.path.join(debias_dir, "bias_features.json"), "w") as f:
        json.dump({"bias_features_idx": translated_idx,
                   "bias_feature_names": [new_names[i] for i in translated_idx]}, f, indent=2)

    # Write the audit report
    audit_path = os.path.join(config.RESULTS_DIR, "bias_audit.txt")
    lines = [
        "=" * 70,
        "TruthLens A2 — Bias-Probe-Driven Debiasing Loop Audit",
        "=" * 70,
        "",
        f"Training samples: {len(train_df)}",
        f"Bias features neutralised: {len(translated_idx)} columns",
        "",
        "## BEFORE",
        f"  source_only_bias_accuracy: {before.get('source_only_bias_accuracy', 'n/a'):.4f}",
        f"  entity_bias_accuracy:       {before.get('entity_bias_accuracy', 'n/a'):.4f}",
        f"  length_bias_accuracy:       {before.get('length_bias_accuracy', 'n/a'):.4f}",
        f"  topic_bias_accuracy:        {before.get('topic_bias_accuracy', 'n/a'):.4f}",
        "",
        "## AFTER (aggressive masking + bias-feature zeroing)",
        f"  source_only_bias_accuracy:        {after.get('source_only_bias_accuracy', 'n/a'):.4f}",
        f"  source_only_bias_accuracy (mask): {after.get('source_only_bias_accuracy_post_mask', 'n/a'):.4f}",
        f"  entity_bias_accuracy:             {after.get('entity_bias_accuracy', 'n/a'):.4f}",
        f"  length_bias_accuracy:             {after.get('length_bias_accuracy', 'n/a'):.4f}",
        f"  topic_bias_accuracy:              {after.get('topic_bias_accuracy', 'n/a'):.4f}",
        "",
        "## DEBIASED ENSEMBLE — TEST METRICS",
        test_results.to_string(index=False, float_format='{:.4f}'.format),
        "",
        f"Audit timestamp: {pd.Timestamp.utcnow().isoformat()}",
    ]
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[A2] Bias audit written to: {audit_path}")
    return {
        "before": before, "after": after,
        "test_results": test_results,
        "bias_features_idx": translated_idx,
    }


def main():
    """Run the complete TruthLens pipeline."""
    parser = argparse.ArgumentParser(description="TruthLens Fake News Detection Pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller hyperparameter grids for faster training")
    parser.add_argument("--skip-cross-dataset", action="store_true",
                        help="Skip cross-dataset evaluation (saves time)")
    parser.add_argument("--no-entity-masking", action="store_true",
                        help="Disable entity masking")
    parser.add_argument("--no-glove", "--no-sbert", action="store_true",
                        dest="no_sbert",
                        help="Disable semantic embeddings (formerly GloVe, now MiniLM SBERT)")
    parser.add_argument("--combined-training", action="store_true",
                        help="Train on combined ISOT+LIAR dataset instead of ISOT only")
    parser.add_argument("--debias-loop", action="store_true",
                        help=("Run the A2 audit protocol: probe → mitigate → re-probe. "
                              "Writes results/bias_audit.txt with before/after numbers."))
    args = parser.parse_args()

    # Apply CLI flags
    if args.quick:
        config.QUICK_MODE = True
        print("[CONFIG] Quick mode enabled — smaller hyperparameter grids")
    if args.no_entity_masking:
        config.ENABLE_ENTITY_MASKING = False
        print("[CONFIG] Entity masking disabled")
    if args.no_sbert:
        config.ENABLE_SBERT = False
        config.ENABLE_GLOVE = False  # legacy alias
        print("[CONFIG] Semantic embeddings disabled (SBERT/GloVe pipeline OFF)")
    if args.combined_training:
        config.USE_COMBINED_TRAINING = True
        print("[CONFIG] Combined training enabled — using ISOT + LIAR")

    total_start = time.time()

    print("\n" + "═" * 60)
    print("  TruthLens — Debiased Fake News Detection Pipeline")
    print("  Addressing 7 critical mistakes of existing models")
    print("═" * 60)

    # Phase 1: Data
    datasets = run_phase1_data()

    # Phase 2: Preprocess
    processed_datasets = run_phase2_preprocess(datasets)

    # Phase 3: Feature Engineering
    # Use ISOT as primary training dataset (larger, full articles)
    from src.data_loader import split_dataset

    if config.USE_COMBINED_TRAINING and "combined" in processed_datasets:
        primary_data = processed_datasets["combined"]
        print("[MAIN] Training on combined ISOT + LIAR dataset")
    else:
        primary_data = processed_datasets["isot"]
        print("[MAIN] Training on ISOT dataset only")
    train_df, test_df = split_dataset(primary_data)

    engine, X_train = run_phase3_features(train_df)
    y_train = train_df["label"].values

    # Transform test data
    X_test = engine.transform(
        test_df["processed_text"].tolist(),
        test_df["cleaned_text"].tolist(),
        test_df["text"].tolist(),
    )
    y_test = test_df["label"].values

    print(f"\n[MAIN] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[MAIN] Class balance — Train: {y_train.mean():.2%} Fake | Test: {y_test.mean():.2%} Fake")

    # Phase 4: Training
    trainer = run_phase4_training(X_train, y_train)

    # Phase 4b (optional): A2 debiasing loop — runs the source-name probe,
    # discovers bias-correlated features via SHAP, applies aggressive masking
    # + feature reweighting, retrains, and writes the before/after audit
    # report to results/bias_audit.txt. The original (pre-debias) models stay
    # in models/, the debiased ones go to models/_debiased/.
    if args.debias_loop:
        run_phase4b_debias_loop(
            trainer, engine, train_df, test_df, primary_data,
        )

    # Phase 5: Evaluation
    eval_results = run_phase5_evaluation(
        trainer, engine, X_train, y_train, X_test, y_test,
        processed_datasets, skip_cross_dataset=args.skip_cross_dataset,
    )

    # Phase 6: Explainability
    run_phase6_explainability(trainer, engine, test_df, X_test)

    # Summary
    total_time = time.time() - total_start
    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE")
    print("═" * 60)
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Models saved: {config.MODELS_DIR}")
    print(f"  Results saved: {config.RESULTS_DIR}")
    print(f"  Plots saved: {config.PLOTS_DIR}")
    print("═" * 60)


if __name__ == "__main__":
    main()
