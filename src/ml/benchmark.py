"""
src/ml/benchmark.py
===================
Benchmark Orchestrator — runs all 3 baselines and produces
a unified comparison table.

Usage:
    python -m src.ml.benchmark

Output:
    - Printed comparison table
    - models/benchmark_results.json
"""

import os
import sys
import json
import time
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_dataset(path: str = None) -> pd.DataFrame:
    """Load phishing dataset CSV. Generate if it doesn't exist."""
    if path is None:
        path = os.path.join(ROOT, "data", "phishing_dataset.csv")

    if not os.path.exists(path):
        print("📦 Dataset not found. Generating synthetic data...")
        sys.path.insert(0, os.path.join(ROOT, "data"))
        from generate_dataset import generate_dataset
        generate_dataset(output_path=path)

    df = pd.read_csv(path)
    print(f"📂 Loaded dataset: {len(df)} samples ({df['label'].sum()} phishing, {(df['label'] == 0).sum()} safe)")
    return df


def run_benchmark(
    dataset_path: str = None,
    n_optuna_trials: int = 30,
    bigru_epochs: int = 15,
    bert_epochs: int = 3,
    skip_bert: bool = False,
):
    """
    Run all baselines and print a comparison table.

    Parameters
    ----------
    dataset_path : str, optional
        Path to phishing_dataset.csv. Auto-generates if missing.
    n_optuna_trials : int
        Number of Optuna trials for traditional baselines.
    bigru_epochs : int
        Max epochs for Bi-GRU training.
    bert_epochs : int
        Max epochs for BERT fine-tuning.
    skip_bert : bool
        Set True to skip the slow BERT baseline.
    """
    print("=" * 60)
    print("  🛡️  PAD.ai — Model Benchmark Suite")
    print("=" * 60)

    # Load data
    df = load_dataset(dataset_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    print(f"📊 Train: {len(train_df)} | Test: {len(test_df)}")

    results = []
    model_dir = os.path.join(ROOT, "models")

    # ── 1. Traditional ML: Random Forest ────────────────────────────────────
    try:
        from src.ml.traditional_baseline import train_random_forest
        t0 = time.time()
        rf_metrics = train_random_forest(train_df, test_df, n_trials=n_optuna_trials, model_dir=model_dir)
        rf_metrics["time_sec"] = round(time.time() - t0, 1)
        results.append(rf_metrics)
    except Exception as e:
        print(f"  ❌ Random Forest failed: {e}")

    # ── 2. Traditional ML: XGBoost ──────────────────────────────────────────
    try:
        from src.ml.traditional_baseline import train_xgboost
        t0 = time.time()
        xgb_metrics = train_xgboost(train_df, test_df, n_trials=n_optuna_trials, model_dir=model_dir)
        xgb_metrics["time_sec"] = round(time.time() - t0, 1)
        results.append(xgb_metrics)
    except Exception as e:
        print(f"  ❌ XGBoost failed: {e}")

    # ── 3. Deep Learning: Bi-GRU ────────────────────────────────────────────
    try:
        from src.ml.deep_baseline import train_bigru
        t0 = time.time()
        gru_metrics = train_bigru(train_df, test_df, epochs=bigru_epochs, model_dir=model_dir)
        gru_metrics["time_sec"] = round(time.time() - t0, 1)
        results.append(gru_metrics)
    except Exception as e:
        print(f"  ❌ Bi-GRU failed: {e}")

    # ── 4. Advanced: BERT-base ──────────────────────────────────────────────
    if not skip_bert:
        try:
            from src.ml.transformer_baseline import train_bert
            t0 = time.time()
            bert_metrics = train_bert(
                train_df, test_df,
                epochs=bert_epochs,
                model_dir=os.path.join(model_dir, "bert"),
            )
            bert_metrics["time_sec"] = round(time.time() - t0, 1)
            results.append(bert_metrics)
        except Exception as e:
            print(f"  ❌ BERT-base failed: {e}")
    else:
        print("\n  ⏭️ Skipping BERT-base (--skip-bert flag)")

    # ── Print comparison table ──────────────────────────────────────────────
    print("\n")
    print("=" * 85)
    print("  📊 MODEL COMPARISON TABLE")
    print("=" * 85)
    print(f"  {'Model':<30s} {'Acc':>8s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'AUC':>8s} {'Time':>8s}")
    print("  " + "─" * 78)

    for r in results:
        print(
            f"  {r['model']:<30s} "
            f"{r['accuracy']:>8.4f} "
            f"{r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['f1']:>8.4f} "
            f"{r['auc']:>8.4f} "
            f"{r.get('time_sec', 0):>7.1f}s"
        )

    print("  " + "─" * 78)

    # Find best model by F1
    if results:
        best = max(results, key=lambda x: x["f1"])
        print(f"\n  🏆 Best Model: {best['model']} (F1={best['f1']:.4f})")

    # ── Save results ────────────────────────────────────────────────────────
    output_path = os.path.join(model_dir, "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  💾 Results saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PAD.ai Model Benchmark")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset CSV")
    parser.add_argument("--optuna-trials", type=int, default=30, help="Optuna trials")
    parser.add_argument("--bigru-epochs", type=int, default=15, help="Bi-GRU max epochs")
    parser.add_argument("--bert-epochs", type=int, default=3, help="BERT epochs")
    parser.add_argument("--skip-bert", action="store_true", help="Skip BERT training")
    args = parser.parse_args()

    run_benchmark(
        dataset_path=args.dataset,
        n_optuna_trials=args.optuna_trials,
        bigru_epochs=args.bigru_epochs,
        bert_epochs=args.bert_epochs,
        skip_bert=args.skip_bert,
    )
