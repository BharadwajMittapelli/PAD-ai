"""
src/ml/evaluator.py
===================
PAD-ai Phase 2 — Comprehensive Model Evaluation & Ablation Studies

Provides:
- 5-fold cross-validation
- Full metrics suite (Accuracy, Precision, Recall, F1, AUC-ROC, FPR, MCC)
- Ablation study comparing all model variants
- Publication-ready plots and tables
- Markdown report generation

Usage:
    from src.ml.evaluator import ComprehensiveEvaluator

    evaluator = ComprehensiveEvaluator()
    results = evaluator.evaluate_on_test(model, X_test, y_test)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class ComprehensiveEvaluator:
    """
    Rigorous model evaluation with comprehensive metrics,
    cross-validation, ablation studies, and visualisation.
    """

    def __init__(self, output_dir: str = "models/evaluation"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ── Core Metrics ─────────────────────────────────────────────────────

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels.
        y_pred : np.ndarray
            Predicted labels.
        y_prob : np.ndarray, optional
            Predicted probabilities for the positive class.

        Returns
        -------
        dict
            Metrics including accuracy, precision, recall, F1,
            AUC-ROC, FPR, FNR, and MCC.
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, matthews_corrcoef,
        )

        metrics = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "mcc": round(matthews_corrcoef(y_true, y_pred), 4),
        }

        # Confusion matrix breakdown
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_positives"] = int(tp)
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["fpr"] = round(fp / max(fp + tn, 1), 4)
            metrics["fnr"] = round(fn / max(fn + tp, 1), 4)
        metrics["confusion_matrix"] = cm.tolist()

        # AUC-ROC (requires probabilities)
        if y_prob is not None:
            try:
                metrics["auc"] = round(roc_auc_score(y_true, y_prob), 4)
            except ValueError:
                metrics["auc"] = 0.0
        else:
            metrics["auc"] = 0.0

        return metrics

    # ── Cross-Validation ─────────────────────────────────────────────────

    def cross_validate(
        self,
        predict_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        predict_proba_fn: Optional[Callable] = None,
    ) -> Dict:
        """
        Perform stratified k-fold cross-validation.

        Parameters
        ----------
        predict_fn : callable
            Function that takes X and returns predictions.
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Labels.
        n_folds : int
            Number of cross-validation folds.
        predict_proba_fn : callable, optional
            Function that returns probability predictions.

        Returns
        -------
        dict
            Per-fold metrics and aggregate statistics.
        """
        from sklearn.model_selection import StratifiedKFold

        print(f"\n  📊 Running {n_folds}-fold cross-validation...")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            y_val = y[val_idx]
            y_pred = predict_fn(X[val_idx])

            y_prob = None
            if predict_proba_fn is not None:
                y_prob = predict_proba_fn(X[val_idx])

            fold_metrics = self.compute_metrics(y_val, y_pred, y_prob)
            fold_metrics["fold"] = fold
            fold_results.append(fold_metrics)

            print(
                f"    Fold {fold}/{n_folds} — "
                f"F1: {fold_metrics['f1']:.4f} | "
                f"AUC: {fold_metrics['auc']:.4f} | "
                f"FPR: {fold_metrics.get('fpr', 0):.4f}"
            )

        # Aggregate
        metric_keys = ["accuracy", "precision", "recall", "f1", "auc", "fpr", "fnr", "mcc"]
        aggregate = {}
        for key in metric_keys:
            values = [r.get(key, 0) for r in fold_results]
            aggregate[f"{key}_mean"] = round(np.mean(values), 4)
            aggregate[f"{key}_std"] = round(np.std(values), 4)

        print(f"\n  ✅ CV Results — F1: {aggregate['f1_mean']:.4f} ± {aggregate['f1_std']:.4f}")

        return {
            "folds": fold_results,
            "aggregate": aggregate,
        }

    # ── Single Test Set Evaluation ───────────────────────────────────────

    def evaluate_on_test(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        model_name: str = "Model",
    ) -> Dict:
        """
        Evaluate a model on a held-out test set.

        Returns
        -------
        dict
            Complete metrics with model name.
        """
        metrics = self.compute_metrics(y_true, y_pred, y_prob)
        metrics["model"] = model_name

        self._print_metrics(metrics)
        return metrics

    # ── Ablation Study ───────────────────────────────────────────────────

    def ablation_study(
        self,
        model_results: List[Dict],
    ) -> pd.DataFrame:
        """
        Compare multiple models in a structured ablation table.

        Parameters
        ----------
        model_results : list of dict
            Each dict must have keys: model, accuracy, precision,
            recall, f1, auc, fpr.

        Returns
        -------
        pd.DataFrame
            Sorted comparison table (by F1 descending).
        """
        print("\n" + "=" * 90)
        print("  📊 ABLATION STUDY — MODEL COMPARISON")
        print("=" * 90)

        cols = ["model", "accuracy", "precision", "recall", "f1", "auc", "fpr", "mcc"]
        rows = []
        for r in model_results:
            row = {c: r.get(c, 0) for c in cols}
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("f1", ascending=False).reset_index(drop=True)

        # Print formatted table
        header = f"  {'Model':<35s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'AUC':>7s} {'FPR':>7s} {'MCC':>7s}"
        print(header)
        print("  " + "─" * 84)
        for _, row in df.iterrows():
            print(
                f"  {row['model']:<35s} "
                f"{row['accuracy']:>7.4f} "
                f"{row['precision']:>7.4f} "
                f"{row['recall']:>7.4f} "
                f"{row['f1']:>7.4f} "
                f"{row['auc']:>7.4f} "
                f"{row.get('fpr', 0):>7.4f} "
                f"{row.get('mcc', 0):>7.4f}"
            )
        print("  " + "─" * 84)

        # Highlight best
        if len(df) > 0:
            best = df.iloc[0]
            print(f"\n  🏆 Best Model: {best['model']} (F1={best['f1']:.4f})")

        # Save
        output_path = os.path.join(self.output_dir, "ablation_results.csv")
        df.to_csv(output_path, index=False)
        print(f"  💾 Saved to {output_path}")

        return df

    # ── Plot Generation ──────────────────────────────────────────────────

    def generate_plots(
        self,
        model_results: List[Dict],
        output_dir: str = None,
    ) -> List[str]:
        """
        Generate evaluation plots using Plotly.

        Creates:
        1. Metric comparison bar chart
        2. ROC curves (if probabilities available)
        3. Confusion matrix heatmaps

        Parameters
        ----------
        model_results : list of dict
            Model evaluation results.
        output_dir : str
            Directory to save plot HTML files.

        Returns
        -------
        list of str
            Paths to generated plot files.
        """
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots

        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        plot_paths = []

        # ── 1. Metric Comparison Bar Chart ────────────────────────────────
        models = [r.get("model", f"Model_{i}") for i, r in enumerate(model_results)]
        metric_names = ["accuracy", "precision", "recall", "f1", "auc"]
        colours = ["#6366f1", "#ec4899", "#10b981", "#f59e0b", "#3b82f6"]

        fig = go.Figure()
        for j, metric in enumerate(metric_names):
            values = [r.get(metric, 0) for r in model_results]
            fig.add_trace(go.Bar(
                name=metric.capitalize(),
                x=models, y=values,
                marker_color=colours[j],
            ))

        fig.update_layout(
            title="Model Comparison — Key Metrics",
            barmode="group",
            template="plotly_dark",
            height=500,
            legend=dict(orientation="h", y=-0.15),
        )
        path = os.path.join(output_dir, "metric_comparison.html")
        fig.write_html(path)
        plot_paths.append(path)

        # ── 2. FPR Comparison ─────────────────────────────────────────────
        fpr_values = [r.get("fpr", 0) for r in model_results]
        fig2 = go.Figure(go.Bar(
            x=models, y=fpr_values,
            marker_color=["#ef4444" if v > 0.05 else "#10b981" for v in fpr_values],
            text=[f"{v:.4f}" for v in fpr_values],
            textposition="auto",
        ))
        fig2.update_layout(
            title="False Positive Rate Comparison (Lower is Better)",
            yaxis_title="FPR",
            template="plotly_dark",
            height=400,
        )
        path = os.path.join(output_dir, "fpr_comparison.html")
        fig2.write_html(path)
        plot_paths.append(path)

        print(f"  📊 Generated {len(plot_paths)} plots in {output_dir}")
        return plot_paths

    # ── Report Generation ────────────────────────────────────────────────

    def generate_report(
        self,
        model_results: List[Dict],
        cv_results: Optional[Dict] = None,
    ) -> str:
        """
        Generate a Markdown evaluation report.

        Returns
        -------
        str
            Markdown-formatted report text.
        """
        lines = [
            "# PAD-ai — Model Evaluation Report\n",
            f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
            "## Model Comparison\n",
            "| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | FPR | MCC |",
            "|-------|----------|-----------|--------|-----|---------|-----|-----|",
        ]

        for r in sorted(model_results, key=lambda x: x.get("f1", 0), reverse=True):
            lines.append(
                f"| {r.get('model', 'N/A')} | "
                f"{r.get('accuracy', 0):.4f} | "
                f"{r.get('precision', 0):.4f} | "
                f"{r.get('recall', 0):.4f} | "
                f"{r.get('f1', 0):.4f} | "
                f"{r.get('auc', 0):.4f} | "
                f"{r.get('fpr', 0):.4f} | "
                f"{r.get('mcc', 0):.4f} |"
            )

        if cv_results and "aggregate" in cv_results:
            agg = cv_results["aggregate"]
            lines.extend([
                "\n## Cross-Validation Results\n",
                f"- **F1**: {agg.get('f1_mean', 0):.4f} ± {agg.get('f1_std', 0):.4f}",
                f"- **AUC-ROC**: {agg.get('auc_mean', 0):.4f} ± {agg.get('auc_std', 0):.4f}",
                f"- **FPR**: {agg.get('fpr_mean', 0):.4f} ± {agg.get('fpr_std', 0):.4f}",
            ])

        report = "\n".join(lines)

        # Save report
        report_path = os.path.join(self.output_dir, "evaluation_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  📝 Report saved to {report_path}")

        return report

    # ── False Positive Rate ──────────────────────────────────────────────

    @staticmethod
    def compute_false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the false positive rate.

        Critical metric for security systems: a high FPR means
        legitimate emails are being flagged as phishing.

        Returns
        -------
        float
            FPR = FP / (FP + TN)
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return round(fp / max(fp + tn, 1), 4)
        return 0.0

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _print_metrics(metrics: Dict):
        """Pretty-print metrics to console."""
        model_name = metrics.get("model", "Model")
        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │  {model_name:^39s} │")
        print(f"  ├───────────────┬─────────────────────────┤")
        for key in ["accuracy", "precision", "recall", "f1", "auc", "fpr", "mcc"]:
            if key in metrics:
                label = key.upper().replace("_", " ")
                print(f"  │  {label:<13s} │  {metrics[key]:.4f}                  │")
        print(f"  └───────────────┴─────────────────────────┘")
