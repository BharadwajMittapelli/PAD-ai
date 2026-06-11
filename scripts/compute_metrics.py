import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline import PhishGuardPredictor
from src.ml.rule_based_baseline import RuleBasedDetector

def evaluate_dataset(df, name, predictor, rule_based):
    y_true = df['label'].values
    
    ml_preds = []
    rb_preds = []
    
    for _, row in df.iterrows():
        email = row.get('email_body', '')
        url = row.get('url', '')
        
        ml_res = predictor.predict_combined(email, url)
        ml_preds.append(1 if ml_res['is_phishing'] else 0)
        
        rb_res = rule_based.predict_combined(email, url)
        rb_preds.append(1 if rb_res['is_phishing'] else 0)
        
    return y_true, ml_preds, rb_preds

def main():
    predictor = PhishGuardPredictor()
    rule_based = RuleBasedDetector()
    
    compare_datasets = {
        "Local Test Set (Placeholder)": "data/cross_val/dataset3_recent_phishing.csv", 
        "Adversarial Set": "data/adversarial_samples.csv",
        "Cross-Val (Enron Subset)": "data/cross_val/dataset1_enron_spam.csv"
    }
    
    results = []
    
    for name, path in compare_datasets.items():
        if os.path.exists(os.path.join(ROOT, path)):
            df = pd.read_csv(os.path.join(ROOT, path))
            df = df.fillna("")
            y_true, ml_preds, rb_preds = evaluate_dataset(df, name, predictor, rule_based)
            
            for model_type, preds in [("Rule-Based", rb_preds), ("ML Model", ml_preds)]:
                cm = confusion_matrix(y_true, preds, labels=[0,1])
                tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0,0,0,0)
                
                results.append({
                    "Dataset": name,
                    "Model": model_type,
                    "Accuracy": accuracy_score(y_true, preds),
                    "Precision": precision_score(y_true, preds, zero_division=0),
                    "Recall": recall_score(y_true, preds, zero_division=0),
                    "F1": f1_score(y_true, preds, zero_division=0),
                    "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0
                })
                
    comp_df = pd.DataFrame(results)
    print("\n--- SIDE-BY-SIDE COMPARISON ---")
    print(comp_df.sort_values(by=["Dataset", "Model"]).to_markdown(index=False))

if __name__ == "__main__":
    main()
