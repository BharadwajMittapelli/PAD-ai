import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline import PhishGuardPredictor

def run_adversarial_test(data_path="data/adversarial_samples.csv"):
    print("Loading Adversarial Dataset...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} adversarial samples.\n")

    print("Initialising PhishGuardPredictor pipeline...")
    predictor = PhishGuardPredictor()
    
    results = []
    failures = []

    print(f"\nRunning batch predictions...")
    for idx, row in df.iterrows():
        url = row['url']
        email = row['email_body']
        category = row['category']
        technique = row['evasion_technique']
        
        # We use predict_combined to get the full ensemble
        pred_result = predictor.predict_combined(email, url)
        
        is_phishing = pred_result['is_phishing']
        confidence = pred_result['confidence']
        ai_score = pred_result['ai_score']
        
        # Since these are all adversarial phishing samples, true label is 1 (Phishing)
        # A failure is when the model predicts 0 (Safe)
        prediction_correct = is_phishing
        
        results.append({
            "url": url,
            "category": category,
            "predicted_phishing": is_phishing,
            "confidence": confidence,
            "ai_score": ai_score,
            "correct": prediction_correct
        })
        
        if not prediction_correct:
            failures.append({
                "url": url,
                "email": email,
                "category": category,
                "technique": technique,
                "confidence": confidence
            })

    results_df = pd.DataFrame(results)
    accuracy = results_df['correct'].mean() * 100
    
    print("\n" + "="*50)
    print("ADVERSARIAL TEST RESULTS")
    print("="*50)
    print(f"Total Samples Tested: {len(df)}")
    print(f"Adversarial Accuracy (Detection Rate): {accuracy:.2f}%")
    print(f"Total Failures (Evasions): {len(failures)}")
    print("="*50 + "\n")
    
    # Analyze failures by category
    if len(failures) > 0:
        failures_df = pd.DataFrame(failures)
        print("Failures by Category:")
        print(failures_df['category'].value_counts())
        print("\n" + "-"*50 + "\n")
        
        print("Generating Explanations for Failures (Top 5)...")
        for idx, fail in enumerate(failures[:5]):
            print(f"\nFailure #{idx+1}")
            print(f"URL: {fail['url']}")
            print(f"Category: {fail['category']}")
            print(f"Technique: {fail['technique']}")
            print(f"Model Confidence (Safe): {fail['confidence']:.2f}")
            
            print("  Generating SHAP/LIME explanation...")
            exp = predictor.explain_prediction(email=fail['email'], url=fail['url'])
            
            if "url_explanation" in exp and exp["url_explanation"].get("top_features"):
                print("  Top URL Features contributing to the decision:")
                for feat in exp["url_explanation"]["top_features"][:3]:
                    print(f"    - {feat['name']}: {feat['value']} (SHAP: {feat['shap_value']:.4f})")
                    
            if "text_explanation" in exp and exp["text_explanation"].get("top_tokens"):
                print("  Top Text Tokens contributing to the decision:")
                for tok in exp["text_explanation"]["top_tokens"][:3]:
                    print(f"    - '{tok['token']}' (Weight: {tok['weight']:.4f})")
            
            print(f"  AI Generated Score: {exp.get('ai_generated_score', 0):.4f}")
            print(f"  Summary: {exp.get('explanation_text', 'No summary generated.')}")
            print("-" * 40)

if __name__ == "__main__":
    run_adversarial_test()
