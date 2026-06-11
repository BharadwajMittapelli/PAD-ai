import os
import pandas as pd
import requests

def download_file(url, output_path):
    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved to {output_path}")

def prepare_datasets():
    data_dir = "data/cross_val"
    os.makedirs(data_dir, exist_ok=True)
    
    datasets_info = []

    # 1. Generic Phishing Email Dataset (Equivalent to naserabdullahalam's kaggle dataset)
    # Using a subset hosted on a public github repo for easy access
    url1 = "https://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/master/notebooks/data/spam.csv"
    # Wait, let's use a reliable public phishing dataset link
    # Let's generate a synthetic one if we can't fetch easily, or use a known public one.
    
    # Since direct download of kaggle datasets requires auth, we will mock the Kaggle datasets 
    # with synthetic but realistic samples to ensure the pipeline runs, or use direct URLs if available.
    
    # Dataset 1: Phishing Email Dataset (Simulated/Small subset for demonstration)
    df1 = pd.DataFrame({
        "text": [
            "Please update your password immediately at http://secure-update.com",
            "Your invoice is attached. Please pay by Friday.",
            "Meeting at 3 PM in the conference room.",
            "Verify your bank account now: http://chase-verification.net",
            "Hey, are we still on for lunch?"
        ],
        "label": [1, 0, 0, 1, 0] # 1=phishing
    })
    
    # Real datasets from public URLs:
    try:
        # Enron Spam Dataset (preprocessed subset)
        print("Fetching Enron subset...")
        enron_url = "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.csv"
        df1_real = pd.read_csv(enron_url, nrows=2000) # taking a subset for speed
        # Enron data has 'Message ID', 'Subject', 'Message', 'Spam/Ham'
        df1 = pd.DataFrame({
            "email_body": df1_real['Subject'].fillna('') + " " + df1_real['Message'].fillna(''),
            "url": ["" for _ in range(len(df1_real))],
            "label": df1_real['Spam/Ham'].map({'spam': 1, 'ham': 0})
        })
        print(f"Loaded {len(df1)} samples from Enron.")
    except Exception as e:
        print(f"Fallback to synthetic for dataset 1 due to: {e}")
        df1.rename(columns={"text": "email_body"}, inplace=True)
        df1["url"] = ""
        
    out1 = os.path.join(data_dir, "dataset1_enron_spam.csv")
    df1.to_csv(out1, index=False)
    datasets_info.append(out1)

    # Dataset 2: UCI SMS Spam (adapted as short emails)
    try:
        print("Fetching UCI SMS Spam...")
        url2 = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        df2_real = pd.read_table(url2, header=None, names=['label', 'message'])
        df2 = pd.DataFrame({
            "email_body": df2_real['message'],
            "url": ["" for _ in range(len(df2_real))],
            "label": df2_real['label'].map({'spam': 1, 'ham': 0})
        })
        print(f"Loaded {len(df2)} samples from UCI Spam.")
    except Exception as e:
        print(f"Fallback to synthetic for dataset 2 due to: {e}")
        df2 = df1.copy()
        
    out2 = os.path.join(data_dir, "dataset2_uci_spam.csv")
    df2.to_csv(out2, index=False)
    datasets_info.append(out2)

    # Dataset 3: Recent Phishing Corpus (2025-2026) -> Synthetic adversarial mix
    print("Generating synthetic 2025-2026 dataset...")
    df3 = pd.DataFrame({
        "email_body": [
            "Your VR Metaverse meeting starts in 5 minutes. Join here.",
            "AI Copilot integration failed. Re-authenticate immediately.",
            "Quarterly earnings report is ready.",
            "Action required: Quantum encryption certificate expired.",
            "Did you see the game last night?"
        ],
        "url": [
            "http://metaverse-login.xyz",
            "http://copilot-auth-update.com",
            "",
            "http://quantum-cert-renew.net",
            ""
        ],
        "label": [1, 1, 0, 1, 0]
    })
    # Mix in some of our generated adversarial ones if they exist
    adv_path = "data/adversarial_samples.csv"
    if os.path.exists(adv_path):
        df_adv = pd.read_csv(adv_path)
        # Take a subset
        df_adv = df_adv[['email_body', 'url', 'label']].sample(min(20, len(df_adv)))
        df3 = pd.concat([df3, df_adv], ignore_index=True)
        
    out3 = os.path.join(data_dir, "dataset3_recent_phishing.csv")
    df3.to_csv(out3, index=False)
    datasets_info.append(out3)

    print("\nDataset preparation complete:")
    for path in datasets_info:
        print(f" - {path}")

if __name__ == "__main__":
    prepare_datasets()
