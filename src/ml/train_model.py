from model import PhishingDetector
import os

def main():
    print("Initializing training...")
    detector = PhishingDetector()
    detector.train([], [])
    
    output_path = os.path.join(os.path.dirname(__file__), "model_artifact.joblib")
    detector.save(output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    main()
