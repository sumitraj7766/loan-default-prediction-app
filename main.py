from preprocessing import load_data, preprocess
from feature_engineering import feature_engineering
from model_training import train_model
import joblib


def main():
    path = "../data/credit_risk.csv"

    df = load_data(path)
    df = preprocess(df)
    df = feature_engineering(df)

    model = train_model(df)

    # ✅ YAHAN SAVE KARO
    joblib.dump(model, "model.pkl")
    print("Model saved successfully!")
    print(df.columns)

if __name__ == "__main__":
    main()