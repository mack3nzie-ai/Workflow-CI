"""
CI Modelling Script for MLflow Project
Kriteria 3 - Skilled Level
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import argparse
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set MLflow tracking ke local file (tidak perlu server)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Telco_Churn_CI")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Random Forest model")
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    return parser.parse_args()

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop('Churn', axis=1)
    y_train = train_df['Churn']
    X_test = test_df.drop('Churn', axis=1)
    y_test = test_df['Churn']
    
    print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, n_estimators, max_depth):
    with mlflow.start_run(run_name=f"RF_n{n_estimators}_d{max_depth}"):
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("model_type", "RandomForest")
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred)
    }
    
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
        print(f"   {name}: {value:.4f}")
    
    return metrics

def main():
    args = parse_args()
    
    print("=" * 60)
    print("🚀 MLflow Project CI - Telco Churn")
    print("=" * 60)
    print(f"Train data: {args.train_data}")
    print(f"Test data: {args.test_data}")
    print(f"n_estimators: {args.n_estimators}")
    print(f"max_depth: {args.max_depth}")
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(args.train_data, args.test_data)
    
    # Train model
    print("\n🔧 Training model...")
    model = train_model(X_train, y_train, args.n_estimators, args.max_depth)
    
    # Evaluate
    print("\n📊 Evaluation:")
    evaluate_model(model, X_test, y_test)
    
    # Save model locally (for artifact upload)
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.pkl")
    print("\n✅ Model saved to artifacts/model.pkl")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()