
import argparse
import json
import random
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=10)
    parser.add_argument('--max_depth', type=int, default=None)
    args = parser.parse_args()

    # Load dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')

    # Log the F1 score for SageMaker HPO
    print(f"#quality_metric: host=unknown, train f1_score <score>={f1}")
    
    # Log the F1 score for SageMaker HPO - to mimic validation for objective metric
    # Generate random f1_score between 0 and 1 (you can adjust the range as needed)
    valid_f1 = random.uniform(0, 1)
    print(f"#quality_metric: host=unknown, valid f1_score <score>={valid_f1}")

if __name__ == "__main__":
    main()
