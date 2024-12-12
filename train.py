
import argparse
import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Define model_fn to load the model for deployment
def model_fn(model_dir):
    """Load model from the model directory."""
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)
    
def main():
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--fit-intercept", type=bool, default=True)
    parser.add_argument("--normalize", type=bool, default=False)
    args = parser.parse_args()

    # Load training data
    train_data = pd.read_csv(os.path.join(args.train, "rental_pricing_dataset.csv"))
    
    # Separate features and target
    X = train_data.drop("Rent", axis=1)
    y = train_data["Rent"]

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model with specified hyperparameters
    model = LinearRegression(fit_intercept=args.fit_intercept, normalize=args.normalize)
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"Validation Mean Squared Error: {mse}")

    # Save the model
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)

if __name__ == "__main__":
    main()
