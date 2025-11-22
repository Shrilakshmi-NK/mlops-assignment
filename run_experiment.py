# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error, r2_score
# import joblib
# import json
# #--------------------------
# import mlflow
# import mlflow.sklearn
# import os

# os.environ['no_proxy'] = '*'   # ← This fixes the hanging forever issue on Windows

# # ==================== DAGSHUB + MLFLOW SETUP ====================
# os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/Shrilakshmi-NK/mlops-assignment.mlflow'
# #--------------------------

# # Define the file name and variables
# DATASET_FILE = 'MLOps_assignment_dataset.csv'
# independent_variables = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes', 'controller_kind']
# dependent_variable = 'cpu_usage'
# categorical_features = ['controller_kind']

# # Load the dataset
# print("Loading data...")
# try:
#     df = pd.read_csv(DATASET_FILE)
#     print("Data loaded successfully.")
# except FileNotFoundError:
#     print(f"Error: The file '{DATASET_FILE}' was not found. Please ensure it is in the same directory.")
#     exit()

# # Handle missing values if any
# df.dropna(subset=[dependent_variable] + independent_variables, inplace=True)
# print(f"Dataset shape after dropping NaNs: {df.shape}")

# # Separate features and target
# X = df[independent_variables]
# y = df[dependent_variable]

# # Identify numerical and categorical features
# numerical_features = [col for col in X.columns if col not in categorical_features]

# # Create a preprocessing pipeline for one-hot encoding
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#     ],
#     remainder='passthrough'
# )

# # Create the full model pipeline
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
# ])

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print("Training model...")
# # Train the model
# pipeline.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = pipeline.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse:.4f}")
# print(f"R-squared: {r2:.4f}")

# # Save the trained model and metrics
# MODEL_FILE = 'model.joblib'
# METRICS_FILE = 'metrics.json'

# joblib.dump(pipeline, MODEL_FILE)
# print(f"Model saved as '{MODEL_FILE}'.")

# metrics = {'mse': mse, 'r2_score': r2}
# with open(METRICS_FILE, 'w') as f:
#     json.dump(metrics, f, indent=4)
# print(f"Metrics saved as '{METRICS_FILE}'.")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import mlflow
import mlflow.sklearn
import os

# Fix hanging + authentication
os.environ['no_proxy'] = '*'
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/Shrilakshmi-NK/mlops-assignment.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'Shrilakshmi-NK'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'b9998ddddb6293b3f94dafb8e973620bc829ae2c'  # ← PASTE YOUR REAL TOKEN HERE

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
mlflow.set_experiment("cpu-usage-prediction")

print("Starting MLflow run...")

# THIS IS THE KEY: everything must be inside this block
with mlflow.start_run(run_name="RandomForest-v1"):
    
    # Your existing code (unchanged)
    DATASET_FILE = 'MLOps_assignment_dataset.csv'
    independent_variables = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes', 'controller_kind']
    dependent_variable = 'cpu_usage'
    categorical_features = ['controller_kind']

    print("Loading data...")
    df = pd.read_csv(DATASET_FILE)
    df.dropna(subset=[dependent_variable] + independent_variables, inplace=True)
    print(f"Dataset shape: {df.shape}")

    X = df[independent_variables]
    y = df[dependent_variable]

    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")

    # LOG EVERYTHING TO DAGSHUB
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.sklearn.log_model(pipeline, "model")

    # Save locally too (for DVC)
    joblib.dump(pipeline, 'model.joblib')
    with open('metrics.json', 'w') as f:
        json.dump({"mse": mse, "r2_score": r2}, f, indent=4)

    print("Experiment successfully logged to DagsHub!")

print("All done! Check your repo → Experiments tab")