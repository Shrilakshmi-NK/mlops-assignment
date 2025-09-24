import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

# Define the file name and variables
DATASET_FILE = 'MLOps_assignment_dataset.csv'
independent_variables = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes', 'controller_kind']
dependent_variable = 'cpu_usage'
categorical_features = ['controller_kind']

# Load the dataset
print("Loading data...")
try:
    df = pd.read_csv(DATASET_FILE)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{DATASET_FILE}' was not found. Please ensure it is in the same directory.")
    exit()

# Handle missing values if any
df.dropna(subset=[dependent_variable] + independent_variables, inplace=True)
print(f"Dataset shape after dropping NaNs: {df.shape}")

# Separate features and target
X = df[independent_variables]
y = df[dependent_variable]

# Identify numerical and categorical features
numerical_features = [col for col in X.columns if col not in categorical_features]

# Create a preprocessing pipeline for one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Create the full model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Save the trained model and metrics
MODEL_FILE = 'model.joblib'
METRICS_FILE = 'metrics.json'

joblib.dump(pipeline, MODEL_FILE)
print(f"Model saved as '{MODEL_FILE}'.")

metrics = {'mse': mse, 'r2_score': r2}
with open(METRICS_FILE, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"Metrics saved as '{METRICS_FILE}'.")