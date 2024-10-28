import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Step 1: Load and prepare data
df = pd.read_csv('C:/Users/muhsina/OneDrive/Desktop/muhsina project/Sprint_3/data_mobile_price_range.csv')
df.columns = df.columns.str.strip()  # Strip any leading/trailing whitespace in column names

# Select specific features
selected_features = [
    'battery_power', 'int_memory', 'mobile_wt', 'n_cores', 'px_height', 'px_width', 'ram'
]
X = df[selected_features]
y = df['price_range']

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # Ensure balanced split
)

# Step 3: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Step 4: Set up GridSearchCV for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'linear'],
    'class_weight': ['balanced', None]
}

# Initialize SVM model
base_svm = SVC(probability=True, random_state=42)
grid_search = GridSearchCV(
    base_svm,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Step 5: Fit model using GridSearchCV
print("Performing grid search...")
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_
print("\nBest parameters:", grid_search.best_params_)

# Step 6: Evaluate model
print("\nTraining Performance:")
train_accuracy = best_model.score(X_train_scaled, y_train)
print(f"Training accuracy: {train_accuracy:.3f}")

print("\nTest Performance:")
y_pred = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy:.3f}")

# Cross-validation performance
def evaluate_model(model, X, y, title="Model Performance"):
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"\n{title} Cross-validation scores:", cv_scores)
    print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

evaluate_model(best_model, X_train_scaled, y_train, "Best Model")

# Step 7: Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Save the trained model
model_filename = 'svm_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)

# Prediction function for new data
def predict_price_range(model, scaler, new_data):
    scaled_data = scaler.transform(new_data)
    predictions = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)
    return predictions, probabilities

# Example of prediction on a sample
sample_data = X_test[:1]
predictions, probabilities = predict_price_range(best_model, scaler, sample_data)
print("\nSample Prediction:")
print(f"Predicted class: {predictions[0]}")
print(f"Class probabilities: {probabilities[0]}")
