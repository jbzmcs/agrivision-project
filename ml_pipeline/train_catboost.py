import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import joblib
import time
import matplotlib.pyplot as plt

print("Loading extracted features...")
# 1. LOAD DATA: 
raw_data = pd.read_csv("train_features.csv")
raw_data['class_label'] = raw_data['class_label'].str.strip().str.replace(' ', '_')
print("Loaded train_features.csv\n")

# 2. DEFINE X and y: 
X = raw_data.iloc[:, :-1].values
y = raw_data.iloc[:, -1].values  

# 3. SPLIT DATA: 
# We use the same 70/15/15 split and random_state=42
X_train, X_others, y_train, y_others = train_test_split(
    X, y, train_size=0.7, stratify=y, random_state=42, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_others, y_others, test_size=0.5, stratify=y_others, random_state=42, shuffle=True
)

# 4. SCALE FEATURES: 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(f"Training set: {len(X_train)} | Validation set: {len(X_val)} | Test set: {len(X_test)}")

# 5. TRAIN MODEL: 
print("\nTraining CatBoost model...")
# iterations=500 is a good starting point
# logging_level='Silent' prevents 500 lines of text from flooding your terminal
model = CatBoostClassifier(iterations=500, random_seed=42, logging_level='Silent') 
start = time.time()
model.fit(X_train, y_train)
print(f"Training completed in {time.time() - start:.2f}s")

# 6. SAVE MODEL: 
joblib.dump(model, "models/catboost.pkl")  
print("Model saved as models/catboost.pkl")

# 7. EVALUATE MODEL: 
y_pred = model.predict(X_test)
print("\nClassification Report (CatBoost):\n") 
print(metrics.classification_report(y_test, y_pred, digits=4))
print(f"Overall Test Accuracy: {metrics.accuracy_score(y_test, y_pred) * 100:.2f}%")

# 8. VISUALIZE: 
print("Generating and saving confusion matrix...")
class_labels = model.classes_
# Create a figure with a specific size (e.g., 12x12 inches)
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, xticks_rotation=45, cmap='viridis')
plt.title("CatBoost Confusion Matrix") 
plt.tight_layout()
plt.show()