import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import joblib
import time
import matplotlib.pyplot as plt

print("Loading extracted features...")
raw_data = pd.read_csv("train_features.csv")
raw_data['class_label'] = raw_data['class_label'].str.strip().str.replace(' ', '_')
print("Loaded train_features.csv\n")

X = raw_data.iloc[:, :-1].values
y = raw_data.iloc[:, -1].values

X_train, X_others, y_train, y_others = train_test_split(
    X, y, train_size=0.7, stratify=y, random_state=42, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_others, y_others, test_size=0.5, stratify=y_others, random_state=42, shuffle=True
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(f"Training set: {len(X_train)} | Validation set: {len(X_val)} | Test set: {len(X_test)}")

print("\nTraining KNN model...")
model = KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='distance')
start = time.time()
model.fit(X_train, y_train)
print(f"Training completed in {time.time() - start:.2f}s")

joblib.dump(model, "KNN_Tomato.pkl")
print("Model saved as KNN_Tomato.pkl")

y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(metrics.classification_report(y_test, y_pred))
print(f"Overall Test Accuracy: {metrics.accuracy_score(y_test, y_pred) * 100:.2f}%")

metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=45)
plt.title("KNN Confusion Matrix")
plt.tight_layout()
plt.show()
