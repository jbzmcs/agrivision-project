import pandas as pd
import joblib
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

df = pd.read_csv('train_features.csv')
print("Loaded train_features.csv into a DataFrame.")
print(df.head())

df['class_label'] = df['class_label'].str.strip()

X = df[['severity_percent', 'spot_count']]
y = df['class_label']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\nTraining set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("\nTraining Support Vector Machine (SVM)...")
start_time = time.time()
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
training_time = time.time() - start_time

joblib.dump(svm_model, 'SVM_Tomato.pkl')
print("Model training complete and saved as 'SVM_Tomato.pkl'")

y_pred = svm_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)

print("\n" + "="*60)
print("Classification Report:")
print("="*60)
print(metrics.classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print(f"\nOverall Test Accuracy: {accuracy * 100:.2f}%")
print(f"Training Time: {training_time:.2f} seconds")

cm = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('SVM Confusion Matrix')
plt.tight_layout()
plt.show()
