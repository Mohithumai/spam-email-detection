# spam_email_detection.py

# ----------------------------------------

# Spam Email Detection using Logistic Regression

# ----------------------------------------

import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
"""Load dataset"""
data = pd.read_csv(file_path, header=None)
print("Dataset loaded successfully!")
return data

def preprocess_data(data):
"""Split features and labels + scaling"""
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

```
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

return X_scaled, y, scaler
```

def train_model(X_train, y_train):
"""Train Logistic Regression model"""
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)
print("Model trained successfully!")
return model

def evaluate_model(model, X_test, y_test):
"""Evaluate model performance"""
y_pred = model.predict(X_test)

```
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {round(accuracy * 100, 2)}%")

return y_pred, accuracy
```

def plot_results(y_test, y_pred, accuracy):
"""Visualize results"""
cm = confusion_matrix(y_test, y_pred)

```
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Accuracy bar
plt.figure()
plt.bar(["Accuracy"], [accuracy * 100])
plt.ylim(0, 100)
plt.title("Model Accuracy (%)")
plt.ylabel("Accuracy")
plt.show()
```

def sample_predictions(model, X_test):
"""Predict random samples"""
print("\nSample Predictions:")
for i in random.sample(range(len(X_test)), 5):
sample = X_test[i].reshape(1, -1)
pred = model.predict(sample)[0]
print(f"Email {i}: {'SPAM' if pred == 1 else 'NOT SPAM'}")

def main():
# 🔹 Change file name if needed
file_path = "spambase.csv"

```
data = load_data(file_path)

X, y, scaler = preprocess_data(data)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = train_model(X_train, y_train)

y_pred, accuracy = evaluate_model(model, X_test, y_test)

plot_results(y_test, y_pred, accuracy)

sample_predictions(model, X_test)
```

if **name** == "**main**":
main()
