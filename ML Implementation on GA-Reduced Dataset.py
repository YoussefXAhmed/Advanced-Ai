import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

columns = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
           "exang","oldpeak","slope","ca","thal","target"]
df = pd.read_csv("heart.csv", names=columns, na_values='?')
df = df.dropna()

X = df.drop("target", axis=1)
y = df["target"].apply(lambda v: 1 if v > 0 else 0)

X = pd.get_dummies(X, columns=["cp","restecg","slope","thal"], drop_first=True)

selected_features = ['age', 'cp_2', 'cp_3', 'thalach', 'exang', 'oldpeak', 'slope_2', 'ca', 'thal_7']
X_reduced = X[selected_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("KNN Performance on GA-Reduced Dataset:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
