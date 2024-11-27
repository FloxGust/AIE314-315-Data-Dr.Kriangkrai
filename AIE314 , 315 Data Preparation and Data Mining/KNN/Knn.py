print("___________________________________________________2_______________________________________________")
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np  


data = pd.read_csv("breast_cancer_bd.csv")

data['Bare Nuclei'] = pd.to_numeric(data['Bare Nuclei'], errors='coerce')
data_cleaned = data.dropna()

X = data_cleaned.drop(columns=['Sample code number', 'Class'])
y = data_cleaned['Class'].replace({2: 0, 4: 1})  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dt_model = DecisionTreeClassifier(random_state=42)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

dt_model.fit(X_scaled, y)

plt.figure(figsize=(15, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=['Benign', 'Malignant'], filled=True, rounded=True)
plt.title("Decision Tree Model")
plt.show()

dt_accuracy = np.mean(cross_val_score(dt_model, X_scaled, y, cv=kf, scoring='accuracy'))
dt_precision = np.mean(cross_val_score(dt_model, X_scaled, y, cv=kf, scoring='precision'))
dt_recall = np.mean(cross_val_score(dt_model, X_scaled, y, cv=kf, scoring='recall'))

k_range = range(1, 25)
knn_accuracies = []

for k in k_range:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    accuracy = np.mean(cross_val_score(knn_model, X_scaled, y, cv=kf, scoring='accuracy'))
    knn_accuracies.append(accuracy)

plt.figure(figsize=(10, 6))
plt.plot(k_range, knn_accuracies, marker='o')
plt.title("Finding Optimal K for KNN")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

optimal_k = 11

knn_model = KNeighborsClassifier(n_neighbors=optimal_k)

knn_accuracy = np.mean(cross_val_score(knn_model, X_scaled, y, cv=kf, scoring='accuracy'))
knn_precision = np.mean(cross_val_score(knn_model, X_scaled, y, cv=kf, scoring='precision'))
knn_recall = np.mean(cross_val_score(knn_model, X_scaled, y, cv=kf, scoring='recall'))

print(f"Decision Tree \nAccuracy: {dt_accuracy:.2f}, \nPrecision: {dt_precision:.2f}, \nRecall: {dt_recall:.2f}")
print(f"\nK-NN (K={optimal_k}) \nAccuracy: {knn_accuracy:.2f}, \nPrecision: {knn_precision:.2f},\nRecall: {knn_recall:.2f}")


