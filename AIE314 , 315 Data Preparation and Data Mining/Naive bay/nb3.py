import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score

mushroom = fetch_ucirepo(id=73)

X = mushroom.data.features
y = mushroom.data.targets.values.ravel()  # Flatten y to 1D array

X_encoded = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
    'k-NN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

results = {}

for name, clf in classifiers.items():
    scores = cross_val_score(clf, X_encoded, y, cv=10, scoring='accuracy')
    accuracy = np.mean(scores)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }


for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")


labels = list(results.keys())
accuracy_scores = [metrics['accuracy'] for metrics in results.values()]
precision_scores = [metrics['precision'] for metrics in results.values()]
recall_scores = [metrics['recall'] for metrics in results.values()]


plt.figure(figsize=(15, 6))


plt.subplot(1, 3, 1)
plt.bar(labels, accuracy_scores, color='pink')
plt.ylabel('Accuracy Scores')
plt.title('Accuracy')
plt.grid()

plt.subplot(1, 3, 2)
plt.bar(labels, precision_scores, color='green')
plt.ylabel('Precision Scores')
plt.title('Precision')
plt.grid()

plt.subplot(1, 3, 3)
plt.bar(labels, recall_scores, color='orange')
plt.ylabel('Recall Scores')
plt.title('Recall')
plt.grid()

plt.tight_layout()
plt.show()
