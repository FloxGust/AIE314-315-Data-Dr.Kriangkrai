import pandas as pd
from ucimlrepo import fetch_ucirepo

mushroom = fetch_ucirepo(id=73)
X = mushroom.data.features
y = mushroom.data.targets

# print(mushroom.metadata)
# print(mushroom.variables)

# print(X.head())           
# print(X.info())          
# print(y.value_counts())  


from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Encode categorical features
X_encoded = X.apply(LabelEncoder().fit_transform)

# Flatten target array
y = y.values.ravel()

# Define models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'k-NN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# Define metrics with pos_label specified
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted', pos_label='e'),  # Assuming 'e' as positive
    'recall': make_scorer(recall_score, average='weighted', pos_label='e')
}

# Perform 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# Store results without newline characters in keys
results = {model_name.strip(): {} for model_name in models}

for model_name, model in models.items():
    for metric_name, metric in scoring.items():
        scores = cross_val_score(model, X_encoded, y, cv=skf, scoring=metric)
        results[model_name.strip()][metric_name.strip()] = scores.mean()  # Strip whitespace from keys

# Confirm results structure
print("Cross-Validation Results:")
print(results)

# Proceed with bar plot visualization
metrics = ['accuracy', 'precision', 'recall']
values = {metric: [results[model][metric] for model in models] for metric in metrics}

# Plotting code follows here...









import matplotlib.pyplot as plt

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots()
for i, metric in enumerate(metrics):
    ax.bar(x + i*width, values[metric], width, label=metric)

ax.set_xlabel('Models')
ax.set_title('Model Comparison')
ax.set_xticks(x + width)
ax.set_xticklabels(models.keys())
ax.legend()
plt.show()
